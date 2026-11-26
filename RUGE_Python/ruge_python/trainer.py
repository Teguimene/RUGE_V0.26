import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.autograd import Function

from .data import GroundRule
from .io import save_checkpoint, save_java_matrices
from .metrics import RankingMetrics, filtered_ranking_metrics
from .models import RUGEEmbeddingModel, RotatE


def _java_sigmoid(values: torch.Tensor) -> torch.Tensor:
    middle = torch.sigmoid(values.clamp(min=-10.0, max=10.0))
    return torch.where(values > 10.0, 1.0, torch.where(values < -10.0, 0.0, middle))


class _JavaLogisticLoss(Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(scores, labels)
        return F.softplus(scores) - labels * scores

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        scores, labels = ctx.saved_tensors
        score_gradient = (_java_sigmoid(scores) - labels) * output_gradient
        return score_gradient, None


class RUGETrainer:
    def __init__(
        self,
        model: RUGEEmbeddingModel,
        training: torch.Tensor,
        validation: torch.Tensor,
        known_triples: set[tuple[int, int, int]],
        rules: list[GroundRule],
        *,
        num_negatives: int = 10,
        num_minibatches: int = 100,
        num_iterations: int = 1000,
        output_iter_skip: int = 50,
        validation_limit: int | None = 1000,
        regularization: float = 0.01,
        rule_weight: float = 0.01,
        gamma_entity: float = 0.5,
        gamma_relation: float = 0.5,
        seed: int = 123,
        output_dir: str | Path = ".",
    ) -> None:
        if num_negatives < 2 or num_minibatches <= 0 or num_iterations < 0:
            raise ValueError("Invalid negative, mini-batch, or iteration count")
        if output_iter_skip <= 0 or regularization < 0.0 or rule_weight < 0.0:
            raise ValueError("Invalid validation, regularization, or rule setting")
        if num_negatives % 2:
            raise ValueError("num_negatives must be even, as in the Java trainer")

        self.model = model
        self.training = training.cpu()
        self.validation = validation.cpu()
        self.known_triples = known_triples
        self.rules = rules
        self.num_negatives = num_negatives
        self.num_minibatches = num_minibatches
        self.num_iterations = num_iterations
        self.output_iter_skip = output_iter_skip
        self.validation_limit = validation_limit
        self.regularization = regularization
        self.rule_weight = rule_weight
        self.gamma_entity = gamma_entity
        self.gamma_relation = gamma_relation
        self.output_dir = Path(output_dir)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.parameters = list(model.parameters())
        self.accumulators = [torch.zeros_like(value) for value in self.parameters]
        self.learning_rates = [
            gamma_relation if name.startswith("relation") else gamma_entity
            for name, _ in model.named_parameters()
        ]
        self.rule_triples, self.rule_targets, self.rule_unseen, self.rule_confidence = (
            self._encode_rules(rules)
        )

    @staticmethod
    def _encode_rules(
        rules: list[GroundRule],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        triples = torch.zeros((len(rules), 4, 3), dtype=torch.long)
        targets = torch.zeros((len(rules), 3), dtype=torch.long)
        unseen = torch.empty(len(rules), dtype=torch.long)
        confidence = torch.empty(len(rules), dtype=torch.float64)
        for index, rule in enumerate(rules):
            triples[index, : len(rule.triples)] = torch.tensor(rule.triples)
            targets[index] = torch.tensor(rule.target)
            unseen[index] = rule.unseen
            confidence[index] = rule.confidence
        return triples, targets, unseen, confidence

    def _sample_replacements(
        self, original_ids: torch.Tensor, count: int
    ) -> torch.Tensor:
        original_ids = original_ids.cpu()
        batch_size = original_ids.shape[0]
        result = torch.empty((batch_size, count), dtype=torch.long)
        for column in range(count):
            draws = torch.randint(
                self.model.num_entities,
                (batch_size,),
                generator=self.generator,
            )
            invalid = draws.eq(original_ids)
            if column:
                invalid |= (result[:, :column] == draws[:, None]).any(dim=1)
            while invalid.any():
                draws[invalid] = torch.randint(
                    self.model.num_entities,
                    (int(invalid.sum()),),
                    generator=self.generator,
                )
                invalid = draws.eq(original_ids)
                if column:
                    invalid |= (result[:, :column] == draws[:, None]).any(dim=1)
            result[:, column] = draws
        return result

    def _make_supervised_batch(
        self, positives_cpu: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positives = positives_cpu.to(self.model.device)
        negatives_per_side = self.num_negatives // 2
        negative_heads = self._sample_replacements(
            positives_cpu[:, 0], negatives_per_side
        )
        negative_tails = self._sample_replacements(
            positives_cpu[:, 2], negatives_per_side
        )

        head_negatives = torch.stack(
            (
                negative_heads.reshape(-1),
                positives_cpu[:, 1].repeat_interleave(negatives_per_side),
                positives_cpu[:, 2].repeat_interleave(negatives_per_side),
            ),
            dim=1,
        )
        tail_negatives = torch.stack(
            (
                positives_cpu[:, 0].repeat_interleave(negatives_per_side),
                positives_cpu[:, 1].repeat_interleave(negatives_per_side),
                negative_tails.reshape(-1),
            ),
            dim=1,
        )
        triples = torch.cat(
            (positives_cpu, head_negatives, tail_negatives), dim=0
        ).to(self.model.device)
        labels = torch.cat(
            (
                torch.ones(len(positives_cpu), dtype=self.model.dtype),
                torch.zeros(
                    len(head_negatives) + len(tail_negatives),
                    dtype=self.model.dtype,
                ),
            )
        ).to(self.model.device)
        return triples, labels

    def _triple_loss(
        self, triples: torch.Tensor, labels: torch.Tensor, denominator: int
    ) -> torch.Tensor:
        scores = self.model.score(triples)
        logistic = _JavaLogisticLoss.apply(scores, labels)
        regularization = self.model.regularization(triples)
        return (
            logistic + (self.regularization / self.model.dimension) * regularization
        ).sum() / denominator

    def _rule_loss(self, rule_indices: torch.Tensor) -> torch.Tensor | None:
        if rule_indices.numel() == 0:
            return None

        device = self.model.device
        selected = rule_indices.cpu()
        triples = self.rule_triples[selected].to(device)
        targets = self.rule_targets[selected].to(device)
        unseen = self.rule_unseen[selected].to(device)
        confidence = self.rule_confidence[selected].to(
            device=device, dtype=self.model.dtype
        )

        with torch.no_grad():
            probabilities = torch.stack(
                [
                    _java_sigmoid(self.model.score(triples[:, index, :]))
                    for index in range(3)
                ],
                dim=1,
            )
            posterior = torch.zeros_like(confidence)
            mask = unseen == 1
            posterior[mask] = (
                self.rule_weight
                * confidence[mask]
                * (probabilities[mask, 1] - 1.0)
            )
            mask = unseen == 2
            posterior[mask] = (
                self.rule_weight * confidence[mask] * probabilities[mask, 0]
            )
            mask = unseen == 3
            posterior[mask] = (
                self.rule_weight
                * confidence[mask]
                * probabilities[mask, 0]
                * probabilities[mask, 1]
            )
            mask = unseen == 4
            posterior[mask] = (
                self.rule_weight
                * confidence[mask]
                * probabilities[mask, 0]
                * probabilities[mask, 1]
                * probabilities[mask, 2]
            )

            unique_targets, inverse = torch.unique(
                targets, dim=0, return_inverse=True
            )
            summed_posterior = torch.zeros(
                unique_targets.shape[0], dtype=self.model.dtype, device=device
            )
            summed_posterior.index_add_(0, inverse, posterior)
            labels = (
                _java_sigmoid(self.model.score(unique_targets)) + summed_posterior
            ).clamp_(0.0, 1.0)

        return self._triple_loss(unique_targets, labels, len(unique_targets))

    def _update_parameters(self) -> None:
        with torch.no_grad():
            for parameter, accumulator, learning_rate in zip(
                self.parameters, self.accumulators, self.learning_rates
            ):
                gradient = parameter.grad
                row_norm = torch.linalg.vector_norm(gradient, dim=1, keepdim=True)
                gradient.mul_(torch.clamp(1.0 / row_norm.clamp_min(1e-12), max=1.0))
                accumulator.addcmul_(gradient, gradient)
                parameter.addcdiv_(
                    gradient,
                    accumulator.sqrt() + 1e-8,
                    value=-learning_rate,
                )
        if isinstance(self.model, RotatE):
            self.model.wrap_phases_()

    def _evaluate(self) -> RankingMetrics:
        return filtered_ranking_metrics(
            self.model,
            self.validation,
            self.known_triples,
            self.validation_limit,
        )

    def fit(self) -> RankingMetrics:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.output_dir / "ruge_python_best.pt"
        log_path = self.output_dir / "result.log"
        initial = self._evaluate()
        best_mrr = initial.mrr
        best_metrics = initial
        best_epoch = 0
        save_java_matrices(self.model, self.output_dir)
        save_checkpoint(
            self.model, self.accumulators, 0, best_mrr, checkpoint_path
        )

        with log_path.open("w", encoding="utf-8") as log:
            log.write(
                f"Initial validation: MRR={initial.mrr}\t"
                f"Hits@10={initial.hits_at_10}\n"
            )
            log.flush()
            print(
                f"Initial validation: MRR={initial.mrr:.6f}, "
                f"Hits@10={initial.hits_at_10:.6f}"
            )

            for epoch in range(1, self.num_iterations + 1):
                train_order = torch.randperm(
                    len(self.training), generator=self.generator
                )
                rule_order = torch.randperm(
                    len(self.rules), generator=self.generator
                )
                epoch_loss = 0.0

                for batch_id in range(self.num_minibatches):
                    train_indices = train_order[batch_id::self.num_minibatches]
                    if train_indices.numel() == 0:
                        continue

                    positives = self.training[train_indices]
                    triples, labels = self._make_supervised_batch(positives)
                    denominator = len(triples)
                    loss = self._triple_loss(triples, labels, denominator)

                    rule_indices = rule_order[batch_id::self.num_minibatches]
                    rule_loss = self._rule_loss(rule_indices)
                    if rule_loss is not None:
                        loss = loss + rule_loss

                    for parameter in self.parameters:
                        parameter.grad = None
                    loss.backward()
                    self._update_parameters()
                    epoch_loss += loss.detach().item()

                if epoch % self.output_iter_skip == 0 or epoch == self.num_iterations:
                    metrics = self._evaluate()
                    print(
                        f"Epoch {epoch}/{self.num_iterations}: "
                        f"loss={epoch_loss:.6f}, MRR={metrics.mrr:.6f}, "
                        f"Hits@10={metrics.hits_at_10:.6f}"
                    )
                    log.write(
                        f"Epoch {epoch}: loss={epoch_loss}\tMRR={metrics.mrr}\t"
                        f"Hits@10={metrics.hits_at_10}\n"
                    )
                    if metrics.mrr > best_mrr:
                        best_mrr = metrics.mrr
                        best_metrics = metrics
                        best_epoch = epoch
                        save_java_matrices(self.model, self.output_dir)
                        save_checkpoint(
                            self.model,
                            self.accumulators,
                            epoch,
                            best_mrr,
                            checkpoint_path,
                        )
                    log.write(
                        f"Best epoch {best_epoch}: MRR={best_mrr}\t"
                        f"Hits@10={best_metrics.hits_at_10}\n"
                    )
                    log.flush()

        return best_metrics
