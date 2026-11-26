from collections import defaultdict
from dataclasses import dataclass

import torch

from .models import RUGEEmbeddingModel


@dataclass(frozen=True)
class RankingMetrics:
    mrr: float
    hits_at_10: float
    mean_rank: float
    ranked_queries: int


def filtered_ranking_metrics(
    model: RUGEEmbeddingModel,
    validation: torch.Tensor,
    known_triples: set[tuple[int, int, int]],
    maximum_triples: int | None = None,
) -> RankingMetrics:
    if maximum_triples is not None:
        validation = validation[:maximum_triples]
    if validation.numel() == 0:
        raise ValueError("Cannot evaluate an empty validation set")

    known_heads: dict[tuple[int, int], set[int]] = defaultdict(set)
    known_tails: dict[tuple[int, int], set[int]] = defaultdict(set)
    for head, relation, tail in known_triples:
        known_heads[(relation, tail)].add(head)
        known_tails[(head, relation)].add(tail)

    reciprocal_rank_sum = 0.0
    hits_at_10_sum = 0
    rank_sum = 0
    rank_count = 0
    entities = torch.arange(model.num_entities, device=model.device)

    model.eval()
    with torch.no_grad():
        for head, relation, tail in validation.tolist():
            target = torch.tensor(
                [[head, relation, tail]], dtype=torch.long, device=model.device
            )
            target_score = model.score(target)[0]

            head_candidates = torch.stack(
                (
                    entities,
                    torch.full_like(entities, relation),
                    torch.full_like(entities, tail),
                ),
                dim=1,
            )
            head_scores = model.score(head_candidates)
            blocked_heads = known_heads.get((relation, tail), ())
            if blocked_heads:
                indices = torch.tensor(
                    tuple(blocked_heads), dtype=torch.long, device=model.device
                )
                head_scores[indices] = -torch.inf
            head_rank = 1 + int((head_scores > target_score).sum().item())

            tail_candidates = torch.stack(
                (
                    torch.full_like(entities, head),
                    torch.full_like(entities, relation),
                    entities,
                ),
                dim=1,
            )
            tail_scores = model.score(tail_candidates)
            blocked_tails = known_tails.get((head, relation), ())
            if blocked_tails:
                indices = torch.tensor(
                    tuple(blocked_tails), dtype=torch.long, device=model.device
                )
                tail_scores[indices] = -torch.inf
            tail_rank = 1 + int((tail_scores > target_score).sum().item())

            for rank in (head_rank, tail_rank):
                reciprocal_rank_sum += 1.0 / rank
                hits_at_10_sum += int(rank <= 10)
                rank_sum += rank
                rank_count += 1

    return RankingMetrics(
        mrr=reciprocal_rank_sum / rank_count,
        hits_at_10=hits_at_10_sum / rank_count,
        mean_rank=rank_sum / rank_count,
        ranked_queries=rank_count,
    )
