import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .data import load_triples, read_id_map
from .models import ComplEx, RUGEEmbeddingModel, RotatE


def _load_checkpoint(path: Path, device: torch.device) -> RUGEEmbeddingModel:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_types = {"ComplEx": ComplEx, "RotatE": RotatE}
    model_type = checkpoint.get("model_type")
    if model_type not in model_types:
        raise ValueError(f"{path}: unsupported or missing checkpoint model type")

    model_class = model_types[model_type]
    state = checkpoint["model"]
    model_args = {
        "num_entities": checkpoint["num_entities"],
        "num_relations": checkpoint["num_relations"],
        "dimension": checkpoint["dimension"],
        "device": device,
        "dtype": state["entity_real"].dtype,
    }
    if model_class is RotatE:
        model = RotatE(
            **model_args,
            margin=checkpoint.get("margin") or 12.0,
        )
    else:
        model = ComplEx(**model_args)
    model.load_state_dict(state)
    model.eval()
    return model


def rank_candidates(
    scores: torch.Tensor,
    blocked_ids: set[int],
    top_k: int,
) -> list[tuple[int, float]]:
    if scores.ndim != 1:
        raise ValueError("scores must be a one-dimensional tensor")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    filtered_scores = scores.detach().cpu().clone()
    if blocked_ids:
        blocked = torch.tensor(sorted(blocked_ids), dtype=torch.long)
        filtered_scores[blocked] = -torch.inf

    available = int(torch.isfinite(filtered_scores).sum().item())
    count = min(top_k, available)
    if count == 0:
        return []
    values, indices = torch.topk(filtered_scores, count)
    return [(int(index), float(value)) for index, value in zip(indices, values)]


def rank_candidate(
    scores: torch.Tensor,
    candidate_id: int,
    blocked_ids: set[int],
) -> tuple[int, float]:
    if scores.ndim != 1:
        raise ValueError("scores must be a one-dimensional tensor")
    if not 0 <= candidate_id < scores.shape[0]:
        raise ValueError("candidate_id is out of range")
    if candidate_id in blocked_ids:
        raise ValueError("candidate is already a known training fact")

    candidate_score = scores[candidate_id]
    filtered_scores = scores.detach().cpu().clone()
    if blocked_ids:
        blocked = torch.tensor(sorted(blocked_ids), dtype=torch.long)
        filtered_scores[blocked] = -torch.inf
    rank = 1 + int((filtered_scores > candidate_score.cpu()).sum().item())
    return rank, float(candidate_score.item())


def _score_candidates(
    model: RUGEEmbeddingModel,
    relation_id: int,
    fixed_entity_id: int,
    predict_subject: bool,
    batch_size: int = 4096,
) -> torch.Tensor:
    scores = []
    with torch.no_grad():
        for start in tqdm(
            range(0, model.num_entities, batch_size),
            desc="Scoring candidates",
            unit="batch",
        ):
            candidate_ids = torch.arange(
                start,
                min(start + batch_size, model.num_entities),
                device=model.device,
            )
            relations = torch.full_like(candidate_ids, relation_id)
            fixed_entities = torch.full_like(candidate_ids, fixed_entity_id)
            if predict_subject:
                triples = torch.stack(
                    (candidate_ids, relations, fixed_entities),
                    dim=1,
                )
            else:
                triples = torch.stack(
                    (fixed_entities, relations, candidate_ids),
                    dim=1,
                )
            scores.append(model.score(triples).cpu())
    return torch.cat(scores)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank entity predictions from a trained RUGE checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--entity-map", type=Path, required=True)
    parser.add_argument("--relation-map", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument(
        "--subject",
        help="Known subject; rank candidate objects for this subject and relation",
    )
    query.add_argument(
        "--object",
        help="Known object; rank candidate subjects for this relation and object",
    )
    parser.add_argument("--relation", required=True)
    parser.add_argument(
        "--candidate",
        help="Also report the filtered rank and score of this candidate entity",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    device = (
        torch.device("cuda")
        if args.device == "auto" and torch.cuda.is_available()
        else torch.device("cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model = _load_checkpoint(args.checkpoint, device)
    entity_ids = read_id_map(args.entity_map)
    entities_by_id = {index: name for name, index in entity_ids.items()}
    relation_ids = read_id_map(args.relation_map)
    relations_by_id = {index: name for name, index in relation_ids.items()}
    if len(entities_by_id) != model.num_entities:
        raise ValueError("Entity map size does not match the checkpoint")
    if len(relations_by_id) != model.num_relations:
        raise ValueError("Relation map size does not match the checkpoint")
    if args.relation not in relation_ids:
        raise ValueError(f"Unknown relation: {args.relation}")
    if args.candidate is not None and args.candidate not in entity_ids:
        raise ValueError(f"Unknown candidate: {args.candidate}")

    training = load_triples(
        args.train,
        num_entities=model.num_entities,
        num_relations=model.num_relations,
    )
    relation_id = relation_ids[args.relation]
    blocked_ids: set[int] = set()

    if args.subject is not None:
        if args.subject not in entity_ids:
            raise ValueError(f"Unknown subject: {args.subject}")
        fixed_id = entity_ids[args.subject]
        blocked_ids = {
            tail
            for head, relation, tail in training.tolist()
            if head == fixed_id and relation == relation_id
        }
    else:
        if args.object not in entity_ids:
            raise ValueError(f"Unknown object: {args.object}")
        fixed_id = entity_ids[args.object]
        blocked_ids = {
            head
            for head, relation, tail in training.tolist()
            if relation == relation_id and tail == fixed_id
        }

    scores = _score_candidates(
        model,
        relation_id,
        fixed_id,
        predict_subject=args.object is not None,
    )
    if args.candidate is not None:
        candidate_id = entity_ids[args.candidate]
        candidate_rank, candidate_score = rank_candidate(
            scores, candidate_id, blocked_ids
        )
        if args.subject is not None:
            candidate_triple = (
                f"{args.subject}\t{args.relation}\t{args.candidate}"
            )
        else:
            candidate_triple = (
                f"{args.candidate}\t{args.relation}\t{args.object}"
            )
        print(
            f"Candidate rank={candidate_rank}, score={candidate_score:.8f}\t"
            f"{candidate_triple}"
        )

    predictions = rank_candidates(scores, blocked_ids, args.top_k)
    if not predictions:
        print("No unseen candidates remain after filtering training facts.")
        return 0

    relation_name = relations_by_id[relation_id]
    print("Rank\tScore\tPredicted triple")
    for rank, (entity_id, score) in enumerate(predictions, start=1):
        candidate_name = entities_by_id[entity_id]
        if args.subject is not None:
            triple = f"{args.subject}\t{relation_name}\t{candidate_name}"
        else:
            triple = f"{candidate_name}\t{relation_name}\t{args.object}"
        print(f"{rank}\t{score:.8f}\t{triple}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
