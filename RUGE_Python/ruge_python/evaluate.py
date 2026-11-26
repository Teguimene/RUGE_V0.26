import argparse
from pathlib import Path

import torch

from .data import load_triples
from .io import load_java_matrices
from .metrics import filtered_ranking_metrics
from .models import ComplEx, RotatE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a RUGE model on triples")
    parser.add_argument("--model", choices=("complex", "rotate"), required=True)
    parser.add_argument("-train", required=True)
    parser.add_argument("-valid", required=True)
    parser.add_argument("-test", required=True)
    parser.add_argument("-m", type=int, required=True)
    parser.add_argument("-n", type=int, required=True)
    parser.add_argument("-k", type=int, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--matrix-dir", type=Path)
    parser.add_argument("-gamma", type=float, default=12.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = (
        torch.device("cuda")
        if args.device == "auto" and torch.cuda.is_available()
        else torch.device("cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    training = load_triples(args.train, args.n, args.m)
    validation = load_triples(args.valid, args.n, args.m)
    testing = load_triples(args.test, args.n, args.m)
    known = {
        tuple(row)
        for split in (training, validation, testing)
        for row in split.tolist()
    }

    if args.checkpoint is not None:
        checkpoint = torch.load(
            args.checkpoint, map_location=device, weights_only=False
        )
        dimension = checkpoint["dimension"]
        if args.model == "complex":
            model = ComplEx(
                args.n,
                args.m,
                dimension,
                device=device,
                dtype=torch.float64,
            )
        else:
            margin = checkpoint.get("margin") or args.gamma
            model = RotatE(
                args.n,
                args.m,
                dimension,
                margin=margin,
                device=device,
                dtype=torch.float64,
            )
        model.load_state_dict(checkpoint["model"])
    else:
        if args.model == "complex":
            model = ComplEx(
                args.n, args.m, args.k, device=device, dtype=torch.float64
            )
        else:
            model = RotatE(
                args.n,
                args.m,
                args.k,
                margin=args.gamma,
                device=device,
                dtype=torch.float64,
            )
        load_java_matrices(model, args.matrix_dir)

    metrics = filtered_ranking_metrics(model, testing, known)
    print(
        f"Filtered test: MRR={metrics.mrr:.6f}, "
        f"Hits@10={metrics.hits_at_10:.6f}, "
        f"mean rank={metrics.mean_rank:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
