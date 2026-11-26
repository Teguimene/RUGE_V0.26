import argparse
from pathlib import Path

import torch

from .data import load_groundings, load_triples
from .models import ComplEx, RotatE
from .trainer import RUGETrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PyTorch RUGE trainer for ComplEx and RotatE."
    )
    parser.add_argument("--model", choices=("complex", "rotate"), required=True)
    parser.add_argument("-train", required=True, help="Numeric training triples")
    parser.add_argument("-valid", required=True, help="Numeric validation triples")
    parser.add_argument("-test", required=True, help="Numeric test triples")
    parser.add_argument("-rule", required=True, help="Grounded soft rules")
    parser.add_argument("-m", type=int, required=True, help="Number of relations")
    parser.add_argument("-n", type=int, required=True, help="Number of entities")
    parser.add_argument("-k", type=int, help="Embedding dimension")
    parser.add_argument("-d", type=float, help="L2 regularization coefficient")
    parser.add_argument("-c", type=float, help="RUGE rule confidence weight")
    parser.add_argument("-ne", type=int, help="Negative triples per positive")
    parser.add_argument("-ge", type=float, help="Entity learning rate")
    parser.add_argument("-gr", type=float, help="Relation learning rate")
    parser.add_argument("-#", dest="iterations", type=int, help="Training epochs")
    parser.add_argument("-skip", type=int, help="Epochs between evaluations")
    parser.add_argument("-eval", type=int, help="Validation triples per evaluation")
    parser.add_argument("-gamma", type=float, help="RotatE score margin")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--output", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    defaults = {
        "complex": {
            "dimension": 20,
            "regularization": 0.01,
            "rule_weight": 0.01,
            "negatives": 10,
            "gamma_entity": 0.01,
            "gamma_relation": 1.0,
            "iterations": 1000,
            "skip": 10,
            "validation_limit": 1000,
        },
        "rotate": {
            "dimension": 200,
            "regularization": 0.01,
            "rule_weight": 0.01,
            "negatives": 10,
            "gamma_entity": 0.5,
            "gamma_relation": 0.5,
            "iterations": 1000,
            "skip": 50,
            "validation_limit": 10,
        },
    }[args.model]

    device = (
        torch.device("cuda")
        if args.device == "auto" and torch.cuda.is_available()
        else torch.device("cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = torch.float32 if args.precision == "float32" else torch.float64
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    training = load_triples(args.train, args.n, args.m)
    validation = load_triples(args.valid, args.n, args.m)
    testing = load_triples(args.test, args.n, args.m)
    rules = load_groundings(args.rule, args.n, args.m)
    known_triples = {
        tuple(row)
        for split in (training, validation, testing)
        for row in split.tolist()
    }

    dimension = args.k if args.k is not None else defaults["dimension"]
    if args.model == "complex":
        model = ComplEx(
            args.n, args.m, dimension, device=device, dtype=dtype, seed=args.seed
        )
    else:
        model = RotatE(
            args.n,
            args.m,
            dimension,
            margin=12.0 if args.gamma is None else args.gamma,
            device=device,
            dtype=dtype,
            seed=args.seed,
        )

    trainer = RUGETrainer(
        model,
        training,
        validation,
        known_triples,
        rules,
        num_negatives=args.ne if args.ne is not None else defaults["negatives"],
        num_iterations=(
            args.iterations
            if args.iterations is not None
            else defaults["iterations"]
        ),
        output_iter_skip=args.skip if args.skip is not None else defaults["skip"],
        validation_limit=(
            args.eval if args.eval is not None else defaults["validation_limit"]
        ),
        regularization=(
            args.d if args.d is not None else defaults["regularization"]
        ),
        rule_weight=args.c if args.c is not None else defaults["rule_weight"],
        gamma_entity=(
            args.ge if args.ge is not None else defaults["gamma_entity"]
        ),
        gamma_relation=(
            args.gr if args.gr is not None else defaults["gamma_relation"]
        ),
        seed=args.seed,
        output_dir=args.output,
    )

    print(
        f"Model={args.model}, device={device}, precision={args.precision}, "
        f"training triples={len(training)}, rules={len(rules)}"
    )
    metrics = trainer.fit()
    print(
        f"Best validation: MRR={metrics.mrr:.6f}, "
        f"Hits@10={metrics.hits_at_10:.6f}, "
        f"mean rank={metrics.mean_rank:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
