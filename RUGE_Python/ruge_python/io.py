from pathlib import Path

import numpy as np
import torch

from .models import RUGEEmbeddingModel


def save_java_matrices(model: RUGEEmbeddingModel, directory: str | Path) -> None:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, parameter in model.export_matrices().items():
        values = parameter.detach().to(device="cpu", dtype=torch.float64).numpy()
        path = output_dir / filename
        with path.open("w", encoding="utf-8", newline="\n") as output:
            output.write(
                f"iNumberOfRows: {values.shape[0]}; "
                f"iNumberOfColumns: {values.shape[1]}\n"
            )
            np.savetxt(output, values, delimiter="\t", fmt="%.17g")


def save_checkpoint(
    model: RUGEEmbeddingModel,
    accumulators: list[torch.Tensor],
    epoch: int,
    best_mrr: float,
    path: str | Path,
) -> None:
    torch.save(
        {
            "model": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "accumulators": [value.detach().cpu() for value in accumulators],
            "epoch": epoch,
            "best_mrr": best_mrr,
            "num_entities": model.num_entities,
            "num_relations": model.num_relations,
            "dimension": model.dimension,
            "model_type": model.__class__.__name__,
            "margin": getattr(model, "margin", None),
        },
        Path(path),
    )


def load_java_matrices(
    model: RUGEEmbeddingModel, directory: str | Path
) -> None:
    input_dir = Path(directory)
    with torch.no_grad():
        for filename, parameter in model.export_matrices().items():
            path = input_dir / filename
            with path.open("r", encoding="utf-8") as source:
                header = source.readline().strip()
            expected_header = (
                f"iNumberOfRows: {parameter.shape[0]}; "
                f"iNumberOfColumns: {parameter.shape[1]}"
            )
            if header != expected_header:
                raise ValueError(f"{path}: expected header {expected_header!r}")
            values = np.loadtxt(path, delimiter=None, skiprows=1, ndmin=2)
            if values.shape != tuple(parameter.shape):
                raise ValueError(
                    f"{path}: expected matrix shape {tuple(parameter.shape)}, "
                    f"got {values.shape}"
                )
            parameter.copy_(
                torch.as_tensor(
                    values, dtype=parameter.dtype, device=parameter.device
                )
            )
