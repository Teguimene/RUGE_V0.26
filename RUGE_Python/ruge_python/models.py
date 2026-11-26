import math
from abc import ABC, abstractmethod

import torch
from torch import nn


class RUGEEmbeddingModel(nn.Module, ABC):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dimension: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if min(num_entities, num_relations, dimension) <= 0:
            raise ValueError("Entity, relation, and embedding counts must be positive")
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dimension = dimension
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def score(self, triples: torch.Tensor) -> torch.Tensor:
        """Return one plausibility score per (head, relation, tail) triple."""

    @abstractmethod
    def regularization(self, triples: torch.Tensor) -> torch.Tensor:
        """Return the unscaled per-triple L2 regularization term."""

    @abstractmethod
    def export_matrices(self) -> dict[str, torch.Tensor]:
        """Return the Java-compatible embedding matrix names and values."""


class ComplEx(RUGEEmbeddingModel):
    """ComplEx score and parameter layout used by Java RUGE."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dimension: int,
        device: torch.device,
        dtype: torch.dtype,
        seed: int = 123,
    ) -> None:
        super().__init__(num_entities, num_relations, dimension, device, dtype)
        self.entity_real = nn.Parameter(
            self._gaussian((num_entities, dimension), seed)
        )
        self.relation_real = nn.Parameter(
            self._gaussian((num_relations, dimension), seed)
        )
        self.entity_imag = nn.Parameter(
            self._gaussian((num_entities, dimension), seed)
        )
        self.relation_imag = nn.Parameter(
            self._gaussian((num_relations, dimension), seed)
        )

    def _gaussian(self, shape: tuple[int, int], seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return torch.randn(shape, generator=generator, dtype=self.dtype).to(self.device)

    def score(self, triples: torch.Tensor) -> torch.Tensor:
        head, relation, tail = triples.long().unbind(dim=1)
        eh_r = self.entity_real[head]
        eh_i = self.entity_imag[head]
        er_r = self.relation_real[relation]
        er_i = self.relation_imag[relation]
        et_r = self.entity_real[tail]
        et_i = self.entity_imag[tail]
        return (
            eh_r * er_r * et_r
            + eh_i * er_r * et_i
            + eh_r * er_i * et_i
            - eh_i * er_i * et_r
        ).sum(dim=1)

    def regularization(self, triples: torch.Tensor) -> torch.Tensor:
        head, relation, tail = triples.long().unbind(dim=1)
        return (
            self.entity_real[head].square().sum(dim=1)
            + self.entity_imag[head].square().sum(dim=1)
            + self.entity_real[tail].square().sum(dim=1)
            + self.entity_imag[tail].square().sum(dim=1)
            + self.relation_real[relation].square().sum(dim=1)
            + self.relation_imag[relation].square().sum(dim=1)
        )

    def export_matrices(self) -> dict[str, torch.Tensor]:
        return {
            "MatrixE.real.best": self.entity_real,
            "MatrixE.imag.best": self.entity_imag,
            "MatrixR.real.best": self.relation_real,
            "MatrixR.imag.best": self.relation_imag,
        }


class RotatE(RUGEEmbeddingModel):
    """RotatE score and phase constraints used by the Java RUGE variant."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dimension: int,
        margin: float = 12.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        seed: int = 123,
    ) -> None:
        resolved_device = device or torch.device("cpu")
        super().__init__(
            num_entities, num_relations, dimension, resolved_device, dtype
        )
        if margin <= 0.0:
            raise ValueError("RotatE margin must be positive")
        self.margin = margin
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        embedding_range = (margin + 2.0) / dimension

        entity_values = torch.empty(
            (num_entities, dimension, 2), dtype=dtype, device="cpu"
        ).uniform_(-embedding_range, embedding_range, generator=generator)
        self.entity_real = nn.Parameter(entity_values[:, :, 0].to(resolved_device))
        self.entity_imag = nn.Parameter(entity_values[:, :, 1].to(resolved_device))
        phases = torch.empty(
            (num_relations, dimension), dtype=dtype, device="cpu"
        ).uniform_(-math.pi, math.pi, generator=generator)
        self.relation_phase = nn.Parameter(phases.to(resolved_device))

    def score(self, triples: torch.Tensor) -> torch.Tensor:
        head, relation, tail = triples.long().unbind(dim=1)
        phase = self.relation_phase[relation]
        cosine = torch.cos(phase)
        sine = torch.sin(phase)
        rotated_real = self.entity_real[head] * cosine - self.entity_imag[head] * sine
        rotated_imag = self.entity_real[head] * sine + self.entity_imag[head] * cosine
        distance = (
            (rotated_real - self.entity_real[tail]).abs()
            + (rotated_imag - self.entity_imag[tail]).abs()
        ).sum(dim=1)
        return self.margin - distance

    def regularization(self, triples: torch.Tensor) -> torch.Tensor:
        head, _, tail = triples.long().unbind(dim=1)
        return (
            self.entity_real[head].square().sum(dim=1)
            + self.entity_imag[head].square().sum(dim=1)
            + self.entity_real[tail].square().sum(dim=1)
            + self.entity_imag[tail].square().sum(dim=1)
        )

    def wrap_phases_(self) -> None:
        with torch.no_grad():
            self.relation_phase.copy_(
                torch.remainder(self.relation_phase + math.pi, 2.0 * math.pi)
                - math.pi
            )

    def export_matrices(self) -> dict[str, torch.Tensor]:
        return {
            "MatrixE.real.best": self.entity_real,
            "MatrixE.imag.best": self.entity_imag,
            "MatrixR.phase.best": self.relation_phase,
        }
