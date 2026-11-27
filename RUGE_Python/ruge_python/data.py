from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch


RulePolarity = Literal["positive", "negative"]


@dataclass(frozen=True)
class GroundRule:
    triples: tuple[tuple[int, int, int], ...]
    unseen: int
    confidence: float
    polarity: RulePolarity = "positive"

    @property
    def target(self) -> tuple[int, int, int]:
        return self.triples[self.unseen - 1]


def load_triples(path: str | Path, num_entities: int, num_relations: int) -> torch.Tensor:
    rows = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected 3 integer columns")
            head, relation, tail = map(int, fields)
            if not 0 <= head < num_entities or not 0 <= tail < num_entities:
                raise ValueError(f"{path}:{line_number}: entity ID out of range")
            if not 0 <= relation < num_relations:
                raise ValueError(f"{path}:{line_number}: relation ID out of range")
            rows.append((head, relation, tail))
    if not rows:
        raise ValueError(f"{path}: no triples found")
    return torch.tensor(rows, dtype=torch.long)


def read_id_map(path: str | Path) -> dict[str, int]:
    mapping = {}
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.rstrip("\r\n").split("\t", 1)
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: malformed ID mapping")
            index = int(fields[0])
            term = fields[1]
            if term in mapping:
                raise ValueError(f"{path}:{line_number}: duplicate term {term!r}")
            mapping[term] = index
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError(f"{path}: IDs must be contiguous and start at zero")
    return mapping


def load_groundings(
    path: str | Path, num_entities: int, num_relations: int
) -> list[GroundRule]:
    """Load legacy positive groundings or explicitly signed groundings.

    Signed rows place ``positive`` or ``negative`` after the unseen position.
    Legacy rows omit that marker and are interpreted as positive.
    """
    rules = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.replace("(", " ").replace(")", " ").split()
            if not fields:
                continue
            unseen = int(fields[0])
            confidence = float(fields[-1])
            if len(fields) > 1 and fields[1] in ("positive", "negative"):
                polarity: RulePolarity = fields[1]
                value_fields = fields[2:-1]
            else:
                polarity = "positive"
                value_fields = fields[1:-1]
            values = list(map(int, value_fields))
            if len(values) % 3:
                raise ValueError(f"{path}:{line_number}: malformed grounding")
            triples = tuple(
                (values[i], values[i + 1], values[i + 2])
                for i in range(0, len(values), 3)
            )
            if len(triples) not in (2, 3, 4):
                raise ValueError(
                    f"{path}:{line_number}: expected 2, 3, or 4 triples"
                )
            if not 1 <= unseen <= len(triples):
                raise ValueError(f"{path}:{line_number}: invalid unseen position")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"{path}:{line_number}: invalid confidence")
            for head, relation, tail in triples:
                if not 0 <= head < num_entities or not 0 <= tail < num_entities:
                    raise ValueError(f"{path}:{line_number}: entity ID out of range")
                if not 0 <= relation < num_relations:
                    raise ValueError(f"{path}:{line_number}: relation ID out of range")
            rules.append(GroundRule(triples, unseen, confidence, polarity))
    return rules
