from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TripleRecord:
    """Internal triple representation used across the encoding pipeline."""

    subject: str
    predicate: str
    object: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def to_dict(self) -> dict[str, str]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
        }


@dataclass(slots=True)
class KGEncodingContext:
    """Holds parsed triples and deterministic mappings for an encoding run."""

    triples: list[TripleRecord]
    entity_to_id: dict[str, int]
    predicate_to_id: dict[str, int]
    entity_bit_width: int
    predicate_bit_width: int
    fixed_thesis_mapping: bool = False
    dataset_path: str | None = None
    triple_to_index: dict[TripleRecord, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.triple_to_index:
            self.triple_to_index = {
                triple: index for index, triple in enumerate(self.triples)
            }

    @property
    def id_to_entity(self) -> dict[int, str]:
        return {identifier: entity for entity, identifier in self.entity_to_id.items()}

    @property
    def id_to_predicate(self) -> dict[int, str]:
        return {
            identifier: predicate
            for predicate, identifier in self.predicate_to_id.items()
        }

    @property
    def triple_count(self) -> int:
        return len(self.triples)

    @property
    def entity_count(self) -> int:
        return len(self.entity_to_id)

    @property
    def predicate_count(self) -> int:
        return len(self.predicate_to_id)

    @property
    def total_basis_qubits(self) -> int:
        return (2 * self.entity_bit_width) + self.predicate_bit_width

    def triple_numeric_ids(self, triple: TripleRecord) -> tuple[int, int, int]:
        return (
            self.entity_to_id[triple.subject],
            self.predicate_to_id[triple.predicate],
            self.entity_to_id[triple.object],
        )

    def to_serializable_dict(self) -> dict[str, object]:
        return {
            "dataset_path": self.dataset_path,
            "fixed_thesis_mapping": self.fixed_thesis_mapping,
            "triple_count": self.triple_count,
            "entity_count": self.entity_count,
            "predicate_count": self.predicate_count,
            "entity_bit_width": self.entity_bit_width,
            "predicate_bit_width": self.predicate_bit_width,
            "triples": [triple.to_dict() for triple in self.triples],
            "entity_to_id": self.entity_to_id,
            "predicate_to_id": self.predicate_to_id,
        }
