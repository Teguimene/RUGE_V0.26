import argparse
from pathlib import Path


def _read_raw_triples(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected 3 tab-separated fields")
            yield tuple(field.strip() for field in fields)


def _read_id_map(path: Path) -> dict[str, int]:
    mapping = {}
    with path.open("r", encoding="utf-8") as source:
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


def _write_id_map(path: Path, mapping: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    inverse = sorted((index, term) for term, index in mapping.items())
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for index, term in inverse:
            output.write(f"{index}\t{term}\n")


def convert_data_form(
    entity_map_path: str | Path,
    relation_map_path: str | Path,
    raw_train_path: str | Path,
    raw_valid_path: str | Path,
    raw_test_path: str | Path,
    train_output_path: str | Path,
    valid_output_path: str | Path,
    test_output_path: str | Path,
) -> tuple[int, int]:
    """Convert string triples, reusing existing Java ID maps when available."""
    entity_map_file = Path(entity_map_path)
    relation_map_file = Path(relation_map_path)
    raw_paths = [Path(raw_train_path), Path(raw_valid_path), Path(raw_test_path)]

    if entity_map_file.is_file() and relation_map_file.is_file():
        entity_ids = _read_id_map(entity_map_file)
        relation_ids = _read_id_map(relation_map_file)
    else:
        entities = set()
        relations = set()
        for head, relation, tail in _read_raw_triples(raw_paths[0]):
            entities.update((head, tail))
            relations.add(relation)
        entity_ids = {term: index for index, term in enumerate(sorted(entities))}
        relation_ids = {term: index for index, term in enumerate(sorted(relations))}
        _write_id_map(entity_map_file, entity_ids)
        _write_id_map(relation_map_file, relation_ids)

    for raw_path, output_path in zip(
        raw_paths,
        (Path(train_output_path), Path(valid_output_path), Path(test_output_path)),
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as output:
            for line_number, (head, relation, tail) in enumerate(
                _read_raw_triples(raw_path), start=1
            ):
                try:
                    ids = (entity_ids[head], relation_ids[relation], entity_ids[tail])
                except KeyError as error:
                    raise ValueError(
                        f"{raw_path}:{line_number}: term absent from training ID maps: "
                        f"{error.args[0]!r}"
                    ) from error
                output.write(f"{ids[0]}\t{ids[1]}\t{ids[2]}\n")

    return len(entity_ids), len(relation_ids)


def _bind_atom(
    atom: tuple[str, str, str],
    fact: tuple[int, int, int],
    relation_ids: dict[str, int],
    bindings: dict[str, int],
) -> dict[str, int] | None:
    head_term, relation_term, tail_term = atom
    head, relation, tail = fact
    if relation_ids.get(relation_term) != relation:
        return None

    result = dict(bindings)
    for term, value in ((head_term, head), (tail_term, tail)):
        if not term.startswith("?"):
            return None
        previous = result.get(term)
        if previous is not None and previous != value:
            return None
        result[term] = value
    return result


def ground_all_rules(
    relation_map_path: str | Path,
    rule_path: str | Path,
    training_path: str | Path,
    output_path: str | Path,
) -> tuple[int, int]:
    """Ground one- and two-atom rule bodies in the Java RUGE text format."""
    relation_ids = _read_id_map(Path(relation_map_path))
    facts_by_relation: dict[int, list[tuple[int, int, int]]] = {}
    facts_by_relation_head: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    facts_by_relation_tail: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    training_facts = set()
    with Path(training_path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"{training_path}:{line_number}: malformed triple")
            head, relation, tail = map(int, fields)
            fact = (head, relation, tail)
            training_facts.add(fact)
            facts_by_relation.setdefault(relation, []).append(fact)
            facts_by_relation_head.setdefault((relation, head), []).append(fact)
            facts_by_relation_tail.setdefault((relation, tail), []).append(fact)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    generated_keys = set()
    generated_count = 0
    skipped_rule_count = 0

    with Path(rule_path).open("r", encoding="utf-8") as rules, output_file.open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for line_number, line in enumerate(rules, start=1):
            line = line.strip()
            if not line.startswith("?"):
                continue
            if line.count("=>") != 1:
                raise ValueError(f"{rule_path}:{line_number}: malformed rule")

            body_text, head_text = line.split("=>", 1)
            body_tokens = body_text.split()
            head_parts = head_text.strip().split("\t")
            if len(head_parts) < 2:
                raise ValueError(
                    f"{rule_path}:{line_number}: rule confidence must follow a tab"
                )
            head_tokens = head_parts[0].split()
            confidence = float(head_parts[1].strip())
            if len(body_tokens) not in (3, 6) or len(head_tokens) != 3:
                skipped_rule_count += 1
                continue

            body = [
                tuple(body_tokens[index:index + 3])
                for index in range(0, len(body_tokens), 3)
            ]
            head_atom = tuple(head_tokens)
            try:
                body_relation_ids = [relation_ids[atom[1]] for atom in body]
                relation_ids[head_atom[1]]
            except KeyError as error:
                raise ValueError(
                    f"{rule_path}:{line_number}: unknown relation {error.args[0]!r}"
                ) from error

            first_facts = facts_by_relation.get(body_relation_ids[0], ())
            for first in first_facts:
                first_bindings = _bind_atom(body[0], first, relation_ids, {})
                if first_bindings is None:
                    continue
                if len(body) == 2:
                    second_head, _, second_tail = body[1]
                    if second_head in first_bindings:
                        second_facts = facts_by_relation_head.get(
                            (body_relation_ids[1], first_bindings[second_head]), ()
                        )
                    elif second_tail in first_bindings:
                        second_facts = facts_by_relation_tail.get(
                            (body_relation_ids[1], first_bindings[second_tail]), ()
                        )
                    else:
                        second_facts = facts_by_relation.get(
                            body_relation_ids[1], ()
                        )
                else:
                    second_facts = (None,)
                for second in second_facts:
                    premise_facts = [first]
                    bindings = first_bindings
                    if second is not None:
                        bindings = _bind_atom(body[1], second, relation_ids, bindings)
                        if bindings is None:
                            continue
                        premise_facts.append(second)

                    try:
                        inferred = (
                            bindings[head_atom[0]],
                            relation_ids[head_atom[1]],
                            bindings[head_atom[2]],
                        )
                    except KeyError as error:
                        raise ValueError(
                            f"{rule_path}:{line_number}: unbound head variable "
                            f"{error.args[0]!r}"
                        ) from error
                    if inferred in training_facts:
                        continue

                    grounded = tuple(premise_facts) + (inferred,)
                    if grounded in generated_keys:
                        continue
                    generated_keys.add(grounded)

                    unseen = len(grounded)
                    serialized = "\t".join(
                        f"({head}\t{relation}\t{tail})"
                        for head, relation, tail in grounded
                    )
                    output.write(f"{unseen}\t{serialized}\t{confidence}\n")
                    generated_count += 1

    return generated_count, skipped_rule_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RUGE data preprocessing utilities")
    commands = parser.add_subparsers(dest="command", required=True)

    convert = commands.add_parser("convert", help="Convert string triples to IDs")
    convert.add_argument("--entity-map", required=True)
    convert.add_argument("--relation-map", required=True)
    convert.add_argument("--raw-train", required=True)
    convert.add_argument("--raw-valid", required=True)
    convert.add_argument("--raw-test", required=True)
    convert.add_argument("--train-out", required=True)
    convert.add_argument("--valid-out", required=True)
    convert.add_argument("--test-out", required=True)

    ground = commands.add_parser("ground", help="Generate grounded soft rules")
    ground.add_argument("--relation-map", required=True)
    ground.add_argument("--rules", required=True)
    ground.add_argument("--train", required=True)
    ground.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "convert":
        entities, relations = convert_data_form(
            args.entity_map,
            args.relation_map,
            args.raw_train,
            args.raw_valid,
            args.raw_test,
            args.train_out,
            args.valid_out,
            args.test_out,
        )
        print(f"Converted data: {entities} entities, {relations} relations")
    else:
        generated, skipped = ground_all_rules(
            args.relation_map, args.rules, args.train, args.output
        )
        print(f"Generated {generated} grounding(s); skipped {skipped} rule(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
