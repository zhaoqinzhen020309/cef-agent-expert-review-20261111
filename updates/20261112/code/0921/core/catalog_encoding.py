"""Lossless sharing of repeated catalog values, without pruning legal decisions."""
from collections import Counter
from copy import deepcopy
import json


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def encode_catalog(catalog: dict) -> dict:
    if len(_canonical(catalog)) < 48 * 1024:
        return catalog
    rows = [*catalog["candidate_options"], *catalog["diagnostic_options"]]
    counts = Counter()
    values = {}
    for row in rows:
        for value in row.values():
            if isinstance(value, (list, dict)):
                key = _canonical(value)
                if len(key) >= 80:
                    counts[key] += 1
                    values[key] = value
    keys = sorted(key for key, count in counts.items() if count > 1)
    if not keys:
        return catalog
    references = {key: f"value_{index}" for index, key in enumerate(keys)}
    encoded = deepcopy(catalog)
    encoded["encoding"] = "shared_values_v1; replace any {$shared: id} by shared_values[id]"
    encoded["shared_values"] = {references[key]: values[key] for key in keys}
    for row in [*encoded["candidate_options"], *encoded["diagnostic_options"]]:
        for name, value in list(row.items()):
            if isinstance(value, (list, dict)):
                reference = references.get(_canonical(value))
                if reference is not None:
                    row[name] = {"$shared": reference}
    all_rows = [*encoded["candidate_options"], *encoded["diagnostic_options"]]
    fields = list(all_rows[0]) if all_rows else []
    if fields and all(set(row) == set(fields) for row in all_rows):
        encoded["option_fields"] = fields
        encoded["encoding"] += "; each option row is an array in option_fields order"
        for group in ("candidate_options", "diagnostic_options"):
            encoded[group] = [[row[field] for field in fields] for row in encoded[group]]
    return encoded if len(_canonical(encoded)) < len(_canonical(catalog)) else catalog


def decode_catalog(catalog: dict) -> dict:
    if "shared_values" not in catalog:
        return catalog
    decoded = deepcopy(catalog)
    shared = decoded.pop("shared_values")
    decoded.pop("encoding")
    fields = decoded.pop("option_fields", None)
    if fields:
        for group in ("candidate_options", "diagnostic_options"):
            decoded[group] = [dict(zip(fields, row, strict=True)) for row in decoded[group]]
    for row in [*decoded["candidate_options"], *decoded["diagnostic_options"]]:
        for name, value in list(row.items()):
            if isinstance(value, dict) and set(value) == {"$shared"}:
                row[name] = deepcopy(shared[value["$shared"]])
    return decoded
