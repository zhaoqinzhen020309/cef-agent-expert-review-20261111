from copy import deepcopy
import json

from agent_ahd.core.catalog_encoding import encode_catalog, decode_catalog


def test_shared_catalog_is_lossless_and_preserves_ids_and_input():
    catalog = {"round_index": 5, "available_actions": ["EXPLOIT", "ABLATE"],
               "candidate_options": [], "diagnostic_options": []}
    for i in range(300):
        catalog["candidate_options"].append({"option_id": f"choice_{i}", "action": "EXPLOIT",
            "scope": {"paths": [f"module_{j}.py" for j in range(40)]},
            "observation_ids": [f"observation_{i % 5}_{j}" for j in range(20)], "cost": {"candidate_slots": 1}})
    original = deepcopy(catalog)
    compact = encode_catalog(catalog)
    assert catalog == original
    assert decode_catalog(compact) == original
    assert len(json.dumps(compact)) < len(json.dumps(catalog)) * .4
    assert [row["option_id"] for row in decode_catalog(compact)["candidate_options"]] == [f"choice_{i}" for i in range(300)]


def test_small_catalog_keeps_ordinary_representation():
    catalog = {"candidate_options": [{"option_id": "x", "scope": {"a": []}}], "diagnostic_options": []}
    assert encode_catalog(catalog) == catalog
