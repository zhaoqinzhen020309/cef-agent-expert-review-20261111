from __future__ import annotations

from types import MappingProxyType


PROTOCOL_VERSIONS = MappingProxyType(
    {
        "schema": "agent_ahd_protocol_v3",
        "selection": "final_selection_v2",
        "sandbox": "candidate_sandbox_v1",
        "scope_evidence": "scope_boundary_evidence_v1",
        "mechanism_evidence": "mechanism_evidence_v2",
        "ood": "ood_evaluation_v2",
        "legacy_audit": "legacy_audit_v1",
    }
)

CURRENT_PROTOCOL_SCHEMA = "agent_ahd_protocol_v4"
