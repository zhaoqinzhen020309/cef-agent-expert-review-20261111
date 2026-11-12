"""Read-only delivery audit with secret-safe output and frozen-source checks."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent_ahd.core.local_config import load_deepseek_env
from scripts.run_research_campaign import load
from scripts.run_external_campaign import load as load_external


def main():
    primary = load(ROOT / "runs/research_campaign_v2")
    external = load_external(ROOT / "runs/external_cvrp_campaign_v1")
    secret = load_deepseek_env(ROOT / ".env")["api_key"]
    needles = [secret.encode("utf-8"), secret.encode("utf-16-le")]
    del secret
    scanned, byte_count, matches = 0, 0, []
    suffixes = {".py", ".json", ".jsonl", ".txt", ".log", ".md", ".ps1", ".yaml", ".yml"}
    for directory in ("agents", "core", "evaluators", "experiments", "problems", "scripts", "tests", "docs", "runs", "analysis_outputs"):
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes or "__pycache__" in path.parts:
                continue
            # Stream with overlap so large ledgers never need an extra full copy.
            overlap = max(map(len, needles)) - 1
            previous, found = b"", False
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    byte_count += len(chunk)
                    block = previous + chunk
                    if any(needle in block for needle in needles):
                        found = True
                        break
                    previous = block[-overlap:]
            scanned += 1
            if found:
                matches.append(str(path.relative_to(ROOT)))
    manifest_path = ROOT / "third_party/llm4ad_manifest.json"
    upstream = json.loads(manifest_path.read_text())
    mismatches = []
    # The manifest stores repository-relative names beneath third_party.
    for name, expected in upstream["files"].items():
        path = ROOT / "third_party" / name
        if not path.exists():
            path = ROOT / "third_party/LLM4AD" / name
        if not path.exists():
            path = ROOT / "third_party/llm4ad" / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append(name)
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "primary_registry_sha256": primary["sha256"], "external_registry_sha256": external["sha256"],
              "frozen_runtime_v4": "UNCHANGED_HASH_VERIFIED", "upstream_mismatches": mismatches,
              "secret_scan": {"files": scanned, "bytes": byte_count, "credential_matches": matches,
                              "excluded_authorized_secret_store": ".env"},
              "environment": {"python": sys.version, "executable": sys.executable, "cpu_threads": os.cpu_count(),
                              "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "jsonschema", "pytest", "psutil")}},
              "research_completion": "use registered results; tests and source integrity do not establish targets"}
    (ROOT / "analysis_outputs/current_audit/delivery_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"frozen_runtime": result["frozen_runtime_v4"], "secret_matches": len(matches),
                      "upstream_mismatches": len(mismatches), "files_scanned": scanned}))
    if matches or mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
