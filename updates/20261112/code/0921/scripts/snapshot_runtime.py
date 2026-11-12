"""Freeze a runnable source closure so long experiments survive workspace edits."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


ROOT = Path(__file__).resolve().parents[1]


def snapshot(destination):
    destination = Path(destination).resolve()
    package = destination / "agent_ahd"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("snapshot destination must be empty")
    package.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / "__init__.py", ROOT / "requirements.runtime.lock"]
    for name in ("core", "agents", "evaluators", "experiments"):
        sources.extend((ROOT / name).glob("*.py"))
    sources.extend((ROOT / "problems").rglob("*.py"))
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    protocols = []
    for source in sorted((ROOT / "data/bundles").glob("**/protocol.json")):
        data = json.loads(source.read_text(encoding="utf-8"))
        baseline = Path(data["task"]["baseline_path"])
        data["task"]["baseline_path"] = str(package / baseline.relative_to(ROOT))
        target = destination / "protocols" / source.relative_to(ROOT / "data/bundles")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
        protocols.append(str(target))
    receipt = {"source_sha256": hashes, "protocols": protocols, "secrets_copied": False}
    (destination / "snapshot.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.destination), indent=2))
