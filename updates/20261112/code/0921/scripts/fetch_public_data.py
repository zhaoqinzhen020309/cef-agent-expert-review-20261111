"""Download public reference data; cache bytes and record URL/content identity."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
import argparse
import hashlib
import json
import time


ROOT = Path(__file__).resolve().parents[1]


def fetch(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "agent-ahd-research/1.0"}), timeout=45) as response:
                payload = response.read()
            if path.suffix == ".zip" and not payload.startswith(b"PK"):
                raise ValueError("server returned non-ZIP content")
            temporary = path.with_suffix(path.suffix + ".download")
            temporary.write_bytes(payload)
            temporary.replace(path)
            path.with_suffix(path.suffix + ".source.json").write_text(json.dumps({
                "url": url, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                "downloaded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, indent=2), encoding="utf-8")
            return path
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["tsplib", "cvrp", "fssp"])
    args = parser.parse_args()
    raw = ROOT / "data/raw" / args.kind
    if args.kind == "fssp":
        base = "https://zenodo.org/records/4542886/files/"
        for name in ("TestSolutions.zip", "Taillard_UB_Schedules.csv"):
            print(fetch(base + name + "?download=1", raw / name), flush=True)
        return
    if args.kind == "tsplib":
        # Full mirror retained; evaluation manifests select a fixed set before search.
        repo, branch, prefix = "mastqe/tsplib", "master", ""
    else:
        repo, branch, prefix = "PyVRP/Instances", "main", "CVRP/X-"
    tree_path = fetch(f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1", raw / "upstream_tree.json")
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    suffixes = (".tsp",) if args.kind == "tsplib" else (".vrp", ".sol")
    paths = [r["path"] for r in tree.get("tree", []) if r["path"].startswith(prefix) and r["path"].endswith(suffixes)]
    if not paths:
        raise RuntimeError("no instances found in upstream tree")
    revision = tree["sha"]
    def download(path):
        return str(fetch(f"https://raw.githubusercontent.com/{repo}/{revision}/{path}", raw / Path(path).name))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, path in enumerate(pool.map(download, paths), 1):
            if index % 20 == 0 or index == len(paths):
                print(json.dumps({"kind": args.kind, "downloaded": index, "total": len(paths)}), flush=True)


if __name__ == "__main__":
    main()
