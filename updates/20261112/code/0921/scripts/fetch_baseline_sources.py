"""Pin the published LLM4AD implementations of four external search baselines."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_public_data import fetch


def main():
    third_party = ROOT / "third_party"
    tree_path = fetch("https://api.github.com/repos/Optima-CityU/LLM4AD/git/trees/main?recursive=1", third_party / "llm4ad_tree.json")
    tree = json.loads(tree_path.read_text())
    revision = tree["sha"]
    root = third_party / "llm4ad"
    prefixes = ["llm4ad/base/", "llm4ad/tools/profiler/"] + ["llm4ad/method/" + k + "/" for k in ("funsearch", "eoh", "reevo", "mcts_ahd")]
    singles = {"LICENSE", "README.md", "setup.py", "requirements.txt", "llm4ad/__init__.py", "llm4ad/tools/__init__.py", "llm4ad/method/__init__.py", "llm4ad/task/__init__.py", "llm4ad/tools/llm/__init__.py"}
    names = [r["path"] for r in tree["tree"] if r["type"] == "blob" and
             (r["path"] in singles or r["path"].endswith(".py") and any(r["path"].startswith(p) for p in prefixes))]
    def download(name):
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("upstream path escapes destination")
        fetch(f"https://raw.githubusercontent.com/Optima-CityU/LLM4AD/{revision}/{name}", path)
        return name, hashlib.sha256(path.read_bytes()).hexdigest()
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(pool.map(download, names))
    receipt = {"repository": "https://github.com/Optima-CityU/LLM4AD", "commit": revision,
               "files": hashes, "modifications_to_upstream": [], "purpose": "research comparator implementations"}
    (third_party / "llm4ad_manifest.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({"commit": revision, "downloaded_files": len(hashes)}))


if __name__ == "__main__":
    main()
