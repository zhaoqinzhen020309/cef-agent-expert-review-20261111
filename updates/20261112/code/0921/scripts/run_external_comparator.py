"""Unmodified pinned LLM4AD search + our common task/evaluator bridge.

Research use: acknowledge LLM4AD, Liu et al., arXiv:2412.17287.
This is a common-interface re-evaluation, not reproduction of paper tables.
The only search instrumentation is a strict proposal/API budget at the LLM
boundary. No changes to selection, reflection, crossover or tree algorithms.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.comparator_contract import make_template, describe_task
sys.path.insert(0, str(ROOT / "third_party/llm4ad"))
from llm4ad.base import Evaluation, LLM
from llm4ad.method.eoh import EoH
from llm4ad.method.funsearch import FunSearch
from llm4ad.method.reevo import ReEvo
from llm4ad.method.mcts_ahd import MCTS_AHD
import numpy as np


class BudgetExhausted(KeyboardInterrupt):
    """Stop upstream loops without being swallowed by their broad Exception handlers."""


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrap_program(program, contract):
    params = contract["parameters"][1:]
    # Parse success upstream never authorizes arbitrary execution in this process.
    return (program + f"\n\nclass {contract['class_name']}:\n"
            f"    def {contract['method_name']}(self, {', '.join(params)}):\n"
            f"        return design_algorithm({', '.join(params)})\n")


def is_reflection(method, prompt):
    # These are exact markers in the pinned upstream prompt implementations.
    # They distinguish extra textual reasoning from algorithm proposal slots.
    return ((method == "reevo" and str(prompt).startswith(
        "You are an expert in the domain of optimization heuristics. Your task is to give hints"))
        or (method == "mcts_ahd" and "now you should re-describe the algorithm using less than 3 sentences" in str(prompt)))


class RunState:
    def __init__(self, args, protocol):
        self.args, self.protocol, self.root = args, protocol, args.run_dir
        self.rows, self.calls, self.tokens, self.active = [], [], 0, None
        self.start = time.time()
        self.template = make_template(protocol, args.seconds)
        self.method = None

    def stop(self):
        # Stop-only instrumentation: prevent upstream empty-population loops
        # after parse failures use up the common proposal budget.
        if self.method is not None:
            self.method._max_sample_nums = self.method._tot_sample_nums

    def flush(self):
        dump(self.root / "progress.json", {"proposals": self.rows, "api_calls": self.calls,
             "tokens": self.tokens, "elapsed_seconds": time.time() - self.start})

    def settle_unparsed(self):
        if self.active is not None:
            self.rows[self.active]["status"] = "INVALID_UPSTREAM_PARSE"
            self.active = None
            self.flush()

    def evaluate(self, program, candidate_id, split="train"):
        args = self.args
        directory = self.root / candidate_id
        directory.mkdir(parents=True, exist_ok=True)
        solver = directory / "solver.py"
        if split == "train":
            solver.write_text(wrap_program(program, self.protocol["task"]["entry_contract"]), encoding="utf-8")
        seeds = list(range(args.seed, args.seed + args.solver_seeds)) if split == "train" else list(range(args.seed + 200, args.seed + 200 + args.final_seeds))
        count = args.instances if split == "train" else len(self.protocol["validation"]["instance_ids"])
        output = directory / f"{split}.json"
        request = {"runtime": str(args.runtime), "problem": self.protocol["task"]["name"], "protocol": str(args.protocol),
                   "candidate_id": candidate_id, "candidate_path": str(solver), "split": split, "output": str(output),
                   "budget": {"n_instances": count, "time_limit": args.seconds, "ite_max": args.iterations, "seeds": seeds}}
        request_path = directory / f"{split}_request.json"
        dump(request_path, request)
        with (directory / f"{split}.out.log").open("w") as out, (directory / f"{split}.err.log").open("w") as err:
            subprocess.run([args.evaluator_python, str(self.root / "bridge_source/comparator_eval_worker.py"), str(request_path)],
                           cwd=ROOT, stdout=out, stderr=err, check=True,
                           env={**os.environ, "PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return json.loads(output.read_text(encoding="utf-8"))


class CommonEvaluation(Evaluation):
    def __init__(self, state):
        self.state = state
        description = describe_task(state.protocol, state.args.seconds)
        description += "\nReference baseline available to every method:\n" + Path(state.protocol["task"]["baseline_path"]).read_text(encoding="utf-8")
        super().__init__(template_program=state.template, task_description=description, exec_code=False, safe_evaluate=False)

    def evaluate_program(self, program_str, callable_func, **kwargs):
        s = self.state
        if s.active is None:
            # FunSearch evaluates its initial template separately, as our method does.
            return -json.loads((s.root / "baseline/train.json").read_text())["score"]
        row = s.rows[s.active]
        try:
            result = s.evaluate(program_str, row["candidate_id"])
        except Exception as exc:
            row.update(status="INFRASTRUCTURE_ERROR", error_type=type(exc).__name__)
            s.active = None
            s.flush()
            raise
        row.update(status="VALID" if result["valid"] else "INVALID_EVALUATION", gap=result["score"],
                   error_type=result.get("error_type"), source_sha256=sha(s.root / row["candidate_id"] / "solver.py"))
        s.active = None
        s.flush()
        return -result["score"] if result["valid"] else None


class BudgetedLLM(LLM):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def draw_sample(self, prompt, *unused, **kwargs):
        s, args = self.state, self.state.args
        reflection = is_reflection(args.method, prompt)
        if not reflection:
            s.settle_unparsed()
        if (time.time() - s.start >= args.wall_seconds or len(s.calls) >= args.api_call_budget
            or s.tokens + args.max_tokens + len(str(prompt)) >= args.token_budget
            or (not reflection and len(s.rows) >= args.candidates)):
            s.stop()
            raise BudgetExhausted("registered experiment budget exhausted")
        if not reflection:
            s.active = len(s.rows)
            s.rows.append({"candidate_id": f"candidate_{s.active:04d}", "status": "GENERATING"})
        call = {"call_id": len(s.calls), "kind": "reflection" if reflection else "proposal", "proposal": s.active}
        s.calls.append(call)
        s.flush()
        path = s.root / "model_calls" / f"{call['call_id']:05d}"
        dump(path.with_suffix(".prompt.json"), {"prompt": prompt})
        if args.offline_fixture:
            response = ("invalid fixture" if args.fixture_invalid else "Preserve the feasible construction." if reflection
                        else "{Baseline feasibility fixture}\n```python\n" + s.template + "\n```")
            call["offline_fixture"] = True
        else:
            sys.path.insert(0, str(ROOT))
            from agent_ahd.core.local_config import load_deepseek_env
            credentials = load_deepseek_env(ROOT / ".env")
            model = credentials["model"] or args.model
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                       "thinking": {"type": "enabled"}, "reasoning_effort": "high", "max_tokens": args.max_tokens}
            request = urllib.request.Request(credentials["base_url"] + "/chat/completions",
                       data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + credentials["api_key"]})
            try:
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=600) as connection:
                    result = json.load(connection)
                response = result["choices"][0]["message"]["content"] or ""
                call["usage"] = result.get("usage", {})
                call["model"] = result.get("model")
                call["finish_reason"] = result["choices"][0].get("finish_reason")
                s.tokens += int(call["usage"].get("total_tokens", 0))
            except Exception as exc:
                # Exception bodies may contain proxy details; persist only the type.
                call["error_type"] = type(exc).__name__
                call["conservative_tokens_unknown_usage"] = args.max_tokens + len(str(prompt))
                s.tokens += call["conservative_tokens_unknown_usage"]
                if s.active is not None:
                    s.rows[s.active].update(status="MODEL_API_ERROR", error_type=type(exc).__name__)
                    s.active = None
                s.flush()
                s.stop()
                raise BudgetExhausted("model service failure; preserve incomplete run") from None
        path.with_suffix(".response.txt").write_text(response, encoding="utf-8")
        s.flush()
        return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["funsearch", "eoh", "reevo", "mcts_ahd"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=ROOT / "runs/runtime_v4")
    parser.add_argument("--evaluator-python", default=r"D:\EdgeDownload\Anaconda\python.exe")
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--instances", type=int, default=80)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=63001)
    parser.add_argument("--solver-seeds", type=int, default=2)
    parser.add_argument("--final-seeds", type=int, default=10)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--token-budget", type=int, default=6000000)
    parser.add_argument("--api-call-budget", type=int, default=600)
    parser.add_argument("--wall-seconds", type=float, default=86400)
    parser.add_argument("--offline-fixture", action="store_true")
    parser.add_argument("--fixture-invalid", action="store_true")
    parser.add_argument("--no-finalize", action="store_true")
    args = parser.parse_args()
    if args.fixture_invalid and not args.offline_fixture:
        raise ValueError("invalid fixture requires offline mode")
    args.run_dir, args.protocol, args.runtime = args.run_dir.resolve(), args.protocol.resolve(), args.runtime.resolve()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError("upstream optimizer states are not resumable by this bridge; use a new run")
    if min(args.candidates, args.instances, args.seconds, args.solver_seeds, args.final_seeds) <= 0:
        raise ValueError("budgets must be positive")
    manifest = json.loads((ROOT / "third_party/llm4ad_manifest.json").read_text())
    for path, expected in manifest["files"].items():
        if sha(ROOT / "third_party/llm4ad" / path) != expected:
            raise ValueError("upstream source integrity violation")
    snapshot = json.loads((args.runtime / "snapshot.json").read_text())
    for path, expected in snapshot["source_sha256"].items():
        if sha(args.runtime / "agent_ahd" / path) != expected:
            raise ValueError("evaluator runtime integrity violation")
    protocol = json.loads(args.protocol.read_text())
    if not args.offline_fixture:
        sys.path.insert(0, str(ROOT))
        from agent_ahd.core.local_config import load_deepseek_env
        credentials = load_deepseek_env(ROOT / ".env")
        args.model = credentials["model"] or args.model
    if args.instances > len(protocol["train"]["instance_ids"]):
        raise ValueError("training budget exceeds manifest")
    registration = {"configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "upstream": manifest, "protocol_sha256": sha(args.protocol), "baseline_sha256": sha(protocol["task"]["baseline_path"]),
                    "bridge_sha256": sha(__file__), "worker_sha256": sha(ROOT / "scripts/comparator_eval_worker.py"),
                    "contract_helper_sha256": sha(ROOT / "scripts/comparator_contract.py"),
                    "source_runtime_sha256": sha(args.runtime / "snapshot.json"),
                    "interpretation": "LLM4AD common-interface re-evaluation; not paper-specific host reproduction",
                    "budget": "All algorithm proposals including upstream parse failures count; reflections recorded separately; baseline outside proposal budget"}
    dump(args.run_dir / "registration.json", registration)
    frozen_source = args.run_dir / "bridge_source"
    frozen_source.mkdir()
    for path in (Path(__file__), ROOT / "scripts/comparator_eval_worker.py", ROOT / "scripts/comparator_contract.py"):
        (frozen_source / path.name).write_bytes(path.read_bytes())
    random.seed(args.seed)
    np.random.seed(args.seed)
    state = RunState(args, protocol)
    thread_errors = []
    original_hook = threading.excepthook
    def record_thread_failure(event):
        if isinstance(event.exc_value, BudgetExhausted):
            return
        thread_errors.append(event.exc_type.__name__)
        original_hook(event)
    threading.excepthook = record_thread_failure
    baseline = state.evaluate(state.template, "baseline")
    if not baseline["valid"]:
        raise RuntimeError("baseline failed common evaluator")
    evaluation, llm = CommonEvaluation(state), BudgetedLLM(state)
    common = dict(llm=llm, evaluation=evaluation, max_sample_nums=args.candidates, num_samplers=1, num_evaluators=1)
    if args.method == "funsearch":
        method = FunSearch(**common, samples_per_prompt=1)
    elif args.method == "eoh":
        method = EoH(**common, max_generations=None, pop_size=5)
    elif args.method == "reevo":
        method = ReEvo(**common, pop_size=5)
    else:
        method = MCTS_AHD(**common, init_size=4, pop_size=5)
    state.method = method
    try:
        method.run()
    except KeyboardInterrupt:
        pass
    finally:
        method._evaluation_executor.shutdown(wait=True, cancel_futures=True)
    state.settle_unparsed()
    completed = (len(state.rows) == args.candidates and not thread_errors
                 and not any(r["status"] in {"INFRASTRUCTURE_ERROR", "MODEL_API_ERROR"} for r in state.rows))
    summary = {"status": "COMPLETED_SEARCH" if completed else "INCOMPLETE", "proposals": len(state.rows),
               "valid_candidates": sum(r["status"] == "VALID" for r in state.rows), "tokens": state.tokens,
               "evidence_type": "offline_fixture_only" if args.offline_fixture else "external_method_search",
               "unexpected_thread_errors": thread_errors}
    if completed and not args.no_finalize:
        contenders = [(baseline["score"], "baseline")] + [(r["gap"], r["candidate_id"]) for r in state.rows if r["status"] == "VALID"]
        gap, selected = min(contenders)
        dump(state.root / "training_selection_lock.json", {"candidate_id": selected, "train_gap": gap,
             "source_sha256": sha(state.root / selected / "solver.py"), "validation_used_for_selection": False})
        result = state.evaluate(None, selected, "validation")
        summary["validation"] = {"valid": result["valid"], "gap_percent": result["score"]}
    dump(state.root / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
