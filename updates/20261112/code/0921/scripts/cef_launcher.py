"""Fresh PAB generation repeats and the retained full-contrast search helper."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from cef.billing import balance_snapshot,write_billing
from cef.cli import load_env
from cef.data import TSPLIB_METRIC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

# Provider defaults shared by independent generation repeats.
MODE = "CEF"
ENV_FILE = ROOT / ".env"
API_KEY_ENV = "apikey"
ROUTE = "direct"
STREAM = True
REASONING_EFFORT = "high"
EXTRA_BODY = {"thinking": {"type": "enabled"}}
MAX_OUTPUT_TOKENS = 65_536
REQUEST_TIMEOUT_SECONDS = 600
TRANSPORT_RETRY_LIMIT = 1
TEST_REPEATS = 1
RUN_FINAL_TEST = True
REUSE_PREFLIGHT_RECEIPT = ROOT / "acceptance/cef_sdk_64k_recovery_20261028/preflight/receipt.json"
COMMON_START_PREFLIGHT_RECEIPT = ROOT / "acceptance/cef_common_start_20261028_223651/preflight/receipt.json"

# ==================== 以下为启动流程，通常不需要修改 ====================

def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def build_config(seed, folder, settings):
    MODEL = settings["model"]
    BASE_URL = settings["base_url"]
    iterations = settings["iterations"]
    workers = settings.get("workers",1)
    import numpy as np
    from importlib.metadata import version
    from cef.config import file_hash,freeze_research_contract

    config = load_json(ROOT / "configs/cef_v1_3/tsp.template.json")
    from cef.data import bind_original_tsplib,TSPLIB_DISTANCE_RULE,TSPLIB_REFERENCE_RULE
    config['data_manifest']=bind_original_tsplib(config['data_manifest'])
    config['reference_manifest'].update(evaluation_metric=TSPLIB_METRIC,distance_rule=TSPLIB_DISTANCE_RULE,test=TSPLIB_REFERENCE_RULE)
    if settings.get('tsplib_metric')=='tsplib_unrounded_original_reference_v1':
        from cef.data import prepare_tsplib_unrounded,UNROUNDED_TSPLIB_METRIC,UNROUNDED_DISTANCE_RULE,UNROUNDED_REFERENCE_RULE
        rows=prepare_tsplib_unrounded(ROOT,config['data_manifest']['test'],ROOT/'data'/UNROUNDED_TSPLIB_METRIC/'instances')
        config['data_manifest'].update(test=rows,evaluation_metric=UNROUNDED_TSPLIB_METRIC,distance_rule=UNROUNDED_DISTANCE_RULE,reference_rule=UNROUNDED_REFERENCE_RULE)
        config['reference_manifest'].update(evaluation_metric=UNROUNDED_TSPLIB_METRIC,distance_rule=UNROUNDED_DISTANCE_RULE,test=UNROUNDED_REFERENCE_RULE)
    if settings.get('supplement_manifest'):
        from cef.data import bind_supplement,receipt
        config['data_manifest']=bind_supplement(config['data_manifest'],settings['supplement_manifest'])
        config['reference_manifest']['training']='original64 author Concorde references + supplement64 independently bounded unrounded HiGHS tours'
        config['reference_manifest']['reference_files'].append(receipt(settings['supplement_manifest']))
        config['reference_manifest']['reference_files'].extend(r['reference_record'] for r in config['data_manifest']['train'][64:])
    config.update(
        mode=MODE, run_dir=str(folder / "search"), provider="deepseek",
        model=MODEL, model_snapshot=MODEL, endpoint=BASE_URL, stream=STREAM,
        wire_api="chat_completions", api_key_env=API_KEY_ENV,
        auth_source=f"{ENV_FILE.name}:{API_KEY_ENV} (secret is not copied)",
        route=ROUTE, model_sampling=dict(reasoning_effort=REASONING_EFFORT, **EXTRA_BODY), model_seed_support="unsupported",
        stop_rule="iterations", max_iterations=iterations, console_progress=True, evaluation_workers=workers,
        max_llm_requests=None, max_llm_tokens=None,
        max_output_tokens=MAX_OUTPUT_TOKENS, request_timeout_s=REQUEST_TIMEOUT_SECONDS,
        transport_retry_limit=TRANSPORT_RETRY_LIMIT,
        common_start_probe=settings.get('common_start_probe',False),
        probe_states=settings.get('probe_states',8),
        probe_seconds=settings.get('probe_seconds',3.0),
        max_case_runs=None, max_wallclock_s=None, max_analysis_seconds=None,
        test_repeats=TEST_REPEATS,
        evaluation_seed_manifest={row["id"]: seed for row in config["data_manifest"]["train"]},
        preflight_dir=str(folder / "preflight"),
        connectivity_receipt=str(REUSE_PREFLIGHT_RECEIPT or folder / "preflight/receipt.json"),
    )
    config["seed_sources"]["solver_seed"] = f"{seed}, fixed per fresh training/test worker"
    config["seed_sources"]["model_seed"] = "unsupported; independent provider sampling, no model seed sent"
    config["rates"] = None
    if settings.get('evaluation_policy')=='joint_only':
        config.update(evaluation_policy='joint_only',experiment_variant='PAB-only generation, no component contrasts',
                      acceptance_note='Independently regenerate complete A+B from the common original seed; only joint PAB is evaluated. No historical winning source or component-effect claim.')
    config=freeze_research_contract(config)
    if config['common_start_probe']:
        config.update(experiment_variant='PAB-only common-start closed-loop experiments',
                      acceptance_note='Complete PAB candidates receive formal training scores. Paired common-start diagnostics inform later Agent revisions and never rank the champion; the original bootstrap source remains unchanged.',
                      connectivity_receipt=str(COMMON_START_PREFLIGHT_RECEIPT))
    config["environment_manifest"] = dict(
        python=sys.version, dependencies={"numpy": np.__version__, "openai": version("openai"), "httpx": version("httpx")},
        hardware=dict(platform=platform.platform(), machine=platform.machine(),
                      processor=platform.processor(), logical_cpus=os.cpu_count()),
        worker_resources=dict(concurrency=workers, threads=1, memory_mb=1024,
                              isolation="restricted fresh Windows Job worker, not hostile-code sandbox"),
    )
    sources = sorted((ROOT / "cef").glob("*.py")) + sorted((ROOT / "cef/seeds").glob("*.py"))
    sources.extend([ROOT / "core/windows_job.py", ROOT / "runCEF.py", ROOT / "scripts/run_cef.py", Path(__file__).resolve()])
    config["implementation_source_manifest"] = [
        dict(path=str(path), sha256=file_hash(path)) for path in sources
    ]
    config["baseline_source_manifest"]["source_version"] = dict(
        path=str(ROOT / "cef/seeds/tsp.py"), sha256=file_hash(ROOT / "cef/seeds/tsp.py")
    )
    test = dict(
        instances=config["data_manifest"]["test"], test_repeats=TEST_REPEATS,
        evaluation_seeds={f"{row['id']}:{repeat}": seed
                          for row in config["data_manifest"]["test"]
                          for repeat in range(TEST_REPEATS)},
    )
    return config, test


def matching_receipt(config):
    from cef.preflight import receipt_matches
    path = Path(config["connectivity_receipt"])
    if not path.is_file():
        return False
    receipt = load_json(path)
    return receipt_matches(config,receipt)


def run_command(*args):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, str(ROOT / "scripts/run_cef.py"), "--quiet", *map(str, args)],
                   cwd=ROOT, env=env, check=True)


def run_one(config, test, folder):
    folder.mkdir(parents=True, exist_ok=False)
    config_path, test_path = folder / "search_config.json", folder / "test_config.json"
    reuse = matching_receipt(config)
    if not reuse:
        config["connectivity_receipt"] = str(folder / "preflight/receipt.json")
    write_json(config_path, config)
    write_json(test_path, test)
    print(f"Run: {folder.resolve()}", flush=True)
    if not reuse:
        print("Preflight...", flush=True)
        run_command("preflight", "--config", config_path, "--env-file", ENV_FILE)
        if not matching_receipt(config):
            report=load_json(config["connectivity_receipt"])
            detail=' -> '.join(report.get('error_chain') or [report.get('error','unknown error')])
            raise RuntimeError(f"Preflight failed: {detail}. Evidence: {folder / 'preflight'}")
    else:
        print("Preflight: reused", flush=True)
    print("Search...", flush=True)
    run_command("run", "--config", config_path, "--env-file", ENV_FILE)
    search = load_json(folder / "search/result.json")
    if search["status"] != "finished" or not search.get("best"):
        raise RuntimeError(f"搜索失败，不自动重跑：{search.get('reason', search['status'])}")
    if search.get('completed_iterations')!=config['max_iterations']:
        raise RuntimeError('Search did not complete the configured iterations')
    if search["best"]["candidate_id"] == "seed":
        print("Selected solver: initial baseline", flush=True)
    summary = dict(solver_seed=next(iter(config["evaluation_seed_manifest"].values())),
                   search=search, billing_file="billing_summary.json",evaluation_metric=TSPLIB_METRIC)
    if RUN_FINAL_TEST:
        print("Test: TSPLIB29", flush=True)
        run_command("test", "--run-dir", folder / "search", "--test-config", test_path,
                    "--output", folder / "test")
        summary["test"] = load_json(folder / "test/summary.json")
        if summary['test'].get('evaluation_metric')!=TSPLIB_METRIC:
            raise RuntimeError('Test metric differs from the approved original-distance TSPLIB reporting policy')
        if not summary["test"].get("search_unchanged"):
            raise RuntimeError("测试期间冻结的搜索记录发生变化，请检查原始记录")
    write_json(folder / "launch_summary.json", summary)
    print(f"Search completed | train_gap={search['best']['fitness']:.6f}%", flush=True)
    if RUN_FINAL_TEST:
        print(f"Test completed | valid={summary['test']['valid']} | gap={summary['test']['mean_gap_pct']}", flush=True)


def main(settings, argv=None):
    MODEL = settings["model"]
    SOLVER_SEEDS = settings["seeds"]
    iterations = settings["iterations"]
    OUTPUT_ROOT = Path(settings["output_dir"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只检查配置；不创建运行目录、不调用API")
    args = parser.parse_args(argv)
    from cef.config import validate
    from cef.data import audit_training, audit_test

    if not SOLVER_SEEDS or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in SOLVER_SEEDS):
        raise ValueError("SOLVER_SEEDS必须是非空列表，每个种子为0至2**32-1的整数")
    if len(set(SOLVER_SEEDS))!=len(SOLVER_SEEDS): raise ValueError('seeds must be distinct for independent repeats')
    joint_only=settings.get('evaluation_policy')=='joint_only'
    outcomes=[]
    series_stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    label='PAB common-start' if settings.get('common_start_probe') else ('PAB generation' if joint_only else 'CEF')
    print(f"{label} | model={MODEL} | iterations={iterations} | seeds={SOLVER_SEEDS} | workers={settings.get('workers',1)}", flush=True)
    for seed in SOLVER_SEEDS:
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
        folder = OUTPUT_ROOT / f"tsp_s{seed}_{stamp}"
        config, test = build_config(seed, folder, settings)
        config = validate(config)
        audit_training("tsp", config["data_manifest"]["train"],config['data_manifest'].get('training_protocol'))
        audit_test("tsp", test["instances"])
        if args.check: print(f"Configuration OK | train={len(config['data_manifest']['train'])} | test=29 | preflight={'reusable' if matching_receipt(config) else 'required'}", flush=True)
        if not args.check:
            if not ENV_FILE.is_file() and not os.environ.get(API_KEY_ENV):
                raise FileNotFoundError(f"找不到 {ENV_FILE}，且未设置环境变量 {API_KEY_ENV}")
            load_env(ENV_FILE if ENV_FILE.is_file() else None)
            before=balance_snapshot(config)
            error=None
            try:
                run_one(config, test, folder)
            except (ValueError,RuntimeError,OSError,subprocess.CalledProcessError) as exc:
                if not joint_only: raise
                error=type(exc).__name__+': '+str(exc)
                print(f'Seed {seed} failed | {error}',flush=True)
            finally:
                if folder.is_dir():
                    billing=write_billing(folder,before,balance_snapshot(config))
                    print(f"Usage | known_tokens={billing['usage']['known_total_tokens']} | unknown_requests={billing['usage']['unknown_usage_requests']}",flush=True)
                    if billing['account_balance_changes']:
                        print('Account balance change | '+' | '.join(row['currency']+' '+row['balance_decrease'] for row in billing['account_balance_changes']),flush=True)
            if joint_only:
                if error:
                    outcome=dict(seed=seed,status='failed',folder=str(folder),error=error,evaluation_metric=TSPLIB_METRIC)
                else:
                    launch=load_json(folder/'launch_summary.json'); tested=launch['test']
                    outcome=dict(seed=seed,status='finished' if tested['valid'] else 'invalid_test',folder=str(folder),train_gap_pct=launch['search']['best']['fitness'],test_gap_pct=tested['mean_gap_pct'],evaluation_metric=TSPLIB_METRIC,test_original_distance_gap_pct=tested['mean_gap_pct'])
                outcomes.append(outcome)
                write_json(folder/'repeat_summary.json',outcome)
    if joint_only and not args.check:
        successful=[r for r in outcomes if r['status']=='finished']
        import statistics
        series=dict(experiment_variant='PAB-only common-start closed-loop experiments' if settings.get('common_start_probe') else 'PAB-only independent generation repeats',attempts=len(outcomes),planned_repeats=len(SOLVER_SEEDS),successful_repeats=len(successful),results=outcomes,
                    evaluation_metric=TSPLIB_METRIC,
                    mean_test_gap_pct=statistics.fmean(r['test_gap_pct'] for r in successful) if len(successful)==len(outcomes) else None,
                    model_seed_support='unsupported; solver seeds are explicit and provider generations are independent')
        series['mean_original_distance_gap_pct']=series['mean_test_gap_pct']
        series_path=OUTPUT_ROOT/f'series_{series_stamp}.json'
        write_json(series_path,series)
        print(f'Series: {series_path}',flush=True)
        return 0 if all(r['status']=='finished' for r in outcomes) else 1
    return 0


def run(settings):
    try:
        return main(settings)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 1


def run_pab(settings):
    return run(dict(settings,evaluation_policy='joint_only'))
