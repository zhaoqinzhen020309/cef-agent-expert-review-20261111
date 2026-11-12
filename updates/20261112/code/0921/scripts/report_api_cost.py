"""Estimate charged API cost from individual reported attempts, not token totals.

Rates are a dated official-price snapshot; account bills remain authoritative.
Unknown usage is reported explicitly. No credentials or request headers are read.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
PRICES = {"deepseek-flash": (0.04, 2.0, 8.0), "deepseek-v4-pro": (0.30, 9.0, 27.0)}


def cost(usage, model, timestamp):
    cached_rate, input_rate, output_rate = PRICES[model]
    clock = datetime.fromtimestamp(timestamp, timezone(timedelta(hours=8)))
    peak = clock.weekday() < 5 and (9 <= clock.hour < 12 or 14 <= clock.hour < 18)
    scale = 1 if peak else .5
    inputs = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    outputs = usage.get("output_tokens", usage.get("completion_tokens", 0))
    cached = usage.get("cached_input_tokens", usage.get("prompt_cache_hit_tokens", 0))
    return scale * (cached * cached_rate + max(0, inputs - cached) * input_rate + outputs * output_rate) / 1000000


def main():
    records = []
    framework_roots = set((ROOT / "runs").glob("pilot_*"))
    for group in ("campaign_v1/jobs", "guidance_pilot_v1", "research_campaign_v2/jobs"):
        framework_roots.update(path.parent for path in (ROOT / "runs" / group).glob("*/model_contract.json"))
    for run in sorted(framework_roots):
        model_path, protocol_path, ledger = run / "model_contract.json", run / "model_protocol.json", run / "attempt_ledger.jsonl"
        if not ledger.exists() or not (model_path.exists() or protocol_path.exists()):
            continue
        if model_path.exists():
            model = json.loads(model_path.read_text())["model"]
        else:
            protocol = json.loads(protocol_path.read_text())
            model = protocol["code_model"]
            if protocol["controller_model"] != model:
                raise ValueError("cost report requires per-role prices for mixed-model pilot")
        if model not in PRICES:
            raise ValueError("unknown model price; do not silently drop charged attempts")
        rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        rows = [r for r in rows if r["event_type"] == "attempt_terminal"]
        estimated = sum(cost(r["payload"].get("usage", {}), model, r["recorded_at_epoch_sec"]) for r in rows)
        records.append({"run": str(run.relative_to(ROOT)), "model": model,
                        "attempts": len(rows), "estimated_cny": estimated,
                        "unknown_usage_attempts": sum(r["payload"].get("usage_status") != "reported" for r in rows),
                        "billing_timestamp_basis": "terminal ledger timestamp"})
    for group in ("comparator_pilot_v1", "comparator_pilot_v2", "external_cvrp_campaign_v1/jobs"):
        for progress_path in sorted((ROOT / "runs" / group).glob("*/progress.json")):
            run = progress_path.parent
            progress = json.loads(progress_path.read_text())
            calls = progress.get("api_calls", [])
            estimated, unknown = 0.0, 0
            for call in calls:
                response = run / "model_calls" / f"{call['call_id']:05d}.response.txt"
                if not response.exists() or not call.get("usage") or call.get("model") not in PRICES:
                    unknown += 1
                    continue
                estimated += cost(call["usage"], call["model"], response.stat().st_mtime)
            records.append({"run": str(run.relative_to(ROOT)), "attempts": len(calls), "estimated_cny": estimated,
                            "unknown_usage_attempts": unknown, "billing_timestamp_basis": "response file mtime approximation"})
    probe_path = ROOT / "runs/probe_deepseek/result.json"
    if probe_path.exists():
        probe = json.loads(probe_path.read_text())
        records.append({"run": "runs/probe_deepseek", "attempts": 1,
                        "estimated_cny": cost(probe, probe["model"], probe_path.stat().st_mtime),
                        "unknown_usage_attempts": 0, "billing_timestamp_basis": "probe receipt file mtime approximation"})
    result = {"pricing_date": "2026-09-19", "pricing_source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
              "scope": "known current-turn framework pilots/campaigns, external-method pilots/campaign and saved API probe; not an account-wide bill",
              "estimated_cny": sum(r["estimated_cny"] for r in records), "runs": records,
              "account_bill_authoritative": True, "generated_at": datetime.now(timezone.utc).isoformat()}
    output = ROOT / "analysis_outputs/current_audit/api_cost_estimate.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}))


if __name__ == "__main__":
    main()
