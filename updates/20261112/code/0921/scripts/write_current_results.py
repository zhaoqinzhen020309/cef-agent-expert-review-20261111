"""Build a dated factual status note from receipts, without exposing credentials."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_campaign import summarize_run, load_registry, report
from scripts.report_campaign import aggregate


def main():
    selected = {
        "TSP-GLS": ROOT / "runs/pilot_gls_v2",
        "CVRP fold 0": ROOT / "runs/pilot_cvrp_v2",
        "FSSP fold 0": ROOT / "runs/pilot_fssp_v2",
        "TSP-Constructive ON": ROOT / "runs/campaign_v1/jobs/pilot_tsp_constructive_fold0_repeat0_on",
        "TSP-Constructive OFF": ROOT / "runs/campaign_v1/jobs/pilot_tsp_constructive_fold0_repeat0_off",
    }
    rows = {name: summarize_run(path) for name, path in selected.items()}
    campaign_root = ROOT / "runs/campaign_v1"
    registry = load_registry(campaign_root)
    progress = report(campaign_root, registry)
    results = aggregate(campaign_root, registry)
    generated_at = datetime.now(timezone.utc).isoformat()
    snapshot = {"generated_at_utc": generated_at, "pilot_runs": rows, "campaign": progress,
                "target_evidence": results["targets"]}
    guidance = {f"{kind}_{arm}": summarize_run(ROOT / "runs/guidance_pilot_v1" / f"{kind}_{arm}")
                for kind in ("tsp_gls", "tsp_constructive") for arm in ("on", "off")}
    snapshot["guidance_pilots"] = guidance
    (ROOT / "analysis_outputs/current_audit/progress_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    lines = ["# 本轮实际实验记录", "", f"自动读取回执时间（UTC）：{generated_at}", "",
             "统一接口、实验队列与优化代码已运行；目标分数、独立重复和外部对照分别核验。",
             "下表保留本轮工程阶段的固定证据；当前正式状态动态汇总在前，工程结果和pilot均不替代它。", "",
             "| 目标 | 已完成的数值证据 | 证据范围与尚缺部分 |",
             "| --- | --- | --- |",
             "| TSP-Constructive ≤13% | 工程版本验证9.9598% | 完整192例；缺少这项工程结果对应的独立Agent搜索证据 |",
             "| TSP-GLS ≤0.20% | 训练记忆ON试跑验证0.0672% | TSPLIB29、12候选、2求解种子；不是正式32候选×3搜索结果 |",
             "| CVRP 优于四种方法 | 早期试跑验证4.0907% | 仅一折20例，不能代替X100和完整外部比较 |",
             "| FSSP ≤0.14% | 工程版本训练0.6264% | 仅11例训练；此结果未达目标，不是110例验证结果 |", "",
             "新增记忆模块在GLS试跑中未胜出：关闭组验证0.0333%，优于启用组0.0672%。保留该负面结果。", "",
             "## 历史工程试跑", "",
             "下表均为4候选pilot，不能作为正式均值/方差、文献超越或目标达成证据。", "",
             "| 试跑 | 有效候选/预算 | 训练基线 gap % | 训练选定 gap % | 完整验证 gap % | 验证状态 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name, row in rows.items():
        val = row.get("validation", {})
        result = val.get("results", {}).get("selected", {})
        same = val.get("selection", {}).get("candidate_id") == "__baseline_control__"
        status = "VALID（选回基线，无算法提升）" if result.get("valid") and same else "VALID" if result.get("valid") else "INVALID" if result else "NOT_RUN"
        def number(x):
            return "—" if x is None else f"{x:.4f}"
        lines.append(f"| {name} | {row.get('valid_candidates', 0)}/{row.get('candidate_budget', 4)} | "
                     f"{number(row.get('baseline_train_gap'))} | {number(row.get('best_train_gap'))} | "
                     f"{number(result.get('gap_percent'))} | {status} |")
    lines.extend(["", "GLS验证为固定TSPLIB29；CVRP为五折中的20例；FSSP为五折中的22例；",
                  "构造任务为N=50/100/200各64例。前三项每实例2秒、2个求解种子；构造任务2秒、1种子。",
                  "不同任务的训练实例数分别为8、10、11、8。这些是工程可行性试跑。",
                  "GLS选定源码与基线相同；两次计时评价分数有波动，不能解释为框架收益。",
                  "构造OFF在N=50训练上得到9.1391%，但全验证因超时失败；不能用该训练分数宣布达到13%目标。", "",
                  "CVRP基线验证gap为21.7572%，选定候选4.0907%；FSSP基线3.2983%，选定候选1.4568%。",
                  "FSSP尚未达到0.14%目标。CVRP的四种外部方法已进入桥接试跑，完整公平对照仍未完成。", "",
                  "## 已停止的历史批次", "",
                  f"campaign_v1完成{progress['completed_searches']}次pilot后停止；已报告模型token：{progress['model_tokens']}。",
                  "CVRP可执行率门槛未通过，未启动登记的72次benchmark。旧结果和负面结果保留，不再自动恢复这批。",
                  "停止回执：`runs/campaign_v1/retirement.json`。", "",
                  "## 工程优化证据", "",
                  "- GLS固定10迭代、N=100：2.287秒→0.223秒，约10.27倍，路径一致。",
                  "- 行为聚类500×256：tracemalloc峰值22.63MB→3.25MB，下降85.6%。",
                  "- CVRP80例距离矩阵共150.83MB；按需加载最大单矩阵8.02MB，输入描述符约40KB。",
                  "- 历史ILS独立工程对照：N=50/100/200固定工作量路径一致，约1.2–1.4倍加速。",
                  "",
                  "上述微基准不代表完整系统RSS、正式吞吐或目标gap。原始数据在 `analysis_outputs/current_audit/`。",
                  "完整设计与实验边界见 `docs/2026-09-19-framework-optimization-design.md`；运行方法见根目录README。"])
    lines += ["", "## 资源说明与训练记忆的真实对照", "",
              "两组都启用原框架，仅切换新训练记忆/研究步骤指导；模型均为deepseek-flash。",
              "每次12候选、4轮；GLS训练16例、2种子，构造任务训练64例、1种子；每实例2秒。",
              "所有操作范围、数据、预算在开始前固定；以下仍是pilot。", "",
              "| 任务 / 训练记忆 | 搜索状态 | 有效候选 | 最佳训练 gap % | 验证 gap % |",
              "| --- | --- | --- | --- | --- |"]
    for name, row in guidance.items():
        result = row.get("validation", {}).get("results", {}).get("selected", {})
        lines.append(f"| {name} | {row['status']} | {row.get('valid_candidates', 0)}/12 | "
                     f"{number(row.get('best_train_gap'))} | {number(result.get('gap_percent')) if result.get('valid') else 'NOT_VALIDATED'} |")
    lines += ["", "GLS两组若均低于0.20%，只能说明本次试跑的数值达标；必须等正式独立重复后再判断最终目标。",
              "当前关闭训练记忆组优于启用组，不支持宣称该新增模块有优势；所有结果均保留。",
              "节点根目录solver.py可能是创建工作区时的基线；真正被评价的代码以search_state.json的solver_relpath和版本哈希为准。"]
    main_root = ROOT / "runs/research_campaign_v2"
    if (main_root / "registry.json").exists():
        from scripts.run_research_campaign import load, status
        latest = status(main_root, load(main_root))
        snapshot["research_campaign_v2"] = latest
        current_results = json.loads((main_root / "results.json").read_text())
        snapshot["target_evidence"] = current_results["targets"]
        formal_lines = ["## 当前正式结果", "",
                        f"已完成搜索 {latest['completed_searches']}/{latest['total_searches']}（含4项pilot）；完整验证及独立重复状态如下。", "",
                        "| 任务/框架 | 完整重复状态 | 验证均值gap % | 独立搜索样本方差 |",
                        "| --- | --- | --- | --- |"]
        for name, result in current_results["results"].items():
            formal_lines.append(f"| {name} | {result['status']} | {number(result['mean_gap_percent'])} | {number(result['sample_variance'])} |")
        formal_lines += ["", "目标状态：" + "；".join(f"{name}={result['status']}" for name, result in current_results["targets"].items()) + "。", ""]
        external_path = ROOT / "runs/external_cvrp_campaign_v1/results.json"
        if external_path.exists():
            external_results = json.loads(external_path.read_text())
            formal_lines += [f"CVRP四方法完整比较：{external_results.get('superiority', 'NOT_ESTABLISHED')}。", ""]
        if (main_root / "resource_stop.json").exists():
            stop = json.loads((main_root / "resource_stop.json").read_text())
            formal_lines += [f"最近资源停点回执：{stop['status']}。进程是否已恢复须同时查看监督器heartbeat。", ""]
        lines[4:4] = formal_lines
        lines += ["", "## 正在推进的正式队列", "",
                  f"research_campaign_v2完成搜索{latest['completed_searches']}/{latest['total_searches']}；报告token {latest['model_tokens']}。",
                  "TSP-Constructive：64例设计，100候选，每个框架条件3次独立搜索，N=50/100/200各64例验证。",
                  "TSP-GLS：64例设计，32候选，每例10秒，2个设计种子、10个验证种子、3次独立搜索。",
                  "CVRP/FSSP：各固定5折，每折设计10/11例分层样本，32候选，10秒，验证10种子，3次独立搜索。",
                  "76项=4次训练可执行性pilot+72次正式搜索；余额和总token上限可能阻止后续队列，登记不代表已完成。",
                  "`resource_stop.json`表示实际资源停点；`DRAIN_REQUEST.json`可请求当前批次结束后停止。",
                  "查看 `runs/research_campaign_v2/progress.json`、`results.json` 和 `jobs/launcher_logs`。"]
    lines += ["", "## 公开方法接入", "",
              "LLM4AD固定提交ffb6acf64497be93932c98d25369352efd3865cf，原始源码未修改。",
              "FunSearch/EoH/ReEvo/MCTS-AHD均已通过离线接口试验，各2候选，非科研结果。",
              "第一版CVRP桥接pilot为4候选：FunSearch 0/4、EoH 2/4、ReEvo 2/4、MCTS-AHD 2/4有效。",
              "FunSearch第一版提示缺少任务预算，不能用该批宣称公平优越。桥接已修正并冻结；第二版使用8候选、10秒，仍属工程pilot。",
              "四方法×五折×三次独立重复登记60项正式对照；以主框架CVRP结果和资源条件为运行前提，完成状态见当前正式汇总。",
              "相同完整求解器接口下重跑，不能代替原论文专属ACO宿主的复现。"]
    comparator_rows = {}
    lines += ["", "第二版桥接训练pilot实际回执（8候选、10例、2种子、10秒）：", "",
              "| 方法 | 状态 | 有效候选 | 最佳训练gap % | 截断调用数 |",
              "| --- | --- | --- | --- | --- |"]
    for name in ("funsearch", "eoh", "reevo", "mcts_ahd"):
        directory = ROOT / "runs/comparator_pilot_v2" / name
        summary_path, progress_path = directory / "summary.json", directory / "progress.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"status": "INCOMPLETE"}
        progress_row = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        scores = [r["gap"] for r in progress_row.get("proposals", []) if r.get("gap") is not None]
        best = min(scores) if scores else None
        truncated = sum(call.get("finish_reason") == "length" for call in progress_row.get("api_calls", []))
        comparator_rows[name] = {**summary, "best_training_gap_percent": best, "truncated_calls": truncated}
        lines.append(f"| {name} | {summary['status']} | {summary.get('valid_candidates', 0)}/8 | {number(best)} | {truncated} |")
    snapshot["comparator_pilot_v2"] = comparator_rows
    lines += ["", "以上只有训练结果，不能用来宣称主框架已经优于四方法。截断和非法候选保留在8个槽位中。"]
    control_path = ROOT / "runs/constructive_training_controls_v1/validation_summary.json"
    if control_path.exists():
        control = json.loads(control_path.read_text())
        snapshot["constructive_engineering_control"] = control
        lines += ["", "## 构造式TSP工程优化结果", "",
                  f"预登记4个有界rollout版本，仅按64例训练结果选择{control['candidate_id']}。训练gap {control['train_gap']:.4f}%。",
                  f"冻结后评价全部192例：有效={control['valid']}，平均验证gap={control['validation_gap_percent']:.4f}%。", "",
                  "| 规模 | 验证例数 | gap % |", "| --- | --- | --- |"]
        for name, row in control["groups"].items():
            lines.append(f"| {name} | {row['count']} | {row['mean_gap_percent']:.4f} |")
        lines += ["", "这项结果是人工优化旧Agent两步rollout后的工程对照；没有100候选、3次独立Agent搜索，不能替代框架最终结果。",
                  "限制首步/次步候选数并批量推进贪心完成：最坏构造工作量从无界两步枚举的O(n^5)降为固定K下O(K*n^3)，辅助空间O(n^2+K*n)。",
                  "这会改变搜索范围，不能称与原无界算法输出等价。选定K=32版本在独立无标签N=200输入上约0.362秒。",
                  "源码、训练分母、选择锁和验证回执：`runs/constructive_training_controls_v1/`。"]
    lines += ["", "## FSSP工程对照进展", "",
              "标量插入DP改为精确的NumPy前缀扫描；逐位置与独立makespan重算相等。每次插入仍为O(n*m)运算/空间，减少Python循环。",
              "相同11例训练、2种子、10秒：原Agent ILS 1.2959%，向量化ILS 1.0672%。",
              "另登记4个文献IG思路的工程版本，最佳训练gap为0.7377%；两个批量插入核版本完成，最佳训练gap进一步降为0.6264%。",
              "这些均为训练工程对照，不是110例验证结果，仍不能宣布达到0.14%。"]
    adaptive_root = ROOT / "runs/fssp_adaptive_training_controls_v1"
    if (adaptive_root / "registry.json").exists():
        grid = json.loads((adaptive_root / "registry.json").read_text())
        controls = {}
        for name in grid["sources"]:
            result_path = adaptive_root / (name + ".json")
            if result_path.exists():
                result = json.loads(result_path.read_text())
                controls[name] = {"valid": result["valid"], "training_gap_percent": result["score"]}
        snapshot["fssp_adaptive_engineering_controls"] = controls
        lines += ["", "后续双向构造、自适应扰动、部分序列搜索和精英池4项工程对照：", "",
                  "| 版本 | 有效 | 训练gap % |", "| --- | --- | --- |"]
        for name, result in controls.items():
            lines.append(f"| {name} | {result['valid']} | {number(result['training_gap_percent'])} |")
        if len(controls) == len(grid["sources"]):
            best_grid = min((r["training_gap_percent"] for r in controls.values() if r["valid"]), default=None)
            if best_grid is not None and best_grid >= .6263672475654716:
                lines += ["", "四项均未超过先前最佳0.6264%；不替换已冻结的工程版本，保留全部负面结果。"]
    lines += ["", "## 工作区新增可靠性能力", "",
              "确定性完整消融还原已在两条真实失败任务副本上通过回放，0次模型调用；原pilot结果没有修改。",
              "三树消融新增/删除辅助文件的校验边界已修复。仅新建合同可显式启用工具，运行中的v4批次保持原样。",
              "统一执行器新增structured输出，支持子集和分配表等原始结构；真实隔离进程、有限性和输出大小限制测试通过。",
              "任务评价器仍必须独立检查可行性并计算目标；该接口测试不是MKP公开基准实验。"]
    (ROOT / "analysis_outputs/current_audit/progress_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    stop_path = ROOT / "runs/legacy_stop_20260919.json"
    if stop_path.exists():
        snapshot["legacy_stop"] = json.loads(stop_path.read_text(encoding="utf-8"))
        lines[2:2] = ["**用户已变更验收范围：仅运行固定最小验证集，不再要求原全量验证。旧队列已停止，以下大规模结果是保留的历史状态。**", "",
                      "最新固定名单：`data/bundles/minimal_v1/selection.json`。旧进程停止回执：`runs/legacy_stop_20260919.json`。", ""]
        (ROOT / "analysis_outputs/current_audit/progress_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    (ROOT / "docs/2026-09-19-current-results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"updated": "docs/2026-09-19-current-results.md",
                      "historical_pilots_completed": progress["completed_searches"],
                      "current_campaign_completed": snapshot.get("research_campaign_v2", {}).get("completed_searches")}))


if __name__ == "__main__":
    main()
