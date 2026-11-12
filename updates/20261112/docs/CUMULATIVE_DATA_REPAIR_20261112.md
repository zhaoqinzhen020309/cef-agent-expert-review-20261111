# 累计测量与候选交接修复

备份：`backups/cumulative_data_20261112_070537/code.zip`。

## 实际修改

`cef/data_coverage.py` 合并同一候选、规则、计划、校准源码与精确执行条件下的有效逐题观测。重复实例不增加覆盖数，保留首个有效匹配观测，不挑选最高成绩。不同规则、参数、计划或条件不能混合。由逐题gap计算完整均分。

`TaskExperiments.measure_data_candidate` 在每次批次结束后读取已有证据并计算累计覆盖，完整后执行既有全题集合检查及难度竞争规则，再返回原有 `_goal_proposal` 交付通道。本次调用的子集成绩保持原义；累计成绩单列 `cumulative_coverage` 与 `cumulative_report`。

`update_research` 在数据任务里将程序重建的 `best_complete_candidate`、其完整候选身份/规则参数/源码哈希/证据，以及 `working_source_id` 同时放入检查点。原有 ToolSession 上下文重启会直接携带这份结构化检查点。自然语言判断不覆盖实际成绩与身份。

## 验证

- 第一代第19与26次真实记录回放：同一方案15+49题，覆盖64/64，均分0.05741307208456444%，条件一致；无模型或求解器调用。
- 加入第64次较弱试验后，最好完整候选仍为原方案。记录：`acceptance/cumulative_data_20261112/real_log_replay.json`。
- 回归79项通过，另6项子测试通过。包括分批与单批一致、重复不计数、不兼容记录隔离、最好候选保持、实际分批测量到 `_goal_proposal` 通道。报告：`acceptance/cumulative_data_20261112/tests.xml`。

## 接续第一代

原实验保持不动。接续副本：`experiments/cef_focused_tasks/continued_s2025_20261112_cumulative`。
`continuation_provenance.json` 明确第一代使用原实现，修复从第二代开始适用。旧源码快照保存在副本的 `search/runtime_snapshot_round1`；历史证据仍指向保留的原文件。接续配置与引擎材料读取已通过离线检查。

运行第二代：

```powershell
python .\runCEFAuto.py --next "C:\Users\48520\Desktop\eohagent\0921\experiments\cef_focused_tasks\continued_s2025_20261112_cumulative"
```

本次没有启动第二代或改变第一代测试结果。修复保证已完成观测能被识别、候选成绩不因摘要而换名，不保证第二代算法一定更好。
