# 当前核心代码与第1轮完整轨迹：阅读指南

这是原项目的独立分析快照，不是向原项目仓库提交代码。只读导出，原实验继续运行。

## 先看哪里

1. [项目、机制、实验与成绩说明](PROJECT_EXPERT_BRIEFING_ZH.md)：不用看代码即可了解项目，正文截点为2026-11-11 16:15。
2. [当前运行入口](../code/0921/runCEFAuto.py)：用户当前使用的配置。
3. [交替运行主流程](../code/0921/cef/alternating.py)：数据轮次、当轮评估身份、父程序重测、手动暂停。
4. [数据Agent流程](../code/0921/cef/integrated_data.py)：旧题校准、新题准备、新旧题竞争。
5. [Agent任务与提交契约](../code/0921/cef/tool_agents.py)、[工具循环](../code/0921/cef/tool_session.py)、[工具接口](../code/0921/cef/task_tools.py)、[研究实验工具](../code/0921/cef/task_experiments.py)。
6. [正式候选与选优](../code/0921/cef/engine.py)、[执行器](../code/0921/cef/runner.py)、[评分](../code/0921/cef/tasks.py)。
7. [SCIP参考后端](../code/0921/cef/reference_windows.py)、[参考MIP实现](../code/0921/cef/reference_scip_solver.py)。
8. [模型请求与usage](../code/0921/cef/model.py)、[逐轮外部测试](../code/0921/cef/round_benchmark.py)。

代码目录包含cef主模块、所需Windows执行辅助、入口和恢复脚本、相关验收测试。历史core辅助文件一并保存用于追踪依赖，不代表所有文件都处在当前主调用链上。

## 下载和打开完整轨迹

仓库包含完整压缩包的三个二进制分片，总约154 MiB；解开后的原始代码与记录合计约454 MiB。分片用于满足GitHub单文件限制，不是截短日志。

```bash
git clone https://github.com/zhaoqinzhen020309/cef-agent-expert-review-20261111.git
cd cef-agent-expert-review-20261111
python tools/unpack_round1.py
```

需要Python 3.9或更高版本，无第三方依赖。Windows可用`py tools/unpack_round1.py`。解包工具先验证各分片和完整ZIP的SHA-256，再逐文件校验导出清单；不会调用模型或运行求解器。

只想验证而不解压：

```bash
python tools/unpack_round1.py --verify-only
```

默认解压到`round1_trace/`，该目录不提交Git。可用`--output 新目录`指定位置；不会覆盖已有非空目录。

## 解压后目录

| 目录/文件 | 内容 |
|---|---|
| `rounds/001/` | 第1轮完整数据任务、工作区、生成器、候选数据测量、坐标/参考/报告、规则、选择记录、最终轮次代码与汇总 |
| `rounds/001/data/agent_task/session/facts.jsonl` | 数据Agent的完整71次原生会话、提示词、模型回复与工具请求/结果 |
| `search/agent_tasks/step_0_planner/` | 第1轮Planner完整36次会话、工作区、临时脚本、实验结果 |
| `search/agent_tasks/step_0_editor/` | 第1轮Editor完整35次会话、工作区、候选代码、全部工具结果与完整实验记录 |
| `search/facts.jsonl` | 主搜索事件与算法角色provider日志，按第2轮启动之前截取原始完整行 |
| `data_agent/facts.jsonl` | 数据模型provider日志，第2轮之前部分，包含继承的第1轮请求与usage |
| `measurements/facts.jsonl` | 当时已保存的实测观测和复用身份 |
| `search/snapshots/`、`search/sources/` | 第2轮前已存在的内容寻址状态与源码 |
| `search/round_feedback/001.json` | 第1轮正式训练反馈与逐题记录 |
| `search/training_epochs/001.json` | 第1轮采用的数据与种子 |
| `round_benchmarks/001/` | 第1轮TSPLIB29结果、每题路径、冻结条件、最终代码 |
| `baseline/` | 起点算法与已保存的旧口径TSPLIB29评估 |
| `training/original64/` | 原64题实际矩阵/参考数据文件 |
| `progress.log`、`console.raw.log` | 原可读日志和终端输出中第2轮开始前的完整前缀，包括故障信息 |
| `runtime/current_frozen/` | 导出时当前冻结运行源码 |
| `runtime/before_continuation/` | 恢复迁移前的历史源码快照，用于解释早期数据任务与修复 |
| `runtime/round1_overrides/cef/` | 第1轮结束时的model.py、tool_session.py版本；它们后来被自动重连修复改变 |
| `search_config.json`、`test_config.json` | 导出时保存的配置；源文件哈希已包含之后的显式通信修复，不要误称第1轮原始冻结配置 |
| `round1_complete_event.json`、`EXPORT_BOUNDARY.json` | 轮次完成事件与截取边界 |

还原第1轮实际算法阶段源码时，以`runtime/current_frozen/`为基础，再用`runtime/round1_overrides/cef/`覆盖对应两个文件。数据任务更早完成，其对应修复前快照也单独保留。当前研究接口与自动重连是不同时间的改动，不要混为算法贡献。

## 完整性的定义与核查

- 三个第1轮任务目录完整复制，不只提取summary，不丢弃模型错误、失败工具调用或未采用的实验。
- 共用JSONL按第2轮手动启动事件的时间取原始完整行，保留原seq与time；不把第2轮的活动轨迹混进第1轮。
- 文件数量、字节数、原文件SHA-256、发布副本SHA-256和是否脱敏，见[EXPORT_MANIFEST.json](../manifests/EXPORT_MANIFEST.json)。
- 没有包含`.env`或凭据存储。导出时按本机实际密钥值和常见Token格式扫描；本批导出未发现需要遮蔽的凭据。
- 保留源码、模型原回复、逐题结果及本地路径。这是用户要求公开的研究分析材料，请勿误称匿名数据集。

## 运行与分析边界

此仓库首先用于专家离线审阅，**不是在另一台机器上不改路径即可恢复原付费实验的发行包**。日志中的绝对路径、冻结哈希、既有观测缓存和环境身份均保留原样。不要批量替换路径后假装原实验身份完全不变。

当前后端要求Windows x64，Python环境版本见`code/0921/requirements-review.txt`；模型密钥需要自行提供。`.env.example`只有占位符，不含真实密钥。Linux机器可以阅读代码和轨迹，但当前原生参考/Windows Job实现并非开箱跨平台运行。

不要直接把几十至几百MiB的日志整体塞进模型上下文。建议先读项目说明和主流程，再针对某个问题定位同一任务的输入→工具→结果→最终修改；读取实际记录，不用摘要冒充核查。

**本次上传只保存现有事实，不宣称这些组件已经证明提高泛化，也没有重新运行任何一轮搜索。**
