# 针对首轮探索偏移的修改

备份：`backups/focused_tasks_20261112_004136/`。原实验记录与冻结配置未覆盖。

## 已修改的实际路径

- `cef/task_prompts.py` 按数据/规划/编辑角色构造任务；`tool_agents._run` 直接调用。删除原 COMMON、角色、runtime、效率和新工具指令层层拼接的构造方式。数据任务不再收到 Editor 包格式与 Planner 接续指令。
- 数据任务交付对象明确为本轮可检验的生成提案，或有依据的 keep_current；不要求内部先搜索出最佳结构家族。仍可根据真实结果修正和换方向，无工具次数硬上限。
- 数据任务初始消息现在直接携带 previous_results、已有坐标与逐题关联材料入口、真实生成器源码入口。无需先做一次工具调查才能知道当前题集的实测弱点。
- 研究与正式评分接口保留；低 gap、参考失败、实现错误具有不同含义。前次修复的后端失败分类与坐标/gap材料继续使用。
- 新实验目录为 `experiments/cef_focused_tasks`。每次默认一代后暂停。参考计算并行数从2降为1，减少本机并行参考计算的内存竞争；算法执行仍8并行。这不是证明单个SCIP进程不会耗尽内存。

## 检查与证据

- 初版角色任务重写后50项回归测试通过。
- 加入起始实测材料后，最新33项相关回归测试通过，报告 `acceptance/focused_tasks_20261112/tests.xml`。
- 当前 `runCEFAuto.py --check` 通过。
- 实际 DeepSeek 小规模决策回放两次，原始请求/响应/usage保存在 `acceptance/focused_tasks_20261112/replay/`，摘要在 `live_replay.json`。第一场景识别已有坐标无需重建，第二场景仍选择先读取材料。因此不能声称探索效率已经得到证明。初始实测材料随后接入，尚未再次完成付费行为回放。

## 未证明的部分

本次没有启动正式一代搜索。不能由测试通过推断耗时、Token或最终TSPLIB改善。参考并行数降低只缓解资源竞争，单实例SCIP内存问题仍可能出现。新一轮应观察研究是否围绕实际弱点推进、是否减少特征代理扫描和重复重建，并统计实际开销。

依据：OpenAI GPT-5 prompting guide 的探索倾向与行动条件；Anthropic prompting best practices 的过度探索/过度指令；Anthropic harness-design-long-running-apps 的逐项精简组件实践。模型专属建议不能直接视为DeepSeek行为保证。
