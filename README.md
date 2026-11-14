# energy_trial_lab

归档终端实验台的测量记录，检查仪器基线并汇总各运行配置的观测。

当前公开命令有两个：

- `inspect` 用于核对数据和查看已记录的观测，不对观测结果作超出数据支持的推断。
- `frontier` 用于多目标选型：把每任务净能耗、完成时延和完成概率放在一起，区分“观测到的优势”与“不确定性下仍然可靠的优势”。

输入和输出使用 UTF-8 JSON。示例数据均为本地虚构记录，不连接终端、供应方或个人账户。

## 安装与运行

使用 Python 3.11.2，依赖精确版本在 `requirements.txt`。依赖可预先装好；运行和测试不执行下载。

```bash
python3 -m pip install -r requirements.txt
python3 -m compileall -q energy_trial_lab
python3 -m unittest discover -s tests -v
python3 -m energy_trial_lab inspect --input fixtures/trials.json --output /tmp/energy-observed.json
python3 -m energy_trial_lab frontier --input fixtures/trials.json --output /tmp/energy-frontier.json
```

正常运行会写出完整结果文件，标准输出给出保存位置。格式冲突、数据边界错误或缺失关键字段会以非零状态退出；不要将失败输出当作空业务结果。复用临时目录前应清理同名输出与数据库，以免混淆不同运行。

## 数据与目录

`energy_trial_lab/measurements.py` 提供基础观测计算；`energy_trial_lab/frontier.py` 提供多目标选型分析；`energy_trial_lab/io.py` 负责严格 JSON 读取和原子输出；`fixtures/` 是可重放的演练记录；`tests/` 验证领域边界和输入输出行为。

duration_ms 是本次观测的墙钟时间，power_mw 为平均功率；能量单位 mJ，功率乘毫秒需除以 1000。baseline 的 se_mw 描述同一批次共享的校正不确定性。completed 是结局，deadline_ms 是未完成测量的停止点，不是完成时延。

演练规模可在普通 CPU 上完成，运行说明不要求外部服务或真实设备。输出中的未定义值使用 JSON null，不用零替代。涉及版本的输入应保留原文件，方便复查当时的事实。

## 选型分析的口径

`frontier` 不把三个指标压成一个加权分数，也不对各指标分别排序后拼出前沿。要点：

- **固定共同支持集**：每个任务先取“所有参与分析的配置都测过”的 `(device_id, task, batch_id)` 区组。多组目标负载只改变任务权重，不改变这份支持集，因此不会针对某个负载挑选有利样本。
- **每任务净能耗**：沿用 `(power_mw - baseline.power_mw) × duration_ms / 1000`，单位 mJ；负载按 `Σ 权重 × 每任务净能耗` 汇总，不对跨任务瓦数取平均。
- **完成时延**：用右删失的受限平均生存时间（RMST）。取消或超时的记录在观测停止点后删失，仍进入完成率和能耗分母；`deadline_ms` 不作为完成时延，也不外推到观测范围之外。
- **配对与不确定性**：同一区组内的配置成对比较；bootstrap 按“任务 × 批次”分层重抽区组，同批次共享的空载校正偏移在同一次重抽样中一起传播。
- **联合优势**：支配概率是同一 replicate 内三目标同时成立的比例，不是三个边际概率的乘积。最低完成率按硬门槛处理：先看双方是否达标，再在达标场景中比较能耗与时延。
- **三类结论**：`robust_keep`（所有负载下稳定保留）、`safe_eliminate`（有同一支配者且证据链完整）、`conditional_keep`（只在某些负载或放宽门槛时值得保留）。
- **可复查证据**：被淘汰配置保留逐任务、逐批次的配对差值、粗平均与配对方向是否相反、空载校正 ±1.96·se 是否改变符号；缺测关系原样保留在 `comparability` 中。

输出包含 `accounting`、`comparability`、`frontiers`、`robustness` 和 `measurement_gaps` 五个部分。`measurement_gaps` 按“先补共同支持缺口，再补决定完成率门槛的删失单元，最后补口径元数据”的顺序给出下一类试验建议。
