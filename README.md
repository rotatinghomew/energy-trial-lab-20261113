# energy_trial_lab

归档终端实验台的测量记录，检查仪器基线并汇总各运行配置的观测。

当前公开命令是 `inspect`，用于核对数据和查看已记录的观测，不对观测结果作超出数据支持的推断。输入和输出使用 UTF-8 JSON。示例数据均为本地虚构记录，不连接终端、供应方或个人账户。

## 安装与运行

使用 Python 3.11.2，依赖精确版本在 `requirements.txt`。依赖可预先装好；运行和测试不执行下载。

```bash
python3 -m pip install -r requirements.txt
python3 -m compileall -q energy_trial_lab
python3 -m unittest discover -s tests -v
python3 -m energy_trial_lab inspect --input fixtures/trials.json --output /tmp/energy-observed.json
```

正常运行会写出完整结果文件，标准输出给出保存位置。格式冲突、数据边界错误或缺失关键字段会以非零状态退出；不要将失败输出当作空业务结果。复用临时目录前应清理同名输出与数据库，以免混淆不同运行。

## 数据与目录

`energy_trial_lab/measurements.py` 提供当前观测计算；`energy_trial_lab/io.py` 负责严格 JSON 读取和原子输出；`fixtures/` 是可重放的演练记录；`tests/` 验证领域边界和输入输出行为。

duration_ms 是本次观测的墙钟时间，power_mw 为平均功率；能量单位 mJ，功率乘毫秒需除以 1000。baseline 的 se_mw 描述同一批次共享的校正不确定性。completed 是结局，deadline_ms 是未完成测量的停止点，不是完成时延。

演练规模可在普通 CPU 上完成，运行说明不要求外部服务或真实设备。输出中的未定义值使用 JSON null，不用零替代。涉及版本的输入应保留原文件，方便复查当时的事实。
