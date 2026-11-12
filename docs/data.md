# 数据契约

设备试验台账计算批次内振动、温度和失效次数，同一设备可有多个运行周期。

主测量 `vibration` 的单位为毫米每秒，主体字段是 `device`，核算维度是 `batch`。`observation_id` 是文件内唯一观测编号，主体允许重复观测。数值只接受有限值；空单元格解析为 None，不能直接当成零。额外列不进入当前核算。

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `observation_id` | str | 必填 |
| `device` | str | 必填 |
| `batch` | str | 必填 |
| `cycle` | int | 必填 |
| `vibration` | float | 必填 |
| `temperature` | float | 必填 |
| `load` | float | 必填 |
| `failed` | int | 必填 |
| `recorded` | int | 必填 |

时间字段 day、week、tick、hour、minute、cycle、start_day、reported_day、known_at、visible_at、opened、observed 均以各字段名所述的相对时间单位表示整数；不同字段并不必然代表同一种时间。缺少的登记码与浓度等信息保持为空。示例只表达产品数据结构，不能代替实际调查资料。

## 业务字段语义

vibration 为毫米每秒，temperature 为摄氏度，load 为合成负荷百分数。failed 为该周期失效标志。cycle 是预测周期，recorded 是记录可见周期，两者不能互换。device 属于一个 batch，可跨周期出现。

## 训练契约

`train` 流程在 summary 口径之外新增，不改变 summary 的任何数值含义。

- 分组边界：device 是不可拆分单元，外层每次留出一整个 batch；同一 device 不会同时出现在任一折的两侧。
- 时间可用性：仅当 `recorded <= cycle` 时记录才在预测时刻可见，可用于特征；晚到记录不进入任何拟合，预测时逐条拒绝并给出原因。
- 字段白名单：默认特征为 vibration、temperature、load；`--extra-field` 可加入额外列，但 failed、observation_id、device、batch、recorded 永远禁止作为特征。
- 内层选择：特征子集（单变量筛选的前 k 项，常数特征直接出局）、缺值策略（均值/中位数）与模型参数（逻辑回归的 L2 强度、k 近邻的 k）只在内层按 device 分组的折上比较；阈值在内层样本外分数上按 `miss_cost * 漏报数 + 误报数` 选取，成本相同取更高阈值。
- 外层指标：逐折报告校准分箱与 Brier、漏报率（漏报/实际失效）、少数类覆盖（被判对的少数类 / 少数类总数，被拒绝的少数类计入分母）、混淆矩阵与 AUC；特征稳定性给出入围频率与折间平均 Jaccard。
- 降级状态：某折训练侧或测试侧只剩单类时，该折标记 `degraded:train_single_class` 或 `degraded:test_single_class`，保留在报告中并列入 `degraded_folds`，聚合均值只在 ok 折上计算且明示 `aggregated_over`。
- 模型产物：包含字段白名单、时间可用性规则、缺值填充值、标准化参数、模型参数与阈值；`predict` 加载后对新批次逐条给概率与报警判定，或对晚到、缺字段、字段值无效的记录给出拒绝原因。
