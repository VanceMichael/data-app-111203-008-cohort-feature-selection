# 数据契约

设备试验台账计算批次内振动、温度和失效次数，同一设备可有多个运行周期。

主测量 `vibration` 的单位为毫米每秒，主体字段是 `device`，核算维度是 `batch`。`observation_id` 是文件内唯一观测编号，主体允许重复观测。数值只接受有限值；空单元格解析为 None，不能直接当成零。额外列不进入当前核算。

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `observation_id` | str | 必填 |
| `device` | str | 必填 |
| `batch` | str | 必填 |
| `cycle` | int | 允许空值，空值不是零 |
| `vibration` | float | 必填 |
| `temperature` | float | 允许空值，空值不是零 |
| `load` | float | 允许空值，空值不是零 |
| `failed` | int | 允许空值，空值不是零 |
| `recorded` | int | 允许空值，空值不是零 |

时间字段 day、week、tick、hour、minute、cycle、start_day、reported_day、known_at、visible_at、opened、observed 均以各字段名所述的相对时间单位表示整数；不同字段并不必然代表同一种时间。缺少的登记码与浓度等信息保持为空。示例只表达产品数据结构，不能代替实际调查资料。
