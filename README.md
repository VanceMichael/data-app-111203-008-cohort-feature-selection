# cohort-feature-selection

设备试验台账计算批次内振动、温度和失效次数，同一设备可有多个运行周期。

## 使用

运行需要 Python 3.11 或更新版本，没有第三方运行依赖。数据在本地处理，不请求外部接口。

```bash
python3 -m analysis_app summary
python3 -m analysis_app summary --input data/observations.csv --output output/summary.json
```

标准输出是完整 JSON，输出文件使用临时文件原子替换；失败退出码为 2，不覆盖既有结果。空数据集保留零计数与空分组，缺失数值不进入数值分母。数据指纹由按 observation_id 排序后的解析记录计算。

## 训练、预测与泄漏演示

`train` 以设备为不可拆分单元做嵌套评估：外层留出一整个 batch，内层只做特征子集、缺值策略与模型参数的选择，并按预设漏报成本在内层样本外分数上定阈值，不看外层标签。recorded 晚于 cycle 的记录不进入任何拟合。候选模型至少包含一个可解释的线性分类器（逻辑回归）和一个非线性候选（k 近邻）。

```bash
python3 -m analysis_app train --model output/model.json --report output/train_report.json
python3 -m analysis_app predict --model output/model.json --input data/observations.csv --output output/predictions.json
python3 -m analysis_app demo
```

报告逐外层折给出校准分箱、漏报率、少数类覆盖与所选特征，并汇总特征选择频率与折间 Jaccard 稳定性；某折只剩单类时标记可验证的降级状态，不静默跳过。模型产物带字段白名单、时间可用性规则与全部拟合变换，`predict` 对新批次给概率，或对每条记录给出拒绝原因。`demo` 用合成重复设备演示随机拆分如何被欺骗，以及分组、晚到记录与常数特征为何穿不透选择边界。

## 验证与容器

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q analysis_app
docker build -t cohort-feature-selection .
docker run --rm cohort-feature-selection
```

容器内也可执行 `python -m unittest discover -s tests -v`。镜像构建阶段取得 Python 镜像，产品启动、测试和演示均无需下载。小规模样例的完整单元测试和 summary 可以连续执行，不需要等待定时任务。

## 代码与数据

- `analysis_app/io.py`：有类型的 CSV 读取、完整性校验及内容指纹。
- `analysis_app/schema.py`：字段类型、主体与聚合维度。
- `analysis_app/domain.py`：当前业务核算函数。
- `analysis_app/summary.py`：确定性的分组与数值口径。
- `analysis_app/ml.py`：拟合变换、线性与非线性模型、阈值与 AUC 原语。
- `analysis_app/train.py`：嵌套分组评估、内层选择与模型产物构建。
- `analysis_app/artifact.py`：产物加载与对新批次的概率/拒绝预测。
- `analysis_app/synthetic.py`：合成重复设备的泄漏演示数据。
- `analysis_app/cli.py`：本地命令及原子输出。
- `docs/data.md`：数据契约。

演示数据全部为本地合成，不对应现实机构、设备或人员；不能用来推断真实业务效果。
