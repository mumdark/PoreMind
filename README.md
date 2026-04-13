# PoreMind

面向 API 的单分子纳米孔分析工具，支持多样本逐步骤流程：

1. 电信号降噪（多方法）
2. 事件检测（多方法）
3. 事件特征提取（内置 + 自定义）
4. 异常事件过滤（noise 标注）
5. 多模型 10 折比较并选择最优模型
6. 新样本逐事件分类

## 安装

```bash
pip install -e .
```

## 快速用法（对象式）

```python
from poremind import create_analysis_object

sample_paths = {
    "std_A_01": "std_A_01.csv",
    "std_B_01": "std_B_01.csv",
}
sample_to_group = {
    "std_A_01": "A",
    "std_B_01": "B",
}

analysis = create_analysis_object(
    sample_paths,
    sample_to_group=sample_to_group,
    reader="csv",
).load()

analysis.denoise(method="drift_corrected_moving_average", drift_window=1001, smooth_window=5)
analysis.detect_events(detect_method="threshold")
features = analysis.extract_features()
filtered = analysis.filter_events(method="isolation_forest", contamination=0.05)
best_pkg = analysis.build_best_model(cv=10)
pred = analysis.classify_new_samples({"unknown_01": "unknown_01.csv"}, reader="csv")
```

完整逐步 notebook：`notebooks/step_by_step_analysis.ipynb`
