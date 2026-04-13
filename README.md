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

## 快速用法（ABF 输入）

```python
from poremind import create_analysis_object

sample_paths = {
    "std_A_01": "std_A_01.abf",
    "std_B_01": "std_B_01.abf",
}
sample_to_group = {
    "std_A_01": "A",
    "std_B_01": "B",
}

analysis = create_analysis_object(
    sample_paths,
    sample_to_group=sample_to_group,
    reader="abf",
).load()

analysis.denoise()  # 默认 butterworth_filtfilt
analysis.detect_events(detect_method="threshold")  # 可选: threshold / zscore_threshold / cusum / pelt / hmm

# 绘图模块（pl）：默认显示 0-1 ms
analysis.pl.current(sample_id=None, current="denoise", start_ms=0.0, end_ms=1.0, width=10, height=3)

features = analysis.extract_features()
filtered = analysis.filter_events(method="isolation_forest", contamination=0.05)
best_pkg = analysis.build_best_model(cv=10)
pred = analysis.classify_new_samples({"unknown_01": "unknown_01.abf"}, reader="abf")
```

> 说明：ABF 模式默认会遍历该文件全部 channel 与 sweep，并在事件表中输出 `channel`、`sweep` 列。
> 默认降噪方法为 `butterworth_filtfilt`（零相位滤波，不引入相位延迟），需安装 `scipy`。
> 事件检测支持 `threshold`、`zscore_threshold`、`cusum`、`pelt`、`hmm`，并提供默认参数。

完整逐步 notebook：`notebooks/step_by_step_analysis.ipynb`
