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
conda create -n poremind python=3.10 -y
conda activate poremind
pip install -e . # -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
pip install pyabf 
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

# 仅在局部时间窗口快速调参（默认 0-1000 ms）
simple_events = analysis.detect_events_simple(
    sample_id=None,
    current="denoise",
    start_ms=0.0,
    end_ms=1000.0,
    detect_method="threshold",
)

# 绘图模块（pl）：默认显示 0-1 ms
analysis.pl.current(sample_id=None, current="denoise", start_ms=0.0, end_ms=1.0, width=10, height=3)
analysis.plot.event_current_simple(sample_id=None, current="denoise", start_event=1, end_event=5)
analysis.plot.event_current(sample_id=None, current="denoise", start_event=1, end_event=5)

features = analysis.extract_features()  # 含 segment_skew / segment_kurt / peak_factor 等特征
filtered = analysis.filter_events()  # 默认 blockade_gmm(基于 blockade_ratio 单特征)
best_pkg = analysis.build_best_model(cv=10, scoring="accuracy")
analysis.pl.model_cm(model_name=best_pkg["best_model"], split="test")
analysis.pl.model_metric_bar(metric="accuracy", split="test")
analysis.pl.plot_2d(data="filtered", value="label")
analysis.pl.plot_3d(data="filtered", value="label")
pred = analysis.classify_new_samples({"unknown_01": "unknown_01.abf"}, reader="abf")
```

> 说明：ABF 模式默认会遍历该文件全部 channel 与 sweep，并在事件表中输出 `channel`、`sweep` 列。
> 默认降噪方法为 `butterworth_filtfilt`（零相位滤波，不引入相位延迟），需安装 `scipy`。
> 事件检测支持 `threshold`、`zscore_threshold`、`cusum`、`pelt`、`hmm`，并提供默认参数。
> 提供 `detect_events_simple` 便于在局部时间窗口做初步方法选择与参数调整。
> `detect_events_simple` 的结果会保存到 `analysis.detect_events_simple_object`（并兼容 `analysis.simple_events`）。
> 默认建模候选包含 RF / LR / SVM / MLP / ElasticNet / Lasso / 决策树 / LDA / AdaBoost / 高斯朴素贝叶斯。
> 提供 `analysis.plot` 作为 `analysis.pl` 的别名；并支持 `analysis.plot.event_current_simple` / `analysis.plot.event_current` 对事件范围进行电流可视化（红色虚线标注事件起止）。
> 同时支持 `analysis.pl.plot_2d` / `analysis.pl.plot_3d`（并兼容 `getattr(analysis.pl, "2d_plot")` / `getattr(analysis.pl, "3d_plot")`）进行2D/3D特征可视化。

完整逐步 notebook：`notebooks/step_by_step_analysis.ipynb`
