# PoreMind

[English](#english) | [中文](#中文)

## English

API-oriented single-molecule nanopore analysis toolkit supporting a stepwise multi-sample workflow:

1. Signal denoising (multiple methods)
2. Event detection (multiple methods)
3. Event feature extraction (built-in + custom)
4. Abnormal event filtering (`noise` labeling)
5. Multi-model 10-fold comparison and best-model selection
6. Event-level classification for new samples

### Documentation

- Framework (English, default): `docs/nanopore_single_molecule_framework.md`
- 快速上手与接口示例（中文）：见下方“中文”部分与本文示例章节

---

## 中文

面向 API 的单分子纳米孔分析工具，支持多样本逐步骤流程：

1. 电信号降噪（多方法）
2. 事件检测（多方法）
3. 事件特征提取（内置 + 自定义）
4. 异常事件过滤（noise 标注）
5. 多模型 10 折比较并选择最优模型
6. 新样本逐事件分类

### 文档介绍

- 方法框架（英文，默认）：`docs/nanopore_single_molecule_framework.md`
- 使用说明（中文）：本文安装与快速用法章节

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
# 可选方向：detect_direction="down"（默认）或 "up"
# 可选基线：baseline_method="global_quantile", baseline_params={"q": 0.5}
# 可选事件合并：merge_event=True, merge_event_params={"merge_gap_ms": 0.2}
# 可选统计电流过滤：exclude_current=True, exclude_current_params={"min": 0, "max": None}

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
analysis.plot.event_current_simple(sample_id=None, current="denoise", start_event=1, end_event=5, ylim=None)
analysis.plot.event_current(sample_id=None, current="denoise", start_event=1, end_event=5, ylim=None)

features = analysis.extract_features(max_event_per_sample=None)  # 每样本可限制提取前N个事件；None表示全部
analysis.filter_events(method="blockade_gmm", parameters={"n_components": 2, "prior_mean": None}, blockage_lim=(0.1, 1.0))
analysis.do_pca(feature_cols=["duration_s", "blockade_ratio"], data="filtered")
analysis.do_tsne(feature_cols=["duration_s", "blockade_ratio"], data="filtered")
# analysis.do_umap(feature_cols=["duration_s", "blockade_ratio"], data="filtered")  # 需安装 umap-learn
best_pkg = analysis.build_best_model(cv=10, scoring="accuracy")
analysis.pl.model_cm(model_name=best_pkg["best_model"], split="test")
analysis.pl.model_metric_bar(metric="accuracy", split="test")
analysis.pl.plot_2d(data="filtered", value="label")
analysis.pl.plot_3d(data="filtered", value="label")
analysis.pl.stacked_bar(group_col="sample_id", value_col="label", data="filtered")
analysis.pl.box_significance(group_col="label", value_col="blockade_ratio", data="filtered", method="ttest")
dl_pkg = analysis.build_DL_model(model_name="1D-CNN", device="cuda", epoch=30, batch_size=64)
analysis.pl.plot_fold_loss(model_name="1D-CNN", type="train")
new_analysis, pred = analysis.classify_new_samples(
    {"unknown_01": "unknown_01.abf"},
    reader="abf",
    custom_feature_fns={"seg": lambda x: {"ptp": float(x.max() - x.min())}},
    model="Random Forest",
)
# 同一接口也支持DL模型：
new_analysis_dl, pred_dl = analysis.classify_new_samples({"unknown_01": "unknown_01.abf"}, reader="abf", model="1D-CNN")
# new_analysis 可直接用于可视化：new_analysis.plot.event_current(...) / new_analysis.pl.plot_2d(...)
# pred 中会包含 pred_label 以及每个类别的概率列（pred_proba_<class>）
```

> 说明：ABF 模式默认会遍历该文件全部 channel 与 sweep，并在事件表中输出 `channel`、`sweep` 列。
> 默认降噪方法为 `butterworth_filtfilt`（零相位滤波，不引入相位延迟），需安装 `scipy`。
> 事件检测支持 `threshold`、`zscore_threshold`、`cusum`、`pelt`、`hmm`，并提供默认参数。
> 事件方向支持 `detect_direction="down"`（向下事件，默认）或 `detect_direction="up"`（向上事件）。
> 基线支持 `baseline_method="global_quantile"`，并通过 `baseline_params={"q": xx}` 指定全局分位数（默认 `q=0.5`，即全局中位值）。
> 支持事件合并：`merge_event=True` + `merge_event_params={"merge_gap_ms": xx}` 可合并时间间隔不超过 `xx` ms 的临近事件。
> 默认 `min_duration_s=0`；`rolling_quantile` 默认参数为 `window=10000, q=0.5`。
> 默认噪声尺度估计为 `noise_method="mad"`（可切换为 `std`）。
> 默认启用 `exclude_current=True`：`up` 方向默认统计区间 `(-inf, 0)`；`down` 方向默认统计区间 `(0, +inf)`；若过滤后有效点 `<=1` 会直接报错。
> `extract_features` 中 `delta_i` 与 `blockade_ratio` 会根据 `detect_direction` 做方向一致化计算（`up` 使用负号展开形式）。
> `filter_events` 通过 `method + parameters` 配置不同过滤方法；默认 `blockade_gmm`，默认参数为 `n_components=2, prior_mean=None`。
> `isolation_forest` / `lof` 默认使用特征：`duration_s`、`blockade_ratio`、`segment_skew`、`segment_kurt`。
> `blockage_lim` 默认为 `(0.1, 1.0)`，会先作为硬阈值过滤事件；`filter_events` 会在 `analysis.feature_df` 内新增 `quality_tag`，并将 `analysis.filtered_df` 仅保留 `valid` 事件。
> 提供 `detect_events_simple` 便于在局部时间窗口做初步方法选择与参数调整。
> `detect_events_simple` 的结果会保存到 `analysis.detect_events_simple_object`（并兼容 `analysis.simple_events`）。
> `detect_events` / `detect_events_simple` 会按样本显示进度条（若环境安装了 `tqdm`）。
> 默认建模候选包含 RF / LR / SVM / MLP / ElasticNet / Lasso / 决策树 / LDA / AdaBoost / 高斯朴素贝叶斯。
> 提供 `analysis.plot` 作为 `analysis.pl` 的别名；并支持 `analysis.plot.event_current_simple` / `analysis.plot.event_current` 对事件范围进行电流可视化（红色虚线标注事件起止）。
> 同时支持 `analysis.pl.plot_2d` / `analysis.pl.plot_3d`（并兼容 `getattr(analysis.pl, "2d_plot")` / `getattr(analysis.pl, "3d_plot")`）进行2D/3D特征可视化。

完整逐步 notebook：`notebooks/step_by_step_analysis.ipynb`
