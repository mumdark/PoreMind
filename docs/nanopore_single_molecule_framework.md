# 单分子纳米孔电流分析工具框架设计（v0.1）

## 1. 目标与范围

本框架面向**离线分析 + 准实时处理**两种模式，覆盖完整链路：

1. 电流数据读入（ABF/CSV/HDF5/自定义采样卡格式）
2. 降噪与漂移校正
3. 基线估计与事件检测（阻断事件识别）
4. 事件级与轨迹级特征提取
5. 分类与异常检测（规则 + 机器学习）
6. 可视化、复核与报告导出

---

## 2. 总体架构（分层 + 插件）

建议采用“**数据层—算法层—应用层**”三层结构，算法层全部插件化。

### 2.1 数据层
- `io/reader`: 统一输入适配器（Reader Interface）
- `io/schema`: 标准内部数据模型（时间戳、电流、采样率、元数据）
- `storage/cache`: 原始波形缓存、事件索引缓存、特征表缓存

### 2.2 算法层
- `preprocess`: 去尖峰、带通/低通、小波、漂移去除
- `baseline`: 滑窗鲁棒基线、分段基线、EM/状态空间基线
- `event_detect`: 阈值法、CUSUM、变点检测、HMM分段
- `feature`: 事件持续时间、阻断幅值、面积、上升/恢复时间、谱特征
- `model`: 分类器（XGBoost/SVM/1D-CNN）、聚类、OOD检测

### 2.3 应用层
- `pipeline/orchestrator`: DAG管线编排（可配置 YAML）
- `ui`: 波形查看、事件回放、人工标注校正
- `report`: PDF/HTML报告（参数、图、混淆矩阵、ROC）
- `service`: CLI + REST API

---

## 3. 核心数据结构（建议）

### 3.1 Trace（连续电流轨迹）
- `trace_id`
- `time[]` / `current[]`
- `sampling_rate`
- `voltage`, `buffer_info`, `pore_id`, `temperature`
- `preprocess_meta`（滤波器参数、版本）

### 3.2 Event（候选阻断事件）
- `event_id`, `trace_id`
- `start_idx`, `end_idx`
- `baseline_local`
- `delta_I`（阻断深度）
- `dwell_time`
- `snr`
- `quality_flag`

### 3.3 FeatureVector
- 统计特征：mean/std/skew/kurtosis
- 形态特征：rise/fall time, plateau ratio
- 频域特征：主频、频带能量、谱熵
- 任务特征：多级台阶数、子事件计数

---

## 4. 推荐算法流程（可先做 MVP）

### Stage A：预处理
1. 抗工频与高频噪声：IIR/FIR 低通（按采样率自适应）
2. 去孤立尖峰：Hampel 或中值滤波
3. 漂移校正：大窗口 LOESS 或形态学开运算

### Stage B：基线与事件检测
1. 先做鲁棒基线（分位数滑窗，如 P50/P60）
2. 计算残差 `r(t)=I(t)-baseline(t)`
3. 候选事件触发：
   - 绝对阈值：`r(t) < -k * sigma_local`
   - 变点确认：CUSUM/PELT
4. 事件后处理：最短持续时间、最小深度、事件合并

### Stage C：特征提取与分类
1. 单事件特征 + 上下文特征（前后窗口稳定性）
2. 先上可解释模型（XGBoost/LightGBM）
3. 数据量上来后再上 1D-CNN/Transformer 编码器
4. 增加不确定性输出（温度缩放/MC Dropout）

### Stage D：质量控制
- 轨迹级 QC：噪声 RMS、漂移斜率、有效事件率
- 模型级 QC：类别召回、PR-AUC、校准误差 ECE
- 运行级 QC：吞吐量（events/s）、端到端延迟

---

## 5. 与参考文献（PMC8546757）的一致性映射

该综述强调纳米孔信号处理中“**从预处理、事件检测、特征工程到机器学习决策**”的系统化流程。你的工具可以按以下方式落地：

- 文献里的传统信号处理思路 -> 对应本框架 `preprocess/baseline/event_detect` 模块
- 文献里的特征与判别任务 -> 对应 `feature/model` 模块
- 文献提到模型评估与泛化问题 -> 对应 `quality control + 数据版本管理`

即：先保证“可解释、可复现”的经典流程，再逐步引入深度学习模型做性能提升。

---

## 6. 工程实现建议（Python 技术栈）

- 数值计算：`numpy`, `scipy`, `pandas`
- 信号处理：`scipy.signal`, `pywt`, `ruptures`
- 机器学习：`scikit-learn`, `xgboost`, `pytorch`
- 管线配置：`pydantic + yaml`
- 存储：`parquet`（特征）、`zarr/hdf5`（波形）
- 可视化：`plotly`（交互），`matplotlib`（报告）
- MLOps：`mlflow`（实验追踪）

---

## 7. 目录骨架（建议）

```text
nanopore_tool/
  pyproject.toml
  src/nanopore_tool/
    io/
    preprocess/
    baseline/
    detect/
    feature/
    model/
    pipeline/
    qc/
    viz/
    cli/
  configs/
    mvp_default.yaml
    high_sensitivity.yaml
  tests/
    test_io.py
    test_detect.py
    test_feature.py
```

---

## 8. MVP 里程碑（4 周）

### Week 1
- 统一 Reader + Trace/Event 数据结构
- 基础滤波与可视化

### Week 2
- 基线估计 + 阈值/CUSUM 事件检测
- 事件人工复核界面（最简版）

### Week 3
- 特征提取 + XGBoost 分类
- 指标评估脚本（ROC/PR/混淆矩阵）

### Week 4
- CLI + 配置化运行
- 报告导出 + 批处理

---

## 9. 你这个项目最容易踩的坑

1. **训练集标签不一致**：建议双人标注 + 仲裁。
2. **不同实验批次漂移严重**：必须做批次归一化/域适配。
3. **只看准确率**：要重点看召回率和误检成本。
4. **过早上深度模型**：先把基线识别和事件定义做扎实。

---

## 10. 下一步（可直接执行）

1. 先沉淀 20–50 条代表性轨迹作为基准数据集。
2. 先固定 1 套基线 + 1 套检测参数，形成“可复现实验基线”。
3. 再对比 3 类检测器（阈值/CUSUM/变点），用同一标注集打分。
4. 最后再进入特征选择与模型迭代。

