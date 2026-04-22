from __future__ import annotations

from pathlib import Path

import pandas as pd

from .controller import AnalysisController


def _parse_mapping_text(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (text or "").splitlines():
        if not line.strip() or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _to_file_map(files: list | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in files or []:
        p = Path(getattr(item, "name", str(item)))
        mapping[p.stem] = str(p)
    return mapping


def create_app():
    try:
        import gradio as gr
    except Exception as exc:  # pragma: no cover
        raise ImportError("Gradio is required for UI. Install with `pip install gradio`.") from exc

    ctl = AnalysisController()

    with gr.Blocks(title="PoreMind Local Analysis UI") as demo:
        gr.Markdown("# PoreMind 本地交互式分析\n架构：UI 层 + 应用服务层 + 原算法层")

        with gr.Tab("1) 数据导入"):
            file_input = gr.File(label="上传 ABF/CSV 文件", file_count="multiple")
            reader = gr.Dropdown(choices=["abf", "csv"], value="abf", label="reader")
            sample_to_group_txt = gr.Textbox(label="sample_to_group（每行 sample:group）", lines=6)
            load_btn = gr.Button("加载样本")
            load_summary = gr.JSON(label="加载摘要")
            sample_df = gr.Dataframe(label="样本列表")

        with gr.Tab("2) 信号预处理"):
            denoise_method = gr.Dropdown(
                choices=["butterworth_filtfilt", "moving_average", "median", "drift_corrected_moving_average", "none"],
                value="butterworth_filtfilt",
                label="降噪方法",
            )
            denoise_params = gr.Textbox(label="参数 JSON", value='{"filtfilt_N": 2, "filtfilt_Wn": 0.1}')
            denoise_btn = gr.Button("运行降噪")
            denoise_result = gr.JSON(label="预处理状态")

        with gr.Tab("3) 事件检测调参"):
            detect_method = gr.Dropdown(
                choices=["threshold", "zscore_threshold", "cusum", "pelt", "hmm"],
                value="threshold",
                label="检测方法",
            )
            detect_direction = gr.Dropdown(choices=["down", "up"], value="down", label="detect_direction")
            baseline_method = gr.Dropdown(
                choices=["rolling_quantile", "global_quantile", "global_median"],
                value="rolling_quantile",
                label="baseline method",
            )
            detect_params = gr.Textbox(label="detect_params JSON", value='{"sigma_k": 5.0, "min_duration_s": 0.0, "noise_method": "mad"}')
            baseline_params = gr.Textbox(label="baseline_params JSON", value='{"window": 10000, "q": 0.5}')
            sample_id = gr.Textbox(label="预览 sample_id（可空）")
            start_ms = gr.Number(label="start_ms", value=0)
            end_ms = gr.Number(label="end_ms", value=1000)
            preview_btn = gr.Button("阶段A：局部快速调参 detect_events_simple")
            apply_all_btn = gr.Button("阶段B：应用全部样本 detect_events")
            detect_result = gr.JSON(label="检测结果统计")

        with gr.Tab("4) 特征与过滤"):
            extract_btn = gr.Button("执行 extract_features")
            feature_df = gr.Dataframe(label="feature_df")
            filter_method = gr.Dropdown(choices=["blockade_gmm", "isolation_forest", "lof"], value="blockade_gmm", label="filter method")
            filter_params = gr.Textbox(label="filter params JSON", value="{}")
            filter_btn = gr.Button("执行 filter_events")
            filtered_df = gr.Dataframe(label="filtered_df")

        with gr.Tab("5) 降维与可视化"):
            dimred_method = gr.Dropdown(choices=["pca", "tsne", "umap"], value="pca", label="方法")
            dimred_cols = gr.Textbox(label="特征列（逗号分隔，空=自动）")
            dimred_btn = gr.Button("执行降维")
            dimred_df = gr.Dataframe(label="降维结果")

        with gr.Tab("6) 模型训练"):
            feature_cols = gr.Textbox(label="训练特征列（逗号分隔，空=默认）")
            cv = gr.Slider(2, 10, value=5, step=1, label="CV folds")
            scoring = gr.Dropdown(choices=["accuracy", "f1", "recall"], value="accuracy", label="评分")
            train_btn = gr.Button("训练模型")
            train_result = gr.JSON(label="训练结果")

        with gr.Tab("7) 新样本预测"):
            pred_files = gr.File(label="上传未知样本", file_count="multiple")
            predict_btn = gr.Button("classify_new_samples")
            pred_df = gr.Dataframe(label="预测结果")

        with gr.Tab("8) 导出与复现"):
            export_dir = gr.Textbox(label="输出目录", value="./ui_exports")
            export_btn = gr.Button("导出CSV/参数/分析脚本")
            export_result = gr.JSON(label="导出结果")

        def do_load(files, reader_name, group_text):
            sample_paths = _to_file_map(files)
            sample_to_group = _parse_mapping_text(group_text)
            out = ctl.load_samples(sample_paths=sample_paths, sample_to_group=sample_to_group, reader=reader_name)
            return out["summary"], out["sample_df"]

        def do_denoise(method, params_json):
            import json

            params = json.loads(params_json or "{}")
            return ctl.run_denoise(method=method, **params)

        def do_detect(stage, d_method, d_params, b_method, b_params, sid, s_ms, e_ms, direction):
            import json

            return ctl.run_detect(
                stage=stage,
                detect_method=d_method,
                detect_params=json.loads(d_params or "{}"),
                baseline_method=b_method,
                baseline_params=json.loads(b_params or "{}"),
                sample_id=(sid or None),
                start_ms=float(s_ms),
                end_ms=float(e_ms),
                detect_direction=direction,
            )

        def do_extract():
            return ctl.extract_features()

        def do_filter(method, params_json):
            import json

            return ctl.filter_events(method=method, parameters=json.loads(params_json or "{}"))

        def do_dimred(method, cols_text):
            cols = [c.strip() for c in (cols_text or "").split(",") if c.strip()] or None
            return ctl.do_dimensionality_reduction(method=method, feature_cols=cols)

        def do_train(cols_text, cv_n, score):
            cols = [c.strip() for c in (cols_text or "").split(",") if c.strip()] or None
            return ctl.train_model(feature_cols=cols, cv=int(cv_n), scoring=score)

        def do_predict(files):
            return ctl.predict_new(new_sample_paths=_to_file_map(files))

        def do_export(path_text):
            out_dir = Path(path_text)
            tables = ctl.export_tables(out_dir)
            params_json = ctl.export_params_json(out_dir / "params_snapshot.json")
            script = ctl.export_analysis_script(out_dir / "reproduce_analysis.py")
            return {"tables": tables, "params_json": params_json, "script": script}

        load_btn.click(do_load, inputs=[file_input, reader, sample_to_group_txt], outputs=[load_summary, sample_df])
        denoise_btn.click(do_denoise, inputs=[denoise_method, denoise_params], outputs=[denoise_result])
        preview_btn.click(
            lambda *args: do_detect("preview", *args),
            inputs=[detect_method, detect_params, baseline_method, baseline_params, sample_id, start_ms, end_ms, detect_direction],
            outputs=[detect_result],
        )
        apply_all_btn.click(
            lambda *args: do_detect("global", *args),
            inputs=[detect_method, detect_params, baseline_method, baseline_params, sample_id, start_ms, end_ms, detect_direction],
            outputs=[detect_result],
        )
        extract_btn.click(do_extract, outputs=[feature_df])
        filter_btn.click(do_filter, inputs=[filter_method, filter_params], outputs=[filtered_df])
        dimred_btn.click(do_dimred, inputs=[dimred_method, dimred_cols], outputs=[dimred_df])
        train_btn.click(do_train, inputs=[feature_cols, cv, scoring], outputs=[train_result])
        predict_btn.click(do_predict, inputs=[pred_files], outputs=[pred_df])
        export_btn.click(do_export, inputs=[export_dir], outputs=[export_result])

    return demo


def main() -> None:
    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()
