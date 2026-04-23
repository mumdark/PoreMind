from __future__ import annotations

from pathlib import Path

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

    with gr.Blocks(title="PoreMind") as demo:
        gr.Markdown("# PoreMind 分析平台")

        with gr.Tab("1) 数据导入"):
            file_input = gr.File(label="上传 ABF/CSV 文件", file_count="multiple")
            reader = gr.Dropdown(choices=["abf", "csv"], value="abf", label="Reader")
            sample_to_group_txt = gr.Textbox(label="sample_to_group（每行 sample:group）", lines=6)
            load_btn = gr.Button("加载样本")
            load_summary = gr.JSON(label="加载摘要")
            sample_df = gr.Dataframe(label="样本信息")

        with gr.Tab("2) 信号预处理"):
            denoise_method = gr.Dropdown(
                choices=["butterworth_filtfilt", "moving_average", "median", "drift_corrected_moving_average", "none"],
                value="butterworth_filtfilt",
                label="降噪方法",
            )
            with gr.Group(visible=True) as grp_bw:
                bw_n = gr.Slider(1, 8, value=2, step=1, label="filtfilt_N")
                bw_wn = gr.Slider(0.01, 0.99, value=0.1, step=0.01, label="filtfilt_Wn")
            with gr.Group(visible=False) as grp_ma:
                ma_window = gr.Slider(1, 101, value=5, step=2, label="window")
            with gr.Group(visible=False) as grp_median:
                median_window = gr.Slider(1, 101, value=5, step=2, label="window")
            with gr.Group(visible=False) as grp_drift:
                drift_window = gr.Slider(51, 5001, value=1001, step=50, label="drift_window")
                smooth_window = gr.Slider(1, 101, value=5, step=2, label="smooth_window")

            denoise_sample = gr.Textbox(label="预览 sample_id（可空）")
            denoise_start = gr.Number(label="start_ms", value=0.0)
            denoise_end = gr.Number(label="end_ms", value=1.0)
            denoise_run_btn = gr.Button("运行降噪并绘图")
            denoise_result = gr.JSON(label="预处理状态")
            denoise_plot = gr.Plot(label="pl.current")

        with gr.Tab("3) 事件检测调参"):
            detect_method = gr.Dropdown(
                choices=["threshold", "zscore_threshold", "cusum", "pelt", "hmm"],
                value="threshold",
                label="检测方法",
            )
            detect_direction = gr.Dropdown(choices=["down", "up"], value="down", label="detect_direction")
            baseline_method = gr.Dropdown(choices=["rolling_quantile", "global_quantile", "global_median"], value="rolling_quantile", label="baseline_method")
            b_window = gr.Slider(11, 30001, value=10000, step=10, label="baseline_window")
            b_q = gr.Slider(0.01, 0.99, value=0.5, step=0.01, label="baseline_q")
            merge_event = gr.Checkbox(label="merge_event", value=False)
            merge_gap_ms = gr.Number(label="merge_gap_ms", value=0.0)
            exclude_current = gr.Checkbox(label="exclude_current", value=True)
            ex_min = gr.Number(label="exclude_current_min (空请填nan)", value=0.0)
            ex_max = gr.Number(label="exclude_current_max (空请填nan)", value=float("nan"))

            with gr.Group(visible=True) as grp_det_threshold:
                d_sigma_k = gr.Number(label="sigma_k", value=5.0)
                d_min_duration = gr.Number(label="min_duration_s", value=0.0)
                d_noise_method = gr.Dropdown(choices=["mad", "std"], value="mad", label="noise_method")
            with gr.Group(visible=False) as grp_det_z:
                z_thr = gr.Number(label="z", value=4.0)
                z_min_duration = gr.Number(label="min_duration_s", value=0.0)
                z_noise_method = gr.Dropdown(choices=["mad", "std"], value="mad", label="noise_method")
            with gr.Group(visible=False) as grp_det_cusum:
                c_drift = gr.Number(label="drift", value=0.02)
                c_threshold = gr.Number(label="threshold", value=8.0)
                c_min_duration = gr.Number(label="min_duration_s", value=0.0)
                c_noise_method = gr.Dropdown(choices=["mad", "std"], value="mad", label="noise_method")
            with gr.Group(visible=False) as grp_det_pelt:
                p_model = gr.Dropdown(choices=["l1", "l2", "rbf"], value="l2", label="model")
                p_penalty = gr.Number(label="penalty", value=8.0)
                p_sigma_k = gr.Number(label="sigma_k", value=3.0)
                p_min_duration = gr.Number(label="min_duration_s", value=0.0)
                p_noise_method = gr.Dropdown(choices=["mad", "std"], value="mad", label="noise_method")
            with gr.Group(visible=False) as grp_det_hmm:
                h_n_components = gr.Slider(2, 6, value=2, step=1, label="n_components")
                h_cov = gr.Dropdown(choices=["diag", "full", "tied", "spherical"], value="diag", label="covariance_type")
                h_n_iter = gr.Slider(20, 500, value=200, step=10, label="n_iter")
                h_min_duration = gr.Number(label="min_duration_s", value=0.0)

            detect_sample = gr.Textbox(label="sample_id（预览可空）")
            detect_current = gr.Dropdown(choices=["denoise", "raw"], value="denoise", label="current")
            detect_start = gr.Number(label="start_ms", value=0.0)
            detect_end = gr.Number(label="end_ms", value=1000.0)
            detect_start_event = gr.Slider(1, 100, value=1, step=1, label="start_event")
            detect_end_event = gr.Slider(1, 200, value=5, step=1, label="end_event")

            preview_btn = gr.Button("阶段A：detect_events_simple")
            preview_result = gr.JSON(label="preview结果")
            preview_plot = gr.Plot(label="pl.event_current_simple")
            preview_table = gr.Dataframe(label="analysis.simple_events")

            global_btn = gr.Button("阶段B：detect_events")
            global_result = gr.JSON(label="global结果")
            global_plot = gr.Plot(label="pl.event_current")
            global_sample = gr.Textbox(label='事件表 sample_id（如 A8__ch0_sw0）')
            global_table = gr.Dataframe(label="analysis.events[sample_id]")

        with gr.Tab("4) 特征提取与可视化"):
            extract_btn = gr.Button("执行 extract_features")
            feat_df = gr.Dataframe(label="feature_df")
            feat_plot_kind = gr.Dropdown(choices=["plot_2d", "plot_3d", "box_significance"], value="plot_2d", label="可视化方法")
            feat_x = gr.Textbox(value="blockade_ratio", label="x")
            feat_y = gr.Textbox(value="duration_s", label="y")
            feat_z = gr.Textbox(value="segment_std", label="z")
            feat_value = gr.Textbox(value="label", label="color/value列")
            feat_group_col = gr.Textbox(value="label", label="box group_col")
            feat_value_col = gr.Textbox(value="blockade_ratio", label="box value_col")
            feat_plot_btn = gr.Button("绘图")
            feat_plot = gr.Plot(label="特征可视化")

        with gr.Tab("5) 过滤与可视化"):
            filter_method = gr.Dropdown(choices=["blockade_gmm", "isolation_forest", "lof"], value="blockade_gmm", label="过滤方法")
            with gr.Group(visible=True) as grp_filter_gmm:
                f_n_components = gr.Slider(2, 6, value=2, step=1, label="n_components")
                f_prior_mean = gr.Number(value=float("nan"), label="prior_mean (nan=none)")
            with gr.Group(visible=False) as grp_filter_if:
                f_if_contam = gr.Slider(0.01, 0.4, value=0.05, step=0.01, label="contamination")
            with gr.Group(visible=False) as grp_filter_lof:
                f_lof_contam = gr.Slider(0.01, 0.4, value=0.05, step=0.01, label="contamination")
            filter_btn = gr.Button("执行 filter_events")
            filtered_df = gr.Dataframe(label="analysis.filtered_df")

            filtered_plot_kind = gr.Dropdown(choices=["plot_2d", "plot_3d", "box_significance"], value="plot_2d", label="可视化方法")
            filtered_plot_btn = gr.Button("绘图")
            filtered_plot = gr.Plot(label="过滤后可视化")

        with gr.Tab("6) 降维分析"):
            dr_method = gr.Dropdown(choices=["pca", "tsne", "umap"], value="pca", label="降维方法")
            with gr.Group(visible=True) as grp_dr_pca:
                dr_random_state = gr.Slider(0, 9999, value=42, step=1, label="random_state")
            with gr.Group(visible=False) as grp_dr_tsne:
                tsne_perplexity = gr.Slider(2, 80, value=30, step=1, label="perplexity")
                tsne_iter = gr.Slider(250, 3000, value=1000, step=50, label="n_iter")
            with gr.Group(visible=False) as grp_dr_umap:
                umap_neighbors = gr.Slider(2, 100, value=15, step=1, label="n_neighbors")
                umap_min_dist = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="min_dist")
            dr_cols = gr.Textbox(label="特征列（逗号分隔，空=自动）")
            dr_data = gr.Dropdown(choices=["filtered", "feature"], value="filtered", label="数据源")
            dr_btn = gr.Button("执行降维并绘图")
            dr_df = gr.Dataframe(label="降维结果表")
            dr_plot = gr.Plot(label="pl.plot_2d")

        with gr.Tab("7) 模型训练"):
            model_type = gr.Dropdown(choices=["classic", "dl"], value="classic", label="训练类型")
            with gr.Group(visible=True) as grp_model_classic:
                m_cols = gr.Textbox(label="feature_cols（逗号分隔，空=默认）")
                m_cv = gr.Slider(2, 10, value=5, step=1, label="cv")
                m_scoring = gr.Dropdown(choices=["accuracy", "f1", "recall"], value="accuracy", label="scoring")
            with gr.Group(visible=False) as grp_model_dl:
                dl_name = gr.Textbox(value="1D-CNN", label="model_name")
                dl_cv = gr.Slider(2, 10, value=5, step=1, label="cv")
                dl_epoch = gr.Slider(1, 100, value=10, step=1, label="epoch")
                dl_bs = gr.Slider(8, 256, value=64, step=8, label="batch_size")
                dl_lr = gr.Number(value=1e-3, label="learning_rate")
                dl_device = gr.Dropdown(choices=["cpu", "cuda"], value="cpu", label="device")
            model_train_btn = gr.Button("训练")
            model_result = gr.JSON(label="训练结果")
            model_name_for_plot = gr.Textbox(label="model_name（绘图）", value="Random Forest")
            cm_split = gr.Dropdown(choices=["train", "test"], value="test", label="cm split")
            metric_name = gr.Dropdown(choices=["accuracy", "f1", "recall"], value="accuracy", label="metric")
            fold_loss_type = gr.Dropdown(choices=["train", "val"], value="train", label="fold loss type")
            model_plot_btn = gr.Button("模型可视化")
            model_cm_plot = gr.Plot(label="pl.model_cm")
            model_metric_plot = gr.Plot(label="pl.model_metric_bar")
            model_fold_plot = gr.Plot(label="pl.plot_fold_loss")
            model_table = gr.Dataframe(label='best_pkg["all_samples_feature_pred"]')

        with gr.Tab("8) 新样本预测"):
            pred_files = gr.File(label="上传未知样本", file_count="multiple")
            pred_model_name = gr.Textbox(label="使用模型名（可空）")
            pred_btn = gr.Button("classify_new_samples")
            pred_df = gr.Dataframe(label="预测 feature_df")

            pred_plot_kind = gr.Dropdown(choices=["plot_2d", "plot_3d", "event_current_label", "stacked_bar"], value="plot_2d", label="可视化方法")
            pred_sample_id = gr.Textbox(label="sample_id（event_current_label用）")
            pred_label_col = gr.Textbox(label="label_col（event_current_label）", value="pred_label")
            pred_plot_btn = gr.Button("绘图")
            pred_plot = gr.Plot(label="预测可视化")

        with gr.Tab("9) 导出与复现"):
            export_dir = gr.Textbox(label="输出目录", value="./ui_exports")
            export_btn = gr.Button("导出")
            export_result = gr.JSON(label="导出结果")

        def on_denoise_method(method):
            return (
                gr.update(visible=method == "butterworth_filtfilt"),
                gr.update(visible=method == "moving_average"),
                gr.update(visible=method == "median"),
                gr.update(visible=method == "drift_corrected_moving_average"),
            )

        def on_detect_method(method):
            return (
                gr.update(visible=method == "threshold"),
                gr.update(visible=method == "zscore_threshold"),
                gr.update(visible=method == "cusum"),
                gr.update(visible=method == "pelt"),
                gr.update(visible=method == "hmm"),
            )

        def on_filter_method(method):
            return (
                gr.update(visible=method == "blockade_gmm"),
                gr.update(visible=method == "isolation_forest"),
                gr.update(visible=method == "lof"),
            )

        def on_dr_method(method):
            return (
                gr.update(visible=method == "pca"),
                gr.update(visible=method == "tsne"),
                gr.update(visible=method == "umap"),
            )

        def on_model_type(method):
            return gr.update(visible=method == "classic"), gr.update(visible=method == "dl")

        def do_load(files, reader_name, group_text):
            sample_paths = _to_file_map(files)
            sample_to_group = _parse_mapping_text(group_text)
            out = ctl.load_samples(sample_paths=sample_paths, sample_to_group=sample_to_group, reader=reader_name)
            return out["summary"], out["sample_df"]

        def do_denoise(method, n, wn, w_ma, w_median, dwin, swin, sid, start_ms, end_ms):
            if method == "butterworth_filtfilt":
                out = ctl.run_denoise(method=method, filtfilt_N=int(n), filtfilt_Wn=float(wn))
            elif method == "moving_average":
                out = ctl.run_denoise(method=method, window=int(w_ma))
            elif method == "median":
                out = ctl.run_denoise(method=method, window=int(w_median))
            elif method == "drift_corrected_moving_average":
                out = ctl.run_denoise(method=method, drift_window=int(dwin), smooth_window=int(swin))
            else:
                out = ctl.run_denoise(method=method)
            fig = ctl.plot_current(sample_id=(sid or None), current="denoise", start_ms=float(start_ms), end_ms=float(end_ms))
            return out, fig

        def _detect_params(method, sigma_k, min_dur, nmethod, z, z_min, z_nm, c_d, c_t, c_min, c_nm, p_m, p_p, p_s, p_min, p_nm, h_n, h_cov_t, h_iter, h_min):
            if method == "threshold":
                return {"sigma_k": float(sigma_k), "min_duration_s": float(min_dur), "noise_method": nmethod}
            if method == "zscore_threshold":
                return {"z": float(z), "min_duration_s": float(z_min), "noise_method": z_nm}
            if method == "cusum":
                return {"drift": float(c_d), "threshold": float(c_t), "min_duration_s": float(c_min), "noise_method": c_nm}
            if method == "pelt":
                return {
                    "model": p_m,
                    "penalty": float(p_p),
                    "sigma_k": float(p_s),
                    "min_duration_s": float(p_min),
                    "noise_method": p_nm,
                }
            return {
                "n_components": int(h_n),
                "covariance_type": h_cov_t,
                "n_iter": int(h_iter),
                "min_duration_s": float(h_min),
            }

        def _none_if_nan(v):
            try:
                if v != v:
                    return None
            except Exception:
                return None
            return float(v)

        def _base_params(method, window, q):
            if method == "rolling_quantile":
                return {"window": int(window), "q": float(q)}
            if method == "global_quantile":
                return {"q": float(q)}
            return {}

        def _common_detect_kwargs(method, baseline_method_value, b_win, bq, direction, mrg, mrg_gap, exc, lo, hi, *param_values):
            detect_params = _detect_params(method, *param_values)
            return {
                "detect_method": method,
                "detect_params": detect_params,
                "baseline_method": baseline_method_value,
                "baseline_params": _base_params(baseline_method_value, b_win, bq),
                "detect_direction": direction,
                "merge_event": bool(mrg),
                "merge_event_params": {"merge_gap_ms": float(mrg_gap)},
                "exclude_current": bool(exc),
                "exclude_current_params": {"min": _none_if_nan(lo), "max": _none_if_nan(hi)},
            }

        def run_preview(method, baseline_method_value, b_win, bq, direction, mrg, mrg_gap, exc, lo, hi, sid, cur, s_ms, e_ms, se, ee, *param_values):
            kwargs = _common_detect_kwargs(method, baseline_method_value, b_win, bq, direction, mrg, mrg_gap, exc, lo, hi, *param_values)
            out = ctl.run_detect(stage="preview", sample_id=(sid or None), current=cur, start_ms=float(s_ms), end_ms=float(e_ms), **kwargs)
            fig = ctl.plot_event_current_simple(sample_id=(sid or None), current=cur, start_event=int(se), end_event=int(ee))
            table = ctl.simple_events_df(sample_id=(sid or None))
            return out, fig, table

        def run_global(method, baseline_method_value, b_win, bq, direction, mrg, mrg_gap, exc, lo, hi, sid, cur, se, ee, table_sid, *param_values):
            kwargs = _common_detect_kwargs(method, baseline_method_value, b_win, bq, direction, mrg, mrg_gap, exc, lo, hi, *param_values)
            out = ctl.run_detect(stage="global", **kwargs)
            fig = ctl.plot_event_current(sample_id=(sid or None), current=cur, start_event=int(se), end_event=int(ee))
            table = ctl.events_df(sample_id=(table_sid or sid or None))
            return out, fig, table

        def do_extract_features():
            df = ctl.extract_features()
            return df

        def do_feature_plot(kind, x, y, z, value, gcol, vcol):
            if kind == "plot_2d":
                return ctl.plot_2d(x=x, y=y, data="full", value=value)
            if kind == "plot_3d":
                return ctl.plot_3d(x=x, y=y, z=z, data="full", value=value)
            return ctl.box_significance(group_col=gcol, value_col=vcol, data="full")

        def do_filter(method, n_comp, prior_mean, if_contam, lof_contam):
            if method == "blockade_gmm":
                params = {"n_components": int(n_comp), "prior_mean": _none_if_nan(prior_mean)}
            elif method == "isolation_forest":
                params = {"contamination": float(if_contam)}
            else:
                params = {"contamination": float(lof_contam)}
            return ctl.filter_events(method=method, parameters=params)

        def do_filtered_plot(kind):
            if kind == "plot_2d":
                return ctl.plot_2d(data="filtered", value="quality_tag")
            if kind == "plot_3d":
                return ctl.plot_3d(data="filtered", value="quality_tag")
            return ctl.box_significance(group_col="quality_tag", value_col="blockade_ratio", data="filtered")

        def do_dr(method, cols_text, data_name, random_state, perplexity, n_iter, nn, min_dist):
            cols = [c.strip() for c in (cols_text or "").split(",") if c.strip()] or None
            kwargs = {}
            if method == "pca":
                kwargs = {"random_state": int(random_state)}
            elif method == "tsne":
                kwargs = {"random_state": int(random_state), "perplexity": float(perplexity), "n_iter": int(n_iter)}
            elif method == "umap":
                kwargs = {"random_state": int(random_state), "n_neighbors": int(nn), "min_dist": float(min_dist)}
            df = ctl.do_dimensionality_reduction(method=method, feature_cols=cols, data=data_name, **kwargs)
            if method == "pca":
                fig = ctl.plot_2d(x="PC1", y="PC2", data=data_name, value="label")
            elif method == "tsne":
                fig = ctl.plot_2d(x="TSNE1", y="TSNE2", data=data_name, value="label", y_log2=False)
            else:
                fig = ctl.plot_2d(x="UMAP1", y="UMAP2", data=data_name, value="label", y_log2=False)
            return df, fig

        def do_train(model_kind, cols, cv, scoring, dl_model_name, dl_cv_v, dl_epoch_v, dl_bs_v, dl_lr_v, dl_device_v):
            if model_kind == "classic":
                fcols = [c.strip() for c in (cols or "").split(",") if c.strip()] or None
                out = ctl.train_model(feature_cols=fcols, cv=int(cv), scoring=scoring)
            else:
                out = ctl.train_dl_model(
                    model_name=dl_model_name,
                    cv=int(dl_cv_v),
                    epoch=int(dl_epoch_v),
                    batch_size=int(dl_bs_v),
                    learning_rate=float(dl_lr_v),
                    device=dl_device_v,
                )
            return out, ctl.model_prediction_table()

        def do_model_plots(model_name, split, metric, loss_type):
            cm = ctl.plot_model_cm(model_name=model_name, split=split)
            bar = ctl.plot_model_metric_bar(metric=metric, split=split)
            try:
                fold = ctl.plot_fold_loss(model_name=model_name, type=loss_type)
            except Exception:
                fold = None
            return cm, bar, fold

        def do_predict(files, model_name):
            df = ctl.predict_new(new_sample_paths=_to_file_map(files), model=(model_name or None))
            return df

        def do_predict_plot(kind, sid, label_col):
            if kind == "plot_2d":
                return ctl.plot_2d(data="full", value=label_col)
            if kind == "plot_3d":
                return ctl.plot_3d(data="full", value=label_col)
            if kind == "event_current_label":
                return ctl.plot_event_current_label(sample_id=(sid or None), label_col=label_col)
            return ctl.plot_stacked_bar(value_col=label_col, data="full")

        def do_export(path_text):
            out_dir = Path(path_text)
            tables = ctl.export_tables(out_dir)
            params_json = ctl.export_params_json(out_dir / "params_snapshot.json")
            script = ctl.export_analysis_script(out_dir / "reproduce_analysis.py")
            return {"tables": tables, "params_json": params_json, "script": script}

        denoise_method.change(on_denoise_method, inputs=[denoise_method], outputs=[grp_bw, grp_ma, grp_median, grp_drift])
        detect_method.change(on_detect_method, inputs=[detect_method], outputs=[grp_det_threshold, grp_det_z, grp_det_cusum, grp_det_pelt, grp_det_hmm])
        filter_method.change(on_filter_method, inputs=[filter_method], outputs=[grp_filter_gmm, grp_filter_if, grp_filter_lof])
        dr_method.change(on_dr_method, inputs=[dr_method], outputs=[grp_dr_pca, grp_dr_tsne, grp_dr_umap])
        model_type.change(on_model_type, inputs=[model_type], outputs=[grp_model_classic, grp_model_dl])

        load_btn.click(do_load, inputs=[file_input, reader, sample_to_group_txt], outputs=[load_summary, sample_df])
        denoise_run_btn.click(
            do_denoise,
            inputs=[denoise_method, bw_n, bw_wn, ma_window, median_window, drift_window, smooth_window, denoise_sample, denoise_start, denoise_end],
            outputs=[denoise_result, denoise_plot],
        )

        detect_param_inputs = [
            d_sigma_k,
            d_min_duration,
            d_noise_method,
            z_thr,
            z_min_duration,
            z_noise_method,
            c_drift,
            c_threshold,
            c_min_duration,
            c_noise_method,
            p_model,
            p_penalty,
            p_sigma_k,
            p_min_duration,
            p_noise_method,
            h_n_components,
            h_cov,
            h_n_iter,
            h_min_duration,
        ]

        preview_btn.click(
            run_preview,
            inputs=[
                detect_method,
                baseline_method,
                b_window,
                b_q,
                detect_direction,
                merge_event,
                merge_gap_ms,
                exclude_current,
                ex_min,
                ex_max,
                detect_sample,
                detect_current,
                detect_start,
                detect_end,
                detect_start_event,
                detect_end_event,
                *detect_param_inputs,
            ],
            outputs=[preview_result, preview_plot, preview_table],
        )
        global_btn.click(
            run_global,
            inputs=[
                detect_method,
                baseline_method,
                b_window,
                b_q,
                detect_direction,
                merge_event,
                merge_gap_ms,
                exclude_current,
                ex_min,
                ex_max,
                detect_sample,
                detect_current,
                detect_start_event,
                detect_end_event,
                global_sample,
                *detect_param_inputs,
            ],
            outputs=[global_result, global_plot, global_table],
        )

        extract_btn.click(do_extract_features, outputs=[feat_df])
        feat_plot_btn.click(do_feature_plot, inputs=[feat_plot_kind, feat_x, feat_y, feat_z, feat_value, feat_group_col, feat_value_col], outputs=[feat_plot])
        filter_btn.click(do_filter, inputs=[filter_method, f_n_components, f_prior_mean, f_if_contam, f_lof_contam], outputs=[filtered_df])
        filtered_plot_btn.click(do_filtered_plot, inputs=[filtered_plot_kind], outputs=[filtered_plot])
        dr_btn.click(
            do_dr,
            inputs=[dr_method, dr_cols, dr_data, dr_random_state, tsne_perplexity, tsne_iter, umap_neighbors, umap_min_dist],
            outputs=[dr_df, dr_plot],
        )
        model_train_btn.click(
            do_train,
            inputs=[model_type, m_cols, m_cv, m_scoring, dl_name, dl_cv, dl_epoch, dl_bs, dl_lr, dl_device],
            outputs=[model_result, model_table],
        )
        model_plot_btn.click(do_model_plots, inputs=[model_name_for_plot, cm_split, metric_name, fold_loss_type], outputs=[model_cm_plot, model_metric_plot, model_fold_plot])
        pred_btn.click(do_predict, inputs=[pred_files, pred_model_name], outputs=[pred_df])
        pred_plot_btn.click(do_predict_plot, inputs=[pred_plot_kind, pred_sample_id, pred_label_col], outputs=[pred_plot])
        export_btn.click(do_export, inputs=[export_dir], outputs=[export_result])

    return demo


def main() -> None:
    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()
