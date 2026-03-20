"""Menu construction mixin for InspectorApp."""

from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING

from shared.i18n import available_languages
from dl_anomaly.gui.platform_keys import accel, accel_shift

if TYPE_CHECKING:
    from dl_anomaly.gui.inspector_app import InspectorApp


class MenuMixin:
    """Builds the full menu bar for InspectorApp."""

    def _build_menu(self: "InspectorApp") -> None:
        menubar = tk.Menu(self, bg="#2b2b2b", fg="#e0e0e0", activebackground="#3a3a5c", activeforeground="#ffffff")

        # -- File --
        file_menu = tk.Menu(menubar, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                            activebackground="#3a3a5c", activeforeground="#ffffff")
        file_menu.add_command(label="\u958b\u555f\u5716\u7247...", command=self._cmd_open_image, accelerator=accel("O"))
        file_menu.add_command(label="\u958b\u555f\u8cc7\u6599\u593e...", command=self._cmd_open_dir)
        file_menu.add_separator()
        file_menu.add_command(label="\u5132\u5b58\u7576\u524d\u5716\u7247...", command=self._cmd_save_image, accelerator=accel("S"))
        file_menu.add_command(label="\u5132\u5b58\u6240\u6709\u6b65\u9a5f...", command=self._cmd_save_all)
        file_menu.add_separator()
        file_menu.add_command(label="\u5132\u5b58\u6d41\u7a0b...", command=self._cmd_save_recipe)
        file_menu.add_command(label="\u8f09\u5165\u4e26\u5957\u7528\u6d41\u7a0b...", command=self._cmd_load_and_apply_recipe)
        file_menu.add_command(label="\u6279\u6b21\u5957\u7528\u6d41\u7a0b...", command=self._cmd_batch_apply_recipe)
        file_menu.add_separator()
        file_menu.add_command(label="\u74b0\u5883\u8a2d\u5b9a...", command=self._cmd_settings)
        file_menu.add_separator()
        self._recent_menu = tk.Menu(file_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                                    activebackground="#3a3a5c", activeforeground="#ffffff")
        file_menu.add_cascade(label="\u6700\u8fd1\u958b\u555f", menu=self._recent_menu)
        file_menu.add_separator()
        lang_menu = tk.Menu(file_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                            activebackground="#3a3a5c", activeforeground="#ffffff")
        _LANG_LABELS = {"en": "English", "zh-TW": "\u7e41\u9ad4\u4e2d\u6587", "zh-CN": "\u7b80\u4f53\u4e2d\u6587"}
        for lang_code in available_languages():
            label = _LANG_LABELS.get(lang_code, lang_code)
            lang_menu.add_radiobutton(
                label=label,
                variable=self._lang_var,
                value=lang_code,
                command=lambda lc=lang_code: self._cmd_switch_language(lc),
            )
        file_menu.add_cascade(label="\u8a9e\u8a00 / Language", menu=lang_menu)
        file_menu.add_separator()
        file_menu.add_command(label="\u7d50\u675f", command=self._on_close)
        menubar.add_cascade(label="\u6a94\u6848", menu=file_menu)

        # -- Operations --
        ops_menu = tk.Menu(menubar, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                           activebackground="#3a3a5c", activeforeground="#ffffff")
        ops_menu.add_command(label="\u7070\u968e\u8f49\u63db", command=lambda: self._ops_panel._op_grayscale())
        ops_menu.add_command(label="\u9ad8\u65af\u6a21\u7cca", command=lambda: self._ops_panel._op_blur())
        ops_menu.add_command(label="\u908a\u7de3\u5075\u6e2c", command=lambda: self._ops_panel._op_edge())
        ops_menu.add_command(label="\u76f4\u65b9\u5716\u5747\u8861", command=lambda: self._ops_panel._op_histeq())
        ops_menu.add_command(label="\u53cd\u8272", command=lambda: self._ops_panel._op_invert())
        ops_menu.add_separator()
        ops_menu.add_command(label="\u57f7\u884c\u81ea\u52d5\u7de8\u78bc\u5668", command=self._cmd_run_autoencoder)
        ops_menu.add_command(label="\u8a08\u7b97\u8aa4\u5dee\u5716", command=self._cmd_compute_error_map)
        ops_menu.add_command(label="\u5957\u7528 SSIM", command=self._cmd_apply_ssim)
        ops_menu.add_command(label="\u5957\u7528\u95be\u503c\u906e\u7f69", command=self._cmd_apply_threshold_mask)
        ops_menu.add_separator()
        ops_menu.add_command(label="VM \u55ae\u5f35\u6aa2\u6e2c", command=self._cmd_vm_inspect_single)
        ops_menu.add_command(label="VM \u6279\u6b21\u6aa2\u6e2c...", command=self._cmd_vm_inspect_batch)
        menubar.add_cascade(label="\u64cd\u4f5c", menu=ops_menu)

        # -- Region --
        region_menu = tk.Menu(menubar, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                              activebackground="#3a3a5c", activeforeground="#ffffff")
        region_menu.add_command(label="\u50cf\u7d20\u503c\u6aa2\u67e5\u5668...", command=self._toggle_pixel_inspector, accelerator=accel("I"))
        region_menu.add_command(
            label="\u50cf\u7d20\u6aa2\u67e5\u5de5\u5177",
            command=lambda: self._cmd_tool_pixel_inspect(
                not self._toolbar.get_toggle_state("tool_pixel_inspect")),
            accelerator=accel_shift("I"),
        )
        region_menu.add_command(
            label="\u5340\u57df\u9078\u53d6\u5de5\u5177",
            command=lambda: self._cmd_tool_region_select(
                not self._toolbar.get_toggle_state("tool_region_select")),
            accelerator=accel_shift("R"),
        )
        region_menu.add_separator()
        region_menu.add_command(label="\u4e8c\u503c\u5316...", command=self._open_binarize_dialog)
        region_menu.add_separator()
        region_menu.add_command(label="\u95be\u503c\u5206\u5272...", command=self._open_threshold_dialog, accelerator=accel("T"))
        region_menu.add_command(label="\u81ea\u52d5\u95be\u503c (Otsu)", command=self._auto_threshold_otsu)
        region_menu.add_command(label="\u81ea\u9069\u61c9\u95be\u503c...", command=self._open_adaptive_threshold_dialog)
        region_menu.add_command(label="\u53ef\u8b8a\u95be\u503c...", command=self._dlg_var_threshold)
        region_menu.add_command(label="\u5c40\u90e8\u95be\u503c...", command=self._dlg_local_threshold)
        region_menu.add_separator()
        region_menu.add_command(label="\u6253\u6563 (Connection)", command=self._region_connection)
        region_menu.add_command(label="\u586b\u5145 (Fill Up)...", command=self._region_fill_up)

        shape_trans_menu = tk.Menu(region_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                                   activebackground="#3a3a5c", activeforeground="#ffffff")
        for st in ["convex", "rectangle", "circle", "ellipse"]:
            shape_trans_menu.add_command(label=st, command=lambda s=st: self._region_shape_trans(s))
        region_menu.add_cascade(label="\u5f62\u72c0\u8b8a\u63db", menu=shape_trans_menu)

        region_menu.add_separator()
        region_menu.add_command(label="\u5340\u57df\u4fb5\u8755...", command=lambda: self._region_morphology("erosion"))
        region_menu.add_command(label="\u5340\u57df\u81a8\u8139...", command=lambda: self._region_morphology("dilation"))
        region_menu.add_command(label="\u5340\u57df\u958b\u904b\u7b97...", command=lambda: self._region_morphology("opening"))
        region_menu.add_command(label="\u5340\u57df\u9589\u904b\u7b97...", command=lambda: self._region_morphology("closing"))
        region_menu.add_separator()
        region_menu.add_command(label="\u7be9\u9078\u5340\u57df...", command=self._open_region_filter)
        region_menu.add_command(label="\u4f9d\u7070\u5ea6\u7be9\u9078...", command=self._region_select_gray)
        region_menu.add_command(label="\u6392\u5e8f\u5340\u57df...", command=self._region_sort)
        region_menu.add_separator()
        region_menu.add_command(label="\u5340\u57df\u806f\u96c6", command=lambda: self._region_set_op("union"))
        region_menu.add_command(label="\u5340\u57df\u4ea4\u96c6", command=lambda: self._region_set_op("intersection"))
        region_menu.add_command(label="\u5340\u57df\u5dee\u96c6", command=lambda: self._region_set_op("difference"))
        region_menu.add_command(label="\u5340\u57df\u88dc\u96c6", command=lambda: self._region_set_op("complement"))
        region_menu.add_separator()
        region_menu.add_command(label="\u7e2e\u6e1b\u57df (Reduce Domain)", command=self._cmd_reduce_domain)
        region_menu.add_command(label="\u88c1\u5207\u57df (Crop Domain)", command=self._cmd_crop_domain)
        region_menu.add_separator()
        region_menu.add_command(label="Blob \u5206\u6790...", command=self._open_blob_analysis)
        region_menu.add_command(label="\u8f2a\u5ed3\u6aa2\u6e2c...", command=self._open_contour_detection_dialog)
        menubar.add_cascade(label="\u5340\u57df", menu=region_menu)

        # -- 影像處理 --
        vision_menu = tk.Menu(menubar, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                              activebackground="#3a3a5c", activeforeground="#ffffff")

        filter_menu = tk.Menu(vision_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                              activebackground="#3a3a5c", activeforeground="#ffffff")
        filter_menu.add_command(label="\u5747\u503c\u6ffe\u6ce2...", command=self._dlg_mean_image)
        filter_menu.add_command(label="\u4e2d\u503c\u6ffe\u6ce2...", command=self._dlg_median_image)
        filter_menu.add_command(label="\u9ad8\u65af\u6a21\u7cca...", command=self._dlg_gauss_blur)
        filter_menu.add_command(label="\u96d9\u908a\u6ffe\u6ce2...", command=self._dlg_bilateral_filter)
        filter_menu.add_command(label="\u92b3\u5316...", command=self._dlg_sharpen)
        filter_menu.add_command(label="\u5f37\u8abf...", command=self._dlg_emphasize)
        filter_menu.add_command(label="Laplacian", command=lambda: self._apply_vision_op("laplace_filter"))
        vision_menu.add_cascade(label="\u6ffe\u6ce2", menu=filter_menu)

        edge_menu2 = tk.Menu(vision_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                             activebackground="#3a3a5c", activeforeground="#ffffff")
        edge_menu2.add_command(label="Canny \u908a\u7de3...", command=self._dlg_canny)
        edge_menu2.add_command(label="Sobel", command=lambda: self._apply_vision_op("sobel_filter"))
        edge_menu2.add_command(label="Prewitt", command=lambda: self._apply_vision_op("prewitt_filter"))
        edge_menu2.add_command(label="\u96f6\u4ea4\u53c9", command=lambda: self._apply_vision_op("zero_crossing"))
        edge_menu2.add_command(label="\u9ad8\u65af\u5c0e\u6578...", command=self._dlg_derivative_gauss)
        vision_menu.add_cascade(label="\u908a\u7de3", menu=edge_menu2)

        morph_menu2 = tk.Menu(vision_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                              activebackground="#3a3a5c", activeforeground="#ffffff")
        morph_menu2.add_command(label="\u7070\u5ea6\u4fb5\u8755...", command=self._dlg_gray_erosion)
        morph_menu2.add_command(label="\u7070\u5ea6\u81a8\u8139...", command=self._dlg_gray_dilation)
        morph_menu2.add_command(label="\u7070\u5ea6\u958b\u904b\u7b97...", command=self._dlg_gray_opening)
        morph_menu2.add_command(label="\u7070\u5ea6\u9589\u904b\u7b97...", command=self._dlg_gray_closing)
        morph_menu2.add_command(label="Top-hat...", command=self._dlg_top_hat)
        morph_menu2.add_command(label="Bottom-hat...", command=self._dlg_bottom_hat)
        morph_menu2.add_separator()
        morph_menu2.add_command(label="\u52d5\u614b\u95be\u503c\u5206\u5272...", command=self._open_dyn_threshold_dialog)
        vision_menu.add_cascade(label="\u5f62\u614b\u5b78", menu=morph_menu2)

        geom_menu = tk.Menu(vision_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                            activebackground="#3a3a5c", activeforeground="#ffffff")
        for label, op in [("\u65cb\u8f49 90\u00b0", "rotate_90"), ("\u65cb\u8f49 180\u00b0", "rotate_180"),
                          ("\u65cb\u8f49 270\u00b0", "rotate_270"),
                          ("\u6c34\u5e73\u93e1\u50cf", "mirror_h"), ("\u5782\u76f4\u93e1\u50cf", "mirror_v"),
                          ("\u7e2e\u653e 50%", "zoom_50"), ("\u7e2e\u653e 200%", "zoom_200")]:
            geom_menu.add_command(label=label, command=lambda o=op: self._apply_vision_op(o))
        vision_menu.add_cascade(label="\u5e7e\u4f55", menu=geom_menu)

        color_menu = tk.Menu(vision_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                             activebackground="#3a3a5c", activeforeground="#ffffff")
        for label, op in [("\u8f49\u7070\u968e", "rgb_to_gray"), ("\u8f49 HSV", "rgb_to_hsv"),
                          ("\u8f49 HLS", "rgb_to_hls"),
                          ("\u76f4\u65b9\u5716\u5747\u8861", "histogram_eq"),
                          ("\u53cd\u8272", "invert_image"), ("\u5149\u7167\u6821\u6b63", "illuminate")]:
            color_menu.add_command(label=label, command=lambda o=op: self._apply_vision_op(o))
        color_menu.add_separator()
        color_menu.add_command(label="CLAHE...", command=self._dlg_clahe)
        vision_menu.add_cascade(label="\u8272\u5f69", menu=color_menu)

        gray_trans_menu = tk.Menu(vision_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                                  activebackground="#3a3a5c", activeforeground="#ffffff")
        gray_trans_menu.add_command(label="\u4eae\u5ea6/\u5c0d\u6bd4\u5ea6\u8abf\u6574...", command=self._dlg_scale_image)
        gray_trans_menu.add_command(label="\u7d55\u5c0d\u503c", command=lambda: self._apply_vision_op("abs_image"))
        gray_trans_menu.add_command(label="\u53cd\u8272", command=lambda: self._apply_vision_op("invert_image"))
        gray_trans_menu.add_command(label="\u5c0d\u6578\u8b8a\u63db...", command=self._dlg_log_image)
        gray_trans_menu.add_command(label="\u6307\u6578\u8b8a\u63db...", command=self._dlg_exp_image)
        gray_trans_menu.add_command(label="Gamma \u6821\u6b63...", command=self._dlg_gamma_image)
        vision_menu.add_cascade(label="\u7070\u5ea6\u8b8a\u63db", menu=gray_trans_menu)

        img_op_menu = tk.Menu(vision_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                              activebackground="#3a3a5c", activeforeground="#ffffff")
        img_op_menu.add_command(label="\u5716\u50cf\u76f8\u6e1b...", command=self._open_subtract_dialog)
        img_op_menu.add_command(label="\u7d55\u5c0d\u5dee\u5206", command=lambda: self._apply_vision_op("abs_diff_image"))
        vision_menu.add_cascade(label="\u5716\u50cf\u904b\u7b97", menu=img_op_menu)

        freq_menu = tk.Menu(vision_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                            activebackground="#3a3a5c", activeforeground="#ffffff")
        freq_menu.add_command(label="FFT \u983b\u8b5c...", command=self._dlg_fft)
        freq_menu.add_command(label="\u4f4e\u901a\u6ffe\u6ce2...", command=lambda: self._dlg_freq_filter("lowpass"))
        freq_menu.add_command(label="\u9ad8\u901a\u6ffe\u6ce2...", command=lambda: self._dlg_freq_filter("highpass"))
        vision_menu.add_cascade(label="\u983b\u57df\u8655\u7406", menu=freq_menu)

        texture_menu = tk.Menu(vision_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                               activebackground="#3a3a5c", activeforeground="#ffffff")
        for label, op in [("\u71b5\u5f71\u50cf", "entropy_image"), ("\u6a19\u6e96\u5dee\u5f71\u50cf", "deviation_image"),
                          ("\u5c40\u90e8\u6700\u5c0f", "local_min"), ("\u5c40\u90e8\u6700\u5927", "local_max")]:
            texture_menu.add_command(label=label, command=lambda o=op: self._apply_vision_op(o))
        vision_menu.add_cascade(label="\u7d0b\u7406", menu=texture_menu)

        barcode_menu = tk.Menu(vision_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                               activebackground="#3a3a5c", activeforeground="#ffffff")
        for label, op in [("\u689d\u78bc\u5075\u6e2c", "find_barcode"), ("QR Code", "find_qrcode"),
                          ("DataMatrix", "find_datamatrix")]:
            barcode_menu.add_command(label=label, command=lambda o=op: self._apply_vision_op(o))
        vision_menu.add_cascade(label="\u689d\u78bc", menu=barcode_menu)

        # 分割
        seg_menu = tk.Menu(vision_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                           activebackground="#3a3a5c", activeforeground="#ffffff")
        seg_menu.add_command(label="\u5206\u6c34\u5dba...", command=self._dlg_watersheds)
        seg_menu.add_command(label="\u8ddd\u96e2\u8b8a\u63db...", command=self._dlg_distance_transform)
        seg_menu.add_command(label="\u9aa8\u67b6\u5316", command=lambda: self._apply_vision_op("skeleton"))
        vision_menu.add_cascade(label="\u5206\u5272", menu=seg_menu)

        # 特徵點
        feat_menu = tk.Menu(vision_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                            activebackground="#3a3a5c", activeforeground="#ffffff")
        feat_menu.add_command(label="Harris \u89d2\u9ede...", command=self._dlg_points_harris)
        feat_menu.add_command(label="Shi-Tomasi \u7279\u5fb5\u9ede...", command=self._dlg_points_shi_tomasi)
        vision_menu.add_cascade(label="\u7279\u5fb5\u9ede", menu=feat_menu)

        # 直線/圓偵測
        hough_menu = tk.Menu(vision_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                             activebackground="#3a3a5c", activeforeground="#ffffff")
        hough_menu.add_command(label="Hough \u76f4\u7dda...", command=self._dlg_hough_lines)
        hough_menu.add_command(label="Hough \u5713...", command=self._dlg_hough_circles)
        vision_menu.add_cascade(label="\u76f4\u7dda/\u5713\u5075\u6e2c", menu=hough_menu)

        # 相機
        camera_menu = tk.Menu(vision_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                              activebackground="#3a3a5c", activeforeground="#ffffff")
        camera_menu.add_command(label="\u64f7\u53d6\u5f71\u50cf", command=lambda: self._apply_vision_op("grab_image"))
        vision_menu.add_cascade(label="\u76f8\u6a5f", menu=camera_menu)

        vision_menu.add_separator()
        vision_menu.add_command(label="\u8173\u672c\u7de8\u8f2f\u5668", command=self._toggle_script_editor, accelerator="F8")
        menubar.add_cascade(label="\u5f71\u50cf\u8655\u7406", menu=vision_menu)

        # -- Model --
        model_menu = tk.Menu(menubar, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                             activebackground="#3a3a5c", activeforeground="#ffffff")

        # DL Model submenu
        dl_model_menu = tk.Menu(model_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                                activebackground="#3a3a5c", activeforeground="#ffffff")
        dl_model_menu.add_command(label="\u8a13\u7df4\u65b0\u6a21\u578b...", command=self._cmd_train, accelerator="F6")
        dl_model_menu.add_command(label="\u8f09\u5165 Checkpoint...", command=self._cmd_load_model)
        dl_model_menu.add_command(label="\u5132\u5b58 Checkpoint...", command=self._cmd_save_checkpoint)
        dl_model_menu.add_separator()
        dl_model_menu.add_command(label="\u6a21\u578b\u8cc7\u8a0a...", command=self._cmd_model_info)
        dl_model_menu.add_command(label="\u8a08\u7b97\u95be\u503c", command=self._cmd_compute_threshold)
        model_menu.add_cascade(label="DL \u6a21\u578b (Autoencoder)", menu=dl_model_menu)

        model_menu.add_separator()

        # Variation Model submenu
        vm_model_menu = tk.Menu(model_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                                activebackground="#3a3a5c", activeforeground="#ffffff")
        vm_model_menu.add_command(label="\u8a13\u7df4\u65b0\u6a21\u578b...", command=self._cmd_vm_train)
        vm_model_menu.add_command(label="\u8f09\u5165\u6a21\u578b (.npz)...", command=self._cmd_vm_load_model)
        vm_model_menu.add_command(label="\u5132\u5b58\u6a21\u578b...", command=self._cmd_vm_save_model)
        vm_model_menu.add_separator()
        vm_model_menu.add_command(label="\u6a21\u578b\u8cc7\u8a0a...", command=self._cmd_vm_model_info)
        vm_model_menu.add_command(label="\u91cd\u65b0\u8a08\u7b97\u95be\u503c", command=self._cmd_vm_reprepare_thresholds)
        vm_model_menu.add_separator()
        vm_model_menu.add_command(label="\u95be\u503c\u8996\u89ba\u5316...", command=self._cmd_vm_show_threshold_viz)
        model_menu.add_cascade(label="Variation Model (\u7d71\u8a08)", menu=vm_model_menu)

        model_menu.add_separator()

        # Pipeline Model submenu
        pipeline_model_menu = tk.Menu(model_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                                       activebackground="#3a3a5c", activeforeground="#ffffff")
        pipeline_model_menu.add_command(label="\u5132\u5b58\u7ba1\u7dda\u6a21\u578b...", command=self._cmd_save_pipeline_model)
        pipeline_model_menu.add_command(label="\u8f09\u5165\u7ba1\u7dda\u6a21\u578b...", command=self._cmd_load_pipeline_model)
        pipeline_model_menu.add_separator()
        pipeline_model_menu.add_command(label="\u7ba1\u7dda\u6a21\u578b\u7ba1\u7406...", command=self._open_pipeline_model_manager)
        model_menu.add_cascade(label="\u7ba1\u7dda\u6a21\u578b (Pipeline Model)", menu=pipeline_model_menu)
        menubar.add_cascade(label="\u6a21\u578b", menu=model_menu)

        # -- View --
        view_menu = tk.Menu(menubar, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                            activebackground="#3a3a5c", activeforeground="#ffffff")
        view_menu.add_command(label="\u7e2e\u653e\u81f3\u7a97\u53e3", command=self._cmd_fit, accelerator="Space")
        view_menu.add_command(label="\u653e\u5927", command=self._cmd_zoom_in, accelerator="+")
        view_menu.add_command(label="\u7e2e\u5c0f", command=self._cmd_zoom_out, accelerator="-")
        view_menu.add_command(label="1:1 \u539f\u59cb\u5927\u5c0f", command=self._cmd_actual_size)
        view_menu.add_separator()
        view_menu.add_command(label="\u76f4\u65b9\u5716...", command=self._cmd_histogram)
        view_menu.add_command(label="\u640d\u5931\u66f2\u7dda", command=self._cmd_toggle_loss_curve)
        view_menu.add_command(label="\u91cd\u5efa\u5c0d\u6bd4...", command=self._cmd_reconstruction_compare)
        view_menu.add_command(label="\u5716\u7247\u6bd4\u5c0d...", command=self._cmd_compare_steps)
        view_menu.add_command(label="\u6279\u6b21\u5716\u7247\u6bd4\u5c0d...", command=self._cmd_batch_compare_steps)
        menubar.add_cascade(label="\u6aa2\u8996", menu=view_menu)

        # -- Tools (工具) --
        tools_menu = tk.Menu(menubar, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                             activebackground="#3a3a5c", activeforeground="#ffffff")
        tools_menu.add_command(label="\u5f62\u72c0\u5339\u914d...", command=self._open_shape_matching, accelerator=accel("M"))
        tools_menu.add_command(label="\u91cf\u6e2c\u5de5\u5177...", command=self._open_metrology, accelerator=accel_shift("M"))
        tools_menu.add_separator()
        tools_menu.add_command(label="ROI \u7ba1\u7406...", command=self._open_roi_manager, accelerator=accel("R"))
        tools_menu.add_separator()
        tools_menu.add_command(label="PatchCore / ONNX \u6a21\u578b...", command=self._open_advanced_models, accelerator=accel_shift("P"))
        tools_menu.add_separator()
        tools_menu.add_command(label="\u6aa2\u6e2c\u5de5\u5177 (FFT/\u8272\u5f69/OCR/\u689d\u78bc)...", command=self._open_inspection_tools, accelerator=accel_shift("T"))
        tools_menu.add_separator()
        tools_menu.add_command(label="\u5de5\u7a0b\u5de5\u5177 (\u6a19\u5b9a/\u7ba1\u7dda/SPC/\u62fc\u63a5)...", command=self._open_engineering_tools, accelerator=accel_shift("E"))
        tools_menu.add_separator()
        tools_menu.add_command(label="MVP \u5de5\u5177 (\u76f8\u6a5f/\u6d41\u7a0b/\u5831\u8868)...", command=self._open_mvp_tools, accelerator=accel_shift("V"))
        tools_menu.add_separator()
        tools_menu.add_command(label="\u81ea\u52d5\u95be\u503c\u6821\u6e96...", command=self._open_auto_tune, accelerator=accel_shift("A"))
        tools_menu.add_separator()
        tools_menu.add_command(label="SPC \u8b66\u5831\u8a2d\u5b9a...", command=self._open_spc_settings)
        menubar.add_cascade(label="\u5de5\u5177", menu=tools_menu)

        # -- Help --
        help_menu = tk.Menu(menubar, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                            activebackground="#3a3a5c", activeforeground="#ffffff")
        help_menu.add_command(label="\u5feb\u6377\u9375...", command=self._cmd_shortcuts)
        help_menu.add_command(label="\u95dc\u65bc...", command=self._cmd_about)
        menubar.add_cascade(label="\u5e6b\u52a9", menu=help_menu)

        self.configure(menu=menubar)
