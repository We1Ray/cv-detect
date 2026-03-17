# CV Defect Detection System

基於電腦視覺的工業瑕疵檢測系統，提供兩種獨立的檢測方法，搭配 HALCON HDevelop 風格的圖形化操作介面。涵蓋從影像擷取、前處理、瑕疵偵測到統計分析的完整檢測流程。

## 瑕疵檢測演算法展示

> 以 PCB（印刷電路板）瑕疵影像即時展示各檢測管線的處理過程與結果。

### 完整檢測管線總覽

從影像輸入、前處理、邊緣偵測、分割到異常圖生成的完整流程：

<p align="center">
  <img src="assets/demo/09_pipeline_overview.gif" alt="Pipeline Overview" width="600">
</p>

---

### 傳統電腦視覺方法

<table>
<tr>
<td align="center" width="50%">

**邊緣檢測 (Edge Detection)**

Canny / Sobel 梯度邊緣提取

<img src="assets/demo/01_edge_detection.gif" alt="Edge Detection" width="400">

</td>
<td align="center" width="50%">

**連通區域分析 (Blob Analysis)**

Otsu 閾值分割 + 連通元件標記

<img src="assets/demo/02_blob_analysis.gif" alt="Blob Analysis" width="400">

</td>
</tr>
<tr>
<td align="center">

**形態學運算 (Morphology)**

侵蝕 / 膨脹 / 開運算 / 閉運算

<img src="assets/demo/03_morphology.gif" alt="Morphology" width="400">

</td>
<td align="center">

**FFT 頻域分析 (Frequency Domain)**

頻譜視覺化 + 高斯高/低通濾波

<img src="assets/demo/04_fft_frequency.gif" alt="FFT Frequency" width="400">

</td>
</tr>
<tr>
<td align="center">

**色彩檢測 (Color Inspection)**

CIE Lab Delta-E 色差圖 + K-means 調色板

<img src="assets/demo/05_color_inspection.gif" alt="Color Inspection" width="400">

</td>
<td align="center">

**形狀匹配 (Shape Matching)**

梯度方向餘弦相似度 + 金字塔搜尋

<img src="assets/demo/06_shape_matching.gif" alt="Shape Matching" width="400">

</td>
</tr>
</table>

### 深度學習方法

<table>
<tr>
<td align="center" width="50%">

**自編碼器異常檢測 (Autoencoder)**

卷積自編碼器重建誤差 + MSE/SSIM 混合評分

<img src="assets/demo/07_autoencoder.gif" alt="Autoencoder" width="400">

</td>
<td align="center" width="50%">

**PatchCore 記憶庫檢測**

預訓練 CNN 特徵 + ball-tree kNN 異常評分

<img src="assets/demo/08_patchcore.gif" alt="PatchCore" width="400">

</td>
</tr>
</table>

<details>
<summary><b>重新生成 Demo GIF</b></summary>

```bash
conda activate cv-detect
python generate_demo_gifs.py
```

GIF 輸出至 `assets/demo/`，使用 PCB Defect Dataset v3 (CC BY 4.0) 的影像。
</details>

---

## 功能總覽

| 類別 | 功能 |
|------|------|
| **異常檢測** | 卷積自編碼器 (MSE+SSIM)、PatchCore (ball-tree kNN)、Welford 統計變異模型 |
| **影像處理** | 邊緣偵測、閾值分割、形態學、Blob 分析、HALCON 風格運算子 (16 類) |
| **形狀匹配** | 梯度方向餘弦相似度、金字塔搜尋、次像素精修、NMS |
| **量測** | 次像素邊緣偵測、1D 量測矩形、直線/圓/橢圓擬合 |
| **頻域** | FFT、高斯/Butterworth/帶通/陷波濾波、週期紋理去除 |
| **色彩** | CIE Lab、CIEDE2000 Delta-E、K-means 調色板、均勻性檢測 |
| **OCR / 條碼** | Tesseract + PaddleOCR 雙引擎、ISO 15416/15415 條碼分級 |
| **工程工具** | 相機標定、並行管線、SPC 統計 (Cp/Cpk)、影像拼接 |
| **部署** | ONNX 匯出 (CUDA/CoreML/CPU)、`.cpmodel` 管線模型打包、PyInstaller 執行檔 |
| **GUI** | HALCON HDevelop 風格三面板介面、工業相機串流、PDF 報告 |

> 詳細架構說明請參閱 [`docs/architecture.md`](docs/architecture.md)

## 環境需求

- **Python** 3.10+
- **GPU 支援**（自動偵測）：
  - NVIDIA CUDA（Linux/Windows）
  - Apple Silicon MPS（macOS M1+）
  - 無 GPU 時自動降回 CPU

### 安裝相依套件

```bash
# 建議使用 conda 環境
conda create -n cv-detect python=3.12
conda activate cv-detect

# 基本安裝
pip install -r requirements.txt

# 或使用 pyproject.toml（支援可選套件群組）
pip install -e .                    # 基本安裝
pip install -e ".[onnx]"            # + ONNX 推論
pip install -e ".[ocr,barcode]"     # + OCR + 條碼
pip install -e ".[camera]"          # + 工業相機
pip install -e ".[dev]"             # + 開發工具（pytest, ruff, pyinstaller）
```

主要相依套件：

| 套件 | 用途 |
|------|------|
| torch, torchvision | 深度學習框架（自編碼器、PatchCore） |
| opencv-python | 影像讀取與處理 |
| numpy | 數值運算 |
| Pillow | 影像格式支援 |
| matplotlib | 視覺化繪圖、SPC 管制圖 |
| scipy | 科學計算（高斯濾波等） |
| scikit-image | SSIM 計算 |
| scikit-learn | NearestNeighbors（PatchCore ball-tree） |
| python-dotenv | .env 組態檔載入 |

選用相依套件：

| 套件 | 用途 |
|------|------|
| onnxruntime / onnxruntime-gpu | ONNX 模型推論加速 |
| pytesseract | Tesseract OCR 文字辨識 |
| paddleocr | PaddleOCR 文字辨識 |
| pyzbar | 條碼 / QR Code 解碼 |
| harvesters | GenICam 工業相機支援 |

## 使用方式

### 以 Python 直接執行

```bash
# DL Anomaly Detector
python dl_anomaly/main.py

# Variation Model Inspector
python variation_model/main.py
```

### GUI 操作

啟動後先顯示 splash screen，接著進入 HALCON HDevelop 風格的三面板介面：

- **左側**：管線面板 — 顯示處理步驟與縮圖
- **中央**：影像檢視器 — 支援滾輪縮放、拖曳平移、像素值追蹤
- **右側**：屬性面板 + 操作面板

**工具選單**提供所有進階功能：

| 快捷鍵 | 功能 |
|--------|------|
| `Ctrl+M` | 形狀匹配 |
| `Ctrl+Shift+M` | 量測工具 |
| `Ctrl+R` | ROI 管理 |
| `Ctrl+Shift+P` | PatchCore / ONNX 模型 |
| `Ctrl+Shift+T` | 檢測工具 (FFT / 色彩 / OCR / 條碼) |
| `Ctrl+Shift+E` | 工程工具 (標定 / 管線 / SPC / 拼接) |
| `Ctrl+Shift+V` | MVP 工具 (相機 / 檢測流程 / PDF 報告) |

### 組態設定

透過各模組目錄下的 `.env` 檔案設定參數：

```env
# dl_anomaly/.env 範例
IMAGE_SIZE=256
GRAYSCALE=false
LATENT_DIM=128
BASE_CHANNELS=32
NUM_ENCODER_BLOCKS=4
BATCH_SIZE=16
LEARNING_RATE=0.001
NUM_EPOCHS=100
EARLY_STOPPING_PATIENCE=10
DEVICE=auto                    # auto | cuda | mps | cpu
ANOMALY_THRESHOLD_PERCENTILE=95
SSIM_WEIGHT=0.5
CHECKPOINT_DIR=./checkpoints
RESULTS_DIR=./results
```

## 編譯執行檔

使用 [PyInstaller](https://pyinstaller.org/) 將應用程式打包為獨立執行檔。

### 前置準備

```bash
pip install pyinstaller
```

### macOS

```bash
# DL Anomaly Detector
cd dl_anomaly
pyinstaller build_mac.spec --clean --noconfirm

# Variation Model Inspector
cd variation_model
pyinstaller build_mac.spec --clean --noconfirm
```

> **注意**：兩個專案須依序編譯（不可平行），以避免 PyInstaller bincache 衝突。

產出路徑：
- `dl_anomaly/dist/DL_AnomalyDetector.app`
- `variation_model/dist/VariationModelInspector.app`

### Windows

```bash
cd dl_anomaly
pyinstaller build.spec --noconfirm
```

產出路徑：
- `dl_anomaly\dist\DL_AnomalyDetector\DL_AnomalyDetector.exe`
- `variation_model\dist\VariationModelInspector\VariationModelInspector.exe`

## 測試

```bash
# pytest 測試套件（108 函式 / 121 test cases，含參數化測試）
pytest tests/ -q

# 整合測試
python test_all_functions.py
```

pytest 測試涵蓋：

| 測試模組 | 數量 | 內容 |
|---------|------|------|
| `test_config.py` | 14 (23) | 組態預設值、裝置選擇、布林解析（含參數化） |
| `test_anomaly_scorer.py` | 17 | 逐像素誤差、影像評分、閾值分類 |
| `test_preprocessor.py` | 8 | 前處理 transform、正規化反轉 |
| `test_variation_model.py` | 17 | Welford 演算法、均值/標準差、存讀 |
| `test_validation.py` | 22 (26) | 影像驗證、核大小、範圍檢查（含參數化） |
| `test_pipeline_model.py` | 30 | 路徑處理、建構/儲存/載入、註冊表 CRUD |

## License

Proprietary - TastyByte
