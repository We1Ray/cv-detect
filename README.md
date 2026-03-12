# CV Defect Detection System

基於電腦視覺的瑕疵檢測系統，提供兩種獨立的檢測方法，搭配 HALCON HDevelop 風格的圖形化操作介面。

## 功能模組

### DL Anomaly Detector（深度學習異常檢測）

使用卷積自編碼器（Convolutional Autoencoder）進行瑕疵檢測：

- **訓練**：以正常（良品）影像訓練自編碼器，學習正常樣本的特徵分佈
- **推論**：比對輸入影像與重建影像的差異，差異越大表示越可能有瑕疵
- **評分**：結合 MSE 與 SSIM 的混合異常評分機制
- **閾值**：基於百分位數自動擬合異常閾值

### Variation Model Inspector（統計變異模型檢測）

使用 Welford 線上統計演算法進行瑕疵檢測：

- **訓練**：逐張影像累積各像素的均值與標準差
- **推論**：比對測試影像是否落在統計閾值範圍內
- **參數**：可調整 `abs_threshold` 與 `var_threshold` 控制靈敏度

## 專案結構

```
cv-detect/
├── dl_anomaly/              # 深度學習異常檢測模組
│   ├── main.py              # 應用程式進入點
│   ├── config.py            # 組態管理（dataclass + .env）
│   ├── core/                # 核心演算法
│   │   ├── autoencoder.py   # 卷積自編碼器架構
│   │   ├── anomaly_scorer.py# 異常評分（MSE + SSIM）
│   │   ├── dataset.py       # PyTorch Dataset
│   │   ├── preprocessor.py  # 影像前處理與增強
│   │   ├── halcon_ops.py    # HALCON 影像操作
│   │   ├── region_ops.py    # 區域操作
│   │   ├── region.py        # 區域資料結構
│   │   └── recipe.py        # 處理管線配方
│   ├── pipeline/            # 管線編排
│   │   ├── trainer.py       # 訓練管線（早停、排程器）
│   │   └── inference.py     # 推論管線
│   ├── gui/                 # Tkinter GUI 元件
│   └── visualization/       # 視覺化（熱力圖、報告）
│
├── variation_model/         # 統計變異模型模組
│   ├── main.py              # 應用程式進入點
│   ├── config.py            # 組態管理
│   ├── core/                # 核心演算法
│   │   ├── variation_model.py # Welford 線上統計
│   │   ├── inspector.py     # 結果檢視
│   │   ├── preprocessor.py  # 影像前處理
│   │   └── postprocessor.py # 後處理
│   ├── pipeline/            # 管線編排
│   ├── gui/                 # Tkinter GUI 元件
│   └── visualization/       # 視覺化
│
├── shared/                  # 共用工具模組
│   ├── app_state.py         # 應用程式狀態管理
│   ├── progress_manager.py  # 進度追蹤
│   ├── error_dialog.py      # 錯誤對話框
│   ├── history_panel.py     # 歷史/復原系統
│   ├── op_logger.py         # 操作日誌
│   └── validation.py        # 輸入驗證
│
├── docs/                    # 文件與教學
├── requirements.txt         # Python 相依套件
└── test_all_functions.py    # 測試套件
```

## 環境需求

- **Python** 3.10+
- **CUDA**（選用）：支援 NVIDIA GPU 加速訓練與推論

### 安裝相依套件

```bash
pip install -r requirements.txt
```

主要相依套件：

| 套件 | 用途 |
|------|------|
| torch, torchvision | 深度學習框架（自編碼器訓練/推論） |
| opencv-python | 影像讀取與處理 |
| numpy | 數值運算 |
| Pillow | 影像格式支援 |
| matplotlib | 視覺化繪圖 |
| scipy | 科學計算（高斯濾波等） |
| scikit-image | SSIM 計算 |
| python-dotenv | .env 組態檔載入 |

## 使用方式

### 以 Python 直接執行

```bash
# DL Anomaly Detector
python dl_anomaly/main.py

# Variation Model Inspector
python variation_model/main.py
```

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
DEVICE=cuda
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
pyinstaller build_mac.spec --noconfirm --distpath dist --workpath build

# Variation Model Inspector
cd variation_model
pyinstaller build_mac.spec --noconfirm --distpath dist --workpath build
```

或直接執行建置腳本：

```bash
bash dl_anomaly/build_mac.sh
bash variation_model/build_mac.sh
```

產出路徑：
- `dl_anomaly/dist/DL_AnomalyDetector.app`
- `variation_model/dist/VariationModelInspector.app`

### Windows

```bash
# DL Anomaly Detector
cd dl_anomaly
pyinstaller build.spec --noconfirm --distpath dist --workpath build

# Variation Model Inspector（需自行建立 build.spec）
cd variation_model
pyinstaller --name VariationModelInspector --windowed --noconfirm ^
    --add-data "..\shared;shared" ^
    --hidden-import config --hidden-import core --hidden-import pipeline ^
    --hidden-import gui --hidden-import visualization ^
    --hidden-import shared --hidden-import shared.app_state ^
    --hidden-import shared.progress_manager --hidden-import shared.error_dialog ^
    --hidden-import shared.history_panel ^
    --distpath dist --workpath build ^
    main.py
```

或使用批次腳本：

```bat
dl_anomaly\BUILD.bat
```

產出路徑：
- `dl_anomaly\dist\DL_AnomalyDetector\DL_AnomalyDetector.exe`
- `variation_model\dist\VariationModelInspector\VariationModelInspector.exe`

## 測試

```bash
python test_all_functions.py
```

測試套件涵蓋：共用模組、組態、前處理、自編碼器、資料集、異常評分、HALCON 操作、區域運算、配方系統、訓練管線、推論管線、視覺化。

## 技術架構

### 自編碼器架構（DL Anomaly）

```
Input → [ConvBlock + MaxPool] × N → AdaptiveAvgPool(4) → Linear(latent_dim)
                                                              ↓
Output ← [Upsample + ConvBlock] × N ← Linear ← Linear(latent_dim)
```

- 殘差捷徑（Residual Shortcuts）
- BatchNorm + LeakyReLU(0.2)
- 可配置：`latent_dim`、`base_channels`、`num_encoder_blocks`

### 訓練流程

1. 載入良品影像 → 建立 `DefectFreeDataset`
2. MSE + SSIM 混合損失函數
3. AdamW 優化器 + CosineAnnealingLR 學習率排程
4. 早停機制（patience-based）+ Checkpoint 儲存
5. 訓練完成後自動擬合異常閾值

### 推論流程

1. 載入 Checkpoint → 前處理輸入影像
2. 自編碼器重建 → 計算逐像素誤差圖（MSE + SSIM）
3. 計算影像級異常分數 → 與閾值比較 → 判定良品/瑕疵
4. 輸出：`InspectionResult`（原圖、重建圖、誤差圖、瑕疵遮罩、分數）

## License

Proprietary - TastyByte
