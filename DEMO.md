# CV Defect Detection — PCB 瑕疵檢測 Demo

本文件示範如何使用 CV Defect Detection System 對 PCB（印刷電路板）影像進行瑕疵檢測。測試資料使用 [PCB Defect Dataset v3](https://universe.roboflow.com/pcbdefect-eftc9/pcb-defect-djz4r/dataset/3)（CC BY 4.0），包含 6 種缺陷類型、共 600 張訓練影像。

## 資料集概覽

| 缺陷類型 | 英文名稱 | 訓練影像數 | 說明 |
|---------|---------|-----------|------|
| 缺孔 | missing_hole | 87 | 焊接孔缺失 |
| 鼠咬 | mouse_bite | 93 | 銅箔邊緣不規則缺口 |
| 斷路 | open_circuit | 103 | 導線斷裂 |
| 短路 | short | 105 | 導線異常連接 |
| 毛刺 | spur | 212 | 銅箔邊緣多餘突出 |
| 殘銅 | spurious_copper | 104 | 不應存在的多餘銅箔 |

## 環境準備

```bash
# 建立虛擬環境
conda create -n cv-detect python=3.12
conda activate cv-detect

# 安裝相依套件
pip install -r requirements.txt
```

## 測試資料路徑

```
/Users/WeiRay/Downloads/pcb defect.v3i.yolov7pytorch/
├── train/images/    # 600 張訓練影像
├── valid/images/    # 驗證集
└── test/images/     # 測試集
```

---

## Demo 1：自編碼器異常檢測（Autoencoder）

以卷積自編碼器學習 PCB 正常外觀，透過重建誤差偵測瑕疵。

### Step 1 — 訓練模型

```python
import sys
sys.path.insert(0, "/Users/WeiRay/Desktop/Work/TastyByte/cv-detect")

from pathlib import Path
from dl_anomaly.config import Config
from dl_anomaly.pipeline.trainer import TrainingPipeline

# 設定訓練參數
config = Config(
    train_image_dir=Path("/Users/WeiRay/Downloads/pcb defect.v3i.yolov7pytorch/train/images"),
    image_size=256,
    grayscale=False,
    latent_dim=128,
    base_channels=32,
    num_encoder_blocks=4,
    batch_size=16,
    learning_rate=0.001,
    num_epochs=50,
    early_stopping_patience=10,
    ssim_weight=0.5,
    anomaly_threshold_percentile=95,
)

# 開始訓練
trainer = TrainingPipeline(config)
result = trainer.run(
    progress_callback=lambda info: print(
        f"Epoch {info['epoch']:03d}/{info['total_epochs']}  "
        f"train={info['train_loss']:.6f}  val={info['val_loss']:.6f}"
    )
)

print(f"\n訓練完成！")
print(f"  最佳驗證損失: {result['best_val_loss']:.6f}")
print(f"  異常閾值:     {result['threshold']:.6f}")
print(f"  模型儲存於:   {result['checkpoint_path']}")
```

### Step 2 — 單張推論

```python
from dl_anomaly.pipeline.inference import InferencePipeline

# 載入已訓練的模型
pipeline = InferencePipeline(
    checkpoint_path=config.checkpoint_dir / "final_model.pt"
)

# 檢測單張影像
test_image = "/Users/WeiRay/Downloads/pcb defect.v3i.yolov7pytorch/test/images"
import os
test_files = sorted([
    os.path.join(test_image, f) for f in os.listdir(test_image)
    if f.lower().endswith(('.jpg', '.png', '.bmp'))
])

result = pipeline.inspect_single(test_files[0])

print(f"影像:       {Path(test_files[0]).name}")
print(f"異常分數:   {result.anomaly_score:.6f}")
print(f"是否瑕疵:   {'瑕疵' if result.is_defective else '良品'}")
print(f"瑕疵區域數: {len(result.defect_regions)}")
for region in result.defect_regions:
    print(f"  區域 {region['id']}: bbox={region['bbox']}, 面積={region['area']}px")
```

### Step 3 — 批次檢測與視覺化

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 批次檢測
results = pipeline.inspect_batch(
    "/Users/WeiRay/Downloads/pcb defect.v3i.yolov7pytorch/test/images",
    progress_callback=lambda cur, total: print(f"\r檢測進度: {cur}/{total}", end="")
)

# 視覺化前 6 張結果
fig, axes = plt.subplots(3, 6, figsize=(24, 12))
for i, res in enumerate(results[:6]):
    # 原圖
    axes[0, i].imshow(cv2.cvtColor(res.original, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f"Score: {res.anomaly_score:.4f}", fontsize=10)
    axes[0, i].axis("off")

    # 誤差熱力圖
    axes[1, i].imshow(res.error_map, cmap="jet")
    axes[1, i].set_title("誤差圖", fontsize=10)
    axes[1, i].axis("off")

    # 瑕疵遮罩疊加
    overlay = cv2.cvtColor(res.original.copy(), cv2.COLOR_BGR2RGB)
    mask_colored = np.zeros_like(overlay)
    mask_colored[res.defect_mask > 0] = [255, 0, 0]
    blended = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    axes[2, i].imshow(blended)
    label = "瑕疵" if res.is_defective else "良品"
    axes[2, i].set_title(f"判定: {label}", fontsize=10,
                          color="red" if res.is_defective else "green")
    axes[2, i].axis("off")

axes[0, 0].set_ylabel("原圖", fontsize=12)
axes[1, 0].set_ylabel("誤差圖", fontsize=12)
axes[2, 0].set_ylabel("瑕疵遮罩", fontsize=12)

plt.suptitle("PCB 瑕疵檢測結果 — Autoencoder", fontsize=16)
plt.tight_layout()
plt.savefig("demo_autoencoder_results.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Step 4 — 統計摘要

```python
# 統計報告
n_total = len(results)
n_defective = sum(1 for r in results if r.is_defective)
scores = [r.anomaly_score for r in results]

print(f"\n{'='*50}")
print(f"PCB 瑕疵檢測統計報告 — Autoencoder")
print(f"{'='*50}")
print(f"  總檢測數:   {n_total}")
print(f"  瑕疵數:     {n_defective}")
print(f"  良品數:     {n_total - n_defective}")
print(f"  瑕疵率:     {n_defective/n_total*100:.1f}%")
print(f"  平均分數:   {np.mean(scores):.6f}")
print(f"  最大分數:   {np.max(scores):.6f}")
print(f"  最小分數:   {np.min(scores):.6f}")
print(f"  閾值:       {pipeline.scorer.threshold:.6f}")
```

---

## Demo 2：PatchCore 異常檢測

使用預訓練 CNN 特徵 + 記憶庫進行異常檢測，無需訓練自編碼器，適合少量樣本場景。

### Step 1 — 建立記憶庫

```python
import sys
sys.path.insert(0, "/Users/WeiRay/Desktop/Work/TastyByte/cv-detect")

from pathlib import Path
from dl_anomaly.config import Config
from shared.core.patchcore import PatchCoreTrainer

config = Config(
    train_image_dir=Path("/Users/WeiRay/Downloads/pcb defect.v3i.yolov7pytorch/train/images"),
    image_size=224,
)

# 建立 PatchCore 記憶庫（使用 Wide ResNet-50 骨幹網路）
trainer = PatchCoreTrainer(
    config=config,
    backbone_name="wide_resnet50_2",
    layers=("layer2", "layer3"),
    coreset_ratio=0.01,           # 保留 1% 核心特徵
    device=config.device,
)

model = trainer.train(
    image_dir=config.train_image_dir,
    progress_callback=lambda info: print(
        f"\r{info.get('phase', 'Training')}... {info.get('progress', 0):.0%}", end=""
    )
)

# 儲存記憶庫
model.save("models/patchcore_pcb.npz")
print(f"\n記憶庫儲存完成！特徵數: {model.memory_bank.shape[0]}")
```

### Step 2 — 推論

```python
from shared.core.patchcore import PatchCoreModel, PatchCoreInference

# 載入記憶庫
model = PatchCoreModel.load("models/patchcore_pcb.npz")

# 建立推論引擎
engine = PatchCoreInference(
    model=model,
    device=config.device,
    n_neighbors=9,
    smooth_sigma=4.0,
)

# 單張檢測
result = engine.inspect_single(
    "/Users/WeiRay/Downloads/pcb defect.v3i.yolov7pytorch/test/images/"
    "01_missing_hole_01_jpg.rf.3caae8342e3835b61b8901d1bdb9b36b.jpg"
)

print(f"異常分數:   {result.anomaly_score:.6f}")
print(f"是否瑕疵:   {'瑕疵' if result.is_defective else '良品'}")
print(f"瑕疵區域數: {len(result.defect_regions)}")
```

### Step 3 — 批次檢測比較

```python
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 批次檢測
results = engine.inspect_batch(
    "/Users/WeiRay/Downloads/pcb defect.v3i.yolov7pytorch/test/images"
)

# 依異常分數排序，取出分數最高的 8 張
results_sorted = sorted(results, key=lambda r: r.anomaly_score, reverse=True)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, res in enumerate(results_sorted[:8]):
    row, col = divmod(i, 4)

    # 疊加誤差熱力圖到原圖
    heatmap = (res.error_map * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    original_rgb = cv2.cvtColor(res.original, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # 調整大小一致
    h, w = original_rgb.shape[:2]
    heatmap_rgb = cv2.resize(heatmap_rgb, (w, h))
    overlay = cv2.addWeighted(original_rgb, 0.6, heatmap_rgb, 0.4, 0)

    axes[row, col].imshow(overlay)
    axes[row, col].set_title(
        f"Score: {res.anomaly_score:.4f}",
        fontsize=11,
        color="red" if res.is_defective else "green",
        fontweight="bold",
    )
    axes[row, col].axis("off")

plt.suptitle("PCB 瑕疵檢測結果 — PatchCore（異常分數 Top 8）", fontsize=16)
plt.tight_layout()
plt.savefig("demo_patchcore_results.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

## Demo 3：GUI 圖形化操作

啟動 Industrial Vision 風格的圖形化介面，支援互動式操作。

```bash
cd /Users/WeiRay/Desktop/Work/TastyByte/cv-detect

# 啟動 DL Anomaly Detector
python dl_anomaly/main.py
```

### GUI 操作流程

1. **載入影像**：`檔案` → `開啟影像` → 選取 PCB 影像
2. **訓練模型**：
   - 切換到 `訓練` 分頁
   - 設定訓練影像目錄：`/Users/WeiRay/Downloads/pcb defect.v3i.yolov7pytorch/train/images`
   - 調整參數（影像大小、epoch 數等）
   - 點擊 `開始訓練`
3. **執行檢測**：
   - 切換到 `檢測` 分頁
   - 載入測試影像
   - 查看異常分數、誤差熱力圖、瑕疵遮罩
4. **進階工具**：
   - `Ctrl+Shift+P`：開啟 PatchCore 對話框
   - `Ctrl+Shift+T`：FFT / 色彩 / OCR 檢測工具
   - `Ctrl+Shift+E`：標定 / SPC / 影像拼接

---

## Demo 4：完整腳本 — 一鍵執行

將以下腳本存為 `demo_pcb.py`，一鍵完成訓練 + 檢測 + 視覺化：

```python
#!/usr/bin/env python3
"""PCB 瑕疵檢測完整 Demo — 自編碼器 + PatchCore 雙引擎比較"""

import sys
sys.path.insert(0, "/Users/WeiRay/Desktop/Work/TastyByte/cv-detect")

import os
import logging
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── 路徑設定 ──────────────────────────────────────────────
TRAIN_DIR = Path("/Users/WeiRay/Downloads/pcb defect.v3i.yolov7pytorch/train/images")
TEST_DIR  = Path("/Users/WeiRay/Downloads/pcb defect.v3i.yolov7pytorch/test/images")

# ── 方法 1：Autoencoder ──────────────────────────────────
print("\n" + "="*60)
print("  方法 1：卷積自編碼器異常檢測")
print("="*60)

from dl_anomaly.config import Config
from dl_anomaly.pipeline.trainer import TrainingPipeline
from dl_anomaly.pipeline.inference import InferencePipeline

config = Config(
    train_image_dir=TRAIN_DIR,
    image_size=256,
    grayscale=False,
    latent_dim=128,
    base_channels=32,
    num_encoder_blocks=4,
    batch_size=16,
    learning_rate=0.001,
    num_epochs=30,          # Demo 用較少 epoch
    early_stopping_patience=8,
    ssim_weight=0.5,
    anomaly_threshold_percentile=95,
)

# 訓練
trainer = TrainingPipeline(config)
train_result = trainer.run(
    progress_callback=lambda info: print(
        f"\r  [AE] Epoch {info['epoch']:03d}/{info['total_epochs']}  "
        f"loss={info['train_loss']:.6f}", end=""
    )
)
print(f"\n  → 訓練完成，閾值: {train_result['threshold']:.6f}")

# 推論
ae_pipeline = InferencePipeline(checkpoint_path=train_result["checkpoint_path"])
ae_results = ae_pipeline.inspect_batch(TEST_DIR)
print(f"  → 檢測完成: {len(ae_results)} 張影像")

# ── 方法 2：PatchCore ────────────────────────────────────
print("\n" + "="*60)
print("  方法 2：PatchCore 記憶庫檢測")
print("="*60)

from shared.core.patchcore import PatchCoreTrainer, PatchCoreInference

pc_config = Config(train_image_dir=TRAIN_DIR, image_size=224)
pc_trainer = PatchCoreTrainer(
    config=pc_config,
    backbone_name="resnet18",   # Demo 用輕量骨幹加速
    coreset_ratio=0.01,
    device=pc_config.device,
)

pc_model = pc_trainer.train(TRAIN_DIR)
print(f"  → 記憶庫建立完成，特徵數: {pc_model.memory_bank.shape[0]}")

pc_engine = PatchCoreInference(model=pc_model, device=pc_config.device)
pc_results = pc_engine.inspect_batch(TEST_DIR)
print(f"  → 檢測完成: {len(pc_results)} 張影像")

# ── 視覺化比較 ───────────────────────────────────────────
print("\n" + "="*60)
print("  視覺化比較結果")
print("="*60)

n_show = min(5, len(ae_results), len(pc_results))
fig, axes = plt.subplots(3, n_show, figsize=(4 * n_show, 12))

for i in range(n_show):
    ae_r = ae_results[i]
    pc_r = pc_results[i]

    # 原圖
    orig = cv2.cvtColor(ae_r.original, cv2.COLOR_BGR2RGB)
    axes[0, i].imshow(orig)
    axes[0, i].set_title("原圖", fontsize=10)
    axes[0, i].axis("off")

    # AE 誤差圖
    axes[1, i].imshow(ae_r.error_map, cmap="jet")
    label = "瑕疵" if ae_r.is_defective else "良品"
    axes[1, i].set_title(f"AE: {ae_r.anomaly_score:.4f} ({label})", fontsize=10,
                          color="red" if ae_r.is_defective else "green")
    axes[1, i].axis("off")

    # PatchCore 誤差圖
    axes[2, i].imshow(pc_r.error_map, cmap="jet")
    label = "瑕疵" if pc_r.is_defective else "良品"
    axes[2, i].set_title(f"PC: {pc_r.anomaly_score:.4f} ({label})", fontsize=10,
                          color="red" if pc_r.is_defective else "green")
    axes[2, i].axis("off")

axes[0, 0].set_ylabel("原圖", fontsize=12)
axes[1, 0].set_ylabel("Autoencoder", fontsize=12)
axes[2, 0].set_ylabel("PatchCore", fontsize=12)

plt.suptitle("PCB 瑕疵檢測 — 雙引擎比較", fontsize=16)
plt.tight_layout()
plt.savefig("demo_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 統計報告 ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  檢測統計")
print(f"{'='*60}")

for name, res_list in [("Autoencoder", ae_results), ("PatchCore", pc_results)]:
    n_def = sum(1 for r in res_list if r.is_defective)
    scores = [r.anomaly_score for r in res_list]
    print(f"\n  [{name}]")
    print(f"    檢測數:   {len(res_list)}")
    print(f"    瑕疵數:   {n_def} ({n_def/len(res_list)*100:.1f}%)")
    print(f"    分數範圍: {min(scores):.6f} ~ {max(scores):.6f}")
    print(f"    平均分數: {np.mean(scores):.6f}")

print(f"\n結果已儲存: demo_comparison.png")
```

執行方式：

```bash
cd /Users/WeiRay/Desktop/Work/TastyByte/cv-detect
python demo_pcb.py
```

---

## 預期輸出

### 訓練過程

```
  [AE] Epoch 001/030  loss=0.142567
  [AE] Epoch 002/030  loss=0.098234
  ...
  → 訓練完成，閾值: 0.023456
  → 檢測完成: XX 張影像
```

### 檢測結果範例

```
==================================================
  檢測統計
==================================================

  [Autoencoder]
    檢測數:   XX
    瑕疵數:   XX (XX.X%)
    分數範圍: 0.XXXXXX ~ 0.XXXXXX
    平均分數: 0.XXXXXX

  [PatchCore]
    檢測數:   XX
    瑕疵數:   XX (XX.X%)
    分數範圍: 0.XXXXXX ~ 0.XXXXXX
    平均分數: 0.XXXXXX
```

### 視覺化輸出

執行後會產生以下圖片：

| 檔案 | 內容 |
|------|------|
| `demo_autoencoder_results.png` | 自編碼器檢測結果（原圖 / 誤差圖 / 瑕疵遮罩） |
| `demo_patchcore_results.png` | PatchCore 檢測結果（熱力圖疊加） |
| `demo_comparison.png` | 雙引擎並排比較 |

---

## 方法比較

| 特性 | Autoencoder | PatchCore |
|------|------------|-----------|
| 訓練時間 | 較長（需多輪 epoch） | 較短（單次特徵萃取） |
| 推論速度 | 快（單次前向傳播） | 中等（kNN 搜尋） |
| 記憶體用量 | 低（僅模型參數） | 高（記憶庫） |
| 少量樣本 | 需足夠訓練資料 | 少量即可（~20 張） |
| 定位精度 | 像素級（但較模糊） | 像素級（邊界清晰） |
| 適用場景 | 紋理型瑕疵 | 結構型 / 複雜瑕疵 |

## 注意事項

- 本 Demo 使用 PCB 瑕疵影像作為訓練資料。實際生產場景中，應使用**良品影像**進行訓練，讓模型學習正常外觀
- 首次執行 PatchCore 會自動下載預訓練骨幹模型（~100MB），需要網路連線
- Apple Silicon (M1/M2/M3) 會自動使用 MPS 加速；NVIDIA GPU 自動使用 CUDA
- 訓練過程中可透過 GUI 的「停止」按鈕中斷訓練
