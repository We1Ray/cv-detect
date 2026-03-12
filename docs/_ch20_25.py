# -*- coding: utf-8 -*-
"""Chapter 20-25 content"""


def write_ch20(p):
    p.ch_title("深度學習瑕疵檢測進階")
    p.txt(
        "隨著深度學習技術的快速發展，基於 CNN 的目標檢測、語義分割和異常檢測方法已成為工業瑕疵檢測的重要工具。"
        "相比傳統方法，深度學習能自動學習特徵，在複雜場景中表現更優。"
        "本章涵蓋 YOLO 目標檢測、U-Net 語義分割、PatchCore/PADIM 無監督異常檢測、遷移學習策略、資料增強技巧以及模型評估與部署。"
    )

    # 20.1
    p.sec_title("目標檢測：YOLO 系列應用於瑕疵檢測")
    p.txt(
        "YOLO（You Only Look Once）是單階段目標檢測模型，將影像分成網格，"
        "每個網格同時預測邊界框和類別機率。YOLO 系列以其高速和高精度著稱，"
        "非常適合產線即時瑕疵檢測。YOLOv5 到 YOLOv8 在工業場景中被廣泛使用。"
    )

    p.formula("IoU（交並比）",
              "IoU = Area(A ∩ B) / Area(A ∪ B)\n\n"
              "  A = 預測邊界框\n"
              "  B = 真實邊界框\n\n"
              "  IoU = 1 : 完美重合\n"
              "  IoU = 0 : 完全不重疊\n"
              "  IoU > 0.5 : 通常視為正確檢測",
              "IoU 是目標檢測中最核心的評估指標，用於衡量預測框與真實框的重疊程度。")

    p.formula("NMS（非極大值抑制）",
              "輸入: B = {b1, b2, ..., bn}  (候選框集合)\n"
              "      S = {s1, s2, ..., sn}  (對應信心分數)\n\n"
              "步驟:\n"
              "  1. 選出最高分數的框 b_max\n"
              "  2. 移除所有與 b_max 的 IoU > T_nms 的框\n"
              "  3. 重複直到無剩餘框\n\n"
              "  T_nms 通常取 0.45 ~ 0.65",
              "NMS 用於消除重複檢測。同一瑕疵可能產生多個重疊框，NMS 僅保留信心最高的一個。")

    p.code("from ultralytics import YOLO\nimport cv2\n\n"
           "# 載入預訓練或自訓練模型\n"
           "model = YOLO('best.pt')  # 自訓練的瑕疵檢測模型\n\n"
           "# 推論\n"
           "img = cv2.imread('test_part.jpg')\n"
           "results = model(img, conf=0.25, iou=0.45)\n\n"
           "# 解析結果\n"
           "for r in results:\n"
           "    boxes = r.boxes\n"
           "    for box in boxes:\n"
           "        x1, y1, x2, y2 = box.xyxy[0].int().tolist()\n"
           "        conf = float(box.conf[0])\n"
           "        cls = int(box.cls[0])\n"
           "        label = model.names[cls]\n"
           "        print(f'{label}: {conf:.2f} '\n"
           "              f'@ ({x1},{y1})-({x2},{y2})')\n\n"
           "        # 繪製\n"
           "        cv2.rectangle(img, (x1,y1), (x2,y2),\n"
           "                      (0,0,255), 2)\n"
           "        cv2.putText(img, f'{label} {conf:.2f}',\n"
           "                    (x1, y1-5),\n"
           "                    cv2.FONT_HERSHEY_SIMPLEX,\n"
           "                    0.5, (0,0,255), 1)")

    p.table(
        ["版本", "特點", "速度(FPS)", "適用場景"],
        [
            ["YOLOv5", "PyTorch 原生, 社群活躍", "~140", "通用瑕疵檢測"],
            ["YOLOv7", "高精度, E-ELAN 架構", "~120", "高精度需求"],
            ["YOLOv8", "統一框架, 支援分割", "~160", "檢測+分割場景"],
            ["YOLOv9", "PGI+GELAN, 資訊保留", "~130", "小目標瑕疵"],
            ["YOLO11", "輕量化, C3k2 模組", "~180", "邊緣部署"],
        ]
    )

    # 20.2
    p.sec_title("語義分割：U-Net 用於瑕疵像素級分割")
    p.txt(
        "語義分割為每個像素分配類別標籤，實現像素級的瑕疵定位。"
        "U-Net 是最經典的分割架構，採用編碼器-解碼器結構和跳躍連接，"
        "在醫學影像和工業檢測中表現優異。跳躍連接將編碼器的低層特徵傳遞給解碼器，"
        "幫助恢復空間細節，這對精確定位瑕疵邊界至關重要。"
    )

    p.formula("Dice Loss",
              "Dice = 2 * |A ∩ B| / (|A| + |B|)\n\n"
              "Dice Loss = 1 - Dice\n\n"
              "  = 1 - 2*SUM(p_i * g_i) / (SUM(p_i) + SUM(g_i))\n\n"
              "  p_i = 預測機率\n"
              "  g_i = 真實標籤 (0 或 1)",
              "Dice Loss 直接優化前景/背景的重疊度。在瑕疵佔比極小時（類別不平衡），"
              "比 BCE 更穩定，因為它不會被大量背景像素主導。")

    p.formula("BCE + Dice 組合損失",
              "Loss = alpha * BCE + (1-alpha) * Dice_Loss\n\n"
              "BCE = -1/N * SUM[g*log(p) + (1-g)*log(1-p)]\n\n"
              "alpha 通常取 0.5",
              "組合損失結合了 BCE 的像素級精確優化和 Dice 的全局重疊度優化，效果通常優於單獨使用任何一種。")

    p.code("import torch\nimport torch.nn as nn\n\n"
           "class UNetBlock(nn.Module):\n"
           "    def __init__(self, in_c, out_c):\n"
           "        super().__init__()\n"
           "        self.conv = nn.Sequential(\n"
           "            nn.Conv2d(in_c, out_c, 3, padding=1),\n"
           "            nn.BatchNorm2d(out_c),\n"
           "            nn.ReLU(inplace=True),\n"
           "            nn.Conv2d(out_c, out_c, 3, padding=1),\n"
           "            nn.BatchNorm2d(out_c),\n"
           "            nn.ReLU(inplace=True))\n"
           "    def forward(self, x):\n"
           "        return self.conv(x)\n\n"
           "class SimpleUNet(nn.Module):\n"
           "    def __init__(self, in_ch=1, out_ch=1):\n"
           "        super().__init__()\n"
           "        self.enc1 = UNetBlock(in_ch, 64)\n"
           "        self.enc2 = UNetBlock(64, 128)\n"
           "        self.pool = nn.MaxPool2d(2)\n"
           "        self.bottleneck = UNetBlock(128, 256)\n"
           "        self.up2 = nn.ConvTranspose2d(\n"
           "            256, 128, 2, stride=2)\n"
           "        self.dec2 = UNetBlock(256, 128)\n"
           "        self.up1 = nn.ConvTranspose2d(\n"
           "            128, 64, 2, stride=2)\n"
           "        self.dec1 = UNetBlock(128, 64)\n"
           "        self.out_conv = nn.Conv2d(64, out_ch, 1)\n\n"
           "    def forward(self, x):\n"
           "        e1 = self.enc1(x)\n"
           "        e2 = self.enc2(self.pool(e1))\n"
           "        b = self.bottleneck(self.pool(e2))\n"
           "        d2 = self.dec2(torch.cat(\n"
           "            [self.up2(b), e2], dim=1))\n"
           "        d1 = self.dec1(torch.cat(\n"
           "            [self.up1(d2), e1], dim=1))\n"
           "        return torch.sigmoid(self.out_conv(d1))")

    # 20.3
    p.sec_title("無監督異常檢測進階：PatchCore, PADIM")
    p.txt(
        "無監督異常檢測方法僅需良品影像即可訓練，無需收集和標註瑕疵樣本。"
        "這在工業場景中極具價值，因為瑕疵樣本稀少且類型多變。\n\n"
        "PatchCore 的核心思想是建立「正常特徵記憶庫」：\n"
        "1. 使用預訓練 CNN（如 ResNet）提取良品影像的中間層特徵\n"
        "2. 將所有良品的 patch 特徵收集為記憶庫\n"
        "3. 使用 coreset 採樣壓縮記憶庫（保留代表性特徵）\n"
        "4. 測試時，計算每個 patch 特徵到記憶庫的最近鄰距離\n"
        "5. 距離越大，越可能是異常\n\n"
        "PADIM（Patch Distribution Modeling）則用多元高斯分佈對每個位置的正常特徵建模。"
        "測試時計算 Mahalanobis 距離作為異常分數。"
    )

    p.code("# PatchCore 推論流程 (anomalib)\n"
           "from anomalib.models import Patchcore\n"
           "from anomalib.data import MVTec\n"
           "from anomalib.engine import Engine\n\n"
           "# 配置\n"
           "model = Patchcore(\n"
           "    backbone='wide_resnet50_2',\n"
           "    layers=['layer2', 'layer3'],\n"
           "    coreset_sampling_ratio=0.1,\n"
           "    num_neighbors=9\n"
           ")\n\n"
           "# 資料模組\n"
           "datamodule = MVTec(\n"
           "    root='./datasets/MVTec',\n"
           "    category='bottle',\n"
           "    image_size=(256, 256),\n"
           "    train_batch_size=32\n"
           ")\n\n"
           "# 訓練（僅使用良品）與推論\n"
           "engine = Engine()\n"
           "engine.fit(model, datamodule=datamodule)\n"
           "predictions = engine.predict(\n"
           "    model, datamodule=datamodule)\n\n"
           "# 每張影像會得到:\n"
           "# - anomaly_map: 像素級異常分數圖\n"
           "# - pred_score: 影像級異常分數\n"
           "# - pred_label: 0=正常, 1=異常")

    # 20.4
    p.sec_title("遷移學習與微調策略")
    p.txt(
        "工業瑕疵檢測的資料量通常有限（幾十到幾百張）。遷移學習利用在大規模資料集（ImageNet）"
        "上預訓練的模型作為起點，僅需少量目標域資料即可達到良好效果。\n\n"
        "策略選擇：\n"
        "- 資料極少（< 100 張）：僅訓練最後分類層，凍結所有卷積層\n"
        "- 資料中等（100~1000 張）：凍結前半部分，微調後半部分\n"
        "- 資料充足（> 1000 張）：全網路微調，使用較小學習率\n\n"
        "學習率排程建議：使用差異學習率（底層小、頂層大）和 CosineAnnealing 排程器。"
    )

    p.code("import torch\nimport torchvision.models as models\nimport torch.nn as nn\n\n"
           "# 載入預訓練 ResNet50\n"
           "model = models.resnet50(pretrained=True)\n\n"
           "# 策略: 凍結前 3/4 的層\n"
           "for name, param in model.named_parameters():\n"
           "    if 'layer4' not in name and 'fc' not in name:\n"
           "        param.requires_grad = False\n\n"
           "# 替換分類頭\n"
           "num_defect_types = 5  # OK + 4 種瑕疵\n"
           "model.fc = nn.Sequential(\n"
           "    nn.Dropout(0.3),\n"
           "    nn.Linear(2048, 256),\n"
           "    nn.ReLU(),\n"
           "    nn.Dropout(0.2),\n"
           "    nn.Linear(256, num_defect_types)\n"
           ")\n\n"
           "# 差異學習率\n"
           "optimizer = torch.optim.AdamW([\n"
           "    {'params': model.layer4.parameters(),\n"
           "     'lr': 1e-4},\n"
           "    {'params': model.fc.parameters(),\n"
           "     'lr': 1e-3},\n"
           "], weight_decay=1e-4)\n\n"
           "# CosineAnnealing 排程\n"
           "scheduler = torch.optim.lr_scheduler\\\n"
           "    .CosineAnnealingLR(optimizer, T_max=50)")

    # 20.5
    p.sec_title("資料增強策略")
    p.txt(
        "資料增強是解決工業檢測資料不足的關鍵技術。通過對訓練影像施加隨機變換，"
        "人為擴充資料集的多樣性，提升模型的泛化能力。\n\n"
        "常用增強方式：\n"
        "- 幾何變換：旋轉、翻轉、縮放、平移、仿射\n"
        "- 色彩變換：亮度/對比度抖動、色相偏移、灰度化\n"
        "- 遮擋類：CutOut（隨機遮擋）、GridMask\n"
        "- 混合類：MixUp（線性混合兩張圖）、CutMix（區域替換）"
    )

    p.code("import albumentations as A\nfrom albumentations.pytorch import ToTensorV2\n\n"
           "# 瑕疵檢測專用資料增強管道\n"
           "train_transform = A.Compose([\n"
           "    # 幾何變換\n"
           "    A.HorizontalFlip(p=0.5),\n"
           "    A.VerticalFlip(p=0.5),\n"
           "    A.RandomRotate90(p=0.5),\n"
           "    A.ShiftScaleRotate(\n"
           "        shift_limit=0.05,\n"
           "        scale_limit=0.1,\n"
           "        rotate_limit=15, p=0.5),\n\n"
           "    # 色彩變換\n"
           "    A.RandomBrightnessContrast(\n"
           "        brightness_limit=0.15,\n"
           "        contrast_limit=0.15, p=0.5),\n"
           "    A.HueSaturationValue(\n"
           "        hue_shift_limit=10,\n"
           "        sat_shift_limit=20,\n"
           "        val_shift_limit=15, p=0.3),\n\n"
           "    # 雜訊與模糊\n"
           "    A.GaussNoise(var_limit=(5,25), p=0.3),\n"
           "    A.GaussianBlur(\n"
           "        blur_limit=(3,5), p=0.2),\n\n"
           "    # CutOut 遮擋\n"
           "    A.CoarseDropout(\n"
           "        max_holes=4, max_height=32,\n"
           "        max_width=32, p=0.3),\n\n"
           "    # 正規化\n"
           "    A.Normalize(\n"
           "        mean=[0.485, 0.456, 0.406],\n"
           "        std=[0.229, 0.224, 0.225]),\n"
           "    ToTensorV2()\n"
           "])")

    # 20.6
    p.sec_title("模型評估與部署")
    p.txt(
        "深度學習模型的評估不能僅看整體準確率。工業場景中需要關注：\n"
        "- 混淆矩陣：各類別的具體表現\n"
        "- PR 曲線：不同閾值下的精確率-召回率取捨\n"
        "- AP/mAP：目標檢測的標準指標\n"
        "- 推論速度：是否滿足產線節拍時間"
    )

    p.code("import numpy as np\nfrom sklearn.metrics import (\n"
           "    confusion_matrix, precision_recall_curve,\n"
           "    average_precision_score\n"
           ")\n\n"
           "def compute_metrics(y_true, y_scores, y_pred):\n"
           '    """計算完整的評估指標"""\n'
           "    # 混淆矩陣\n"
           "    cm = confusion_matrix(y_true, y_pred)\n"
           "    tn, fp, fn, tp = cm.ravel()\n\n"
           "    # 基本指標\n"
           "    precision = tp / (tp + fp + 1e-8)\n"
           "    recall = tp / (tp + fn + 1e-8)\n"
           "    f1 = 2*precision*recall / (\n"
           "        precision + recall + 1e-8)\n"
           "    specificity = tn / (tn + fp + 1e-8)\n\n"
           "    # PR 曲線與 AP\n"
           "    prec_curve, rec_curve, thresholds = \\\n"
           "        precision_recall_curve(y_true, y_scores)\n"
           "    ap = average_precision_score(\n"
           "        y_true, y_scores)\n\n"
           "    # 漏檢率（最關鍵）\n"
           "    escape_rate = fn / (tp + fn + 1e-8)\n\n"
           "    return {\n"
           "        'confusion_matrix': cm,\n"
           "        'precision': precision,\n"
           "        'recall': recall,\n"
           "        'f1': f1,\n"
           "        'specificity': specificity,\n"
           "        'AP': ap,\n"
           "        'escape_rate': escape_rate,\n"
           "        'pr_curve': (prec_curve, rec_curve)\n"
           "    }")

    p.case("MVTec AD 資料集異常檢測",
           "MVTec AD 是工業異常檢測的標準基準資料集，包含 15 個類別（5 種紋理 + 10 種物件）。\n\n"
           "實驗設定：\n"
           "- 訓練集：僅良品影像（每類約 200-300 張）\n"
           "- 測試集：良品 + 多種瑕疵影像\n"
           "- 評估指標：Image-level AUROC 和 Pixel-level AUROC\n\n"
           "各方法在 MVTec AD 上的表現：\n"
           "- PatchCore: Image AUROC 99.1%, Pixel AUROC 98.1%\n"
           "- PADIM:     Image AUROC 95.3%, Pixel AUROC 97.5%\n"
           "- 自編碼器:  Image AUROC 86.0%, Pixel AUROC 91.0%\n"
           "- STFPM:     Image AUROC 95.5%, Pixel AUROC 97.0%\n\n"
           "結論：PatchCore 在大多數類別上表現最佳，但記憶庫佔用空間較大。"
           "PADIM 在速度與精度之間取得良好平衡。")

    p.tip(
        "小樣本策略：\n"
        "1. 優先使用無監督方法（PatchCore/PADIM），僅需良品樣本\n"
        "2. 使用預訓練模型的特徵提取器，避免從頭訓練\n"
        "3. 積極使用資料增強，至少擴充 5-10 倍\n"
        "4. 考慮合成瑕疵：在良品上人工添加模擬瑕疵（貼片、變色、畸變）\n"
        "5. 使用 few-shot learning 或 meta-learning 技術\n"
        "6. 先用簡單模型驗證可行性，確認後再使用複雜模型"
    )

    p.warn(
        "深度學習的陷阱：\n"
        "1. 過擬合：訓練集表現好但測試集差，特別是在小資料集上\n"
        "2. 資料偏差：訓練集與實際產線影像分佈不同（照明、角度、批次差異）\n"
        "3. 黑箱問題：模型預測難以解釋，品質人員難以信任\n"
        "4. 部署差距：開發環境的精度無法在產線環境重現\n"
        "5. 維護成本：模型需要定期重新訓練以適應產品或環境變化\n"
        "6. 不要迷信深度學習——簡單場景用傳統方法更穩定、更快、更易維護"
    )


def write_ch21(p):
    p.ch_title("進階分割方法")
    p.txt(
        "影像分割是將影像分成多個有意義區域的過程。在瑕疵檢測中，精確的分割是後續分析的基礎。"
        "除了第 2 章介紹的閾值分割外，本章涵蓋更進階的分割技術：分水嶺、GrabCut、超像素、區域生長，"
        "以及深度學習語義分割在工業中的應用。這些方法在處理複雜場景（重疊物體、漸變邊界、不均勻背景）時，"
        "表現優於簡單閾值方法。"
    )

    # 21.1
    p.sec_title("分水嶺分割")
    p.txt(
        "分水嶺演算法將灰度影像視為地形圖（像素值=海拔高度）。從標記的「種子點」開始注水，"
        "水位上升直到不同盆地的水相遇，相遇處即為分割邊界。\n\n"
        "關鍵步驟：\n"
        "1. 計算距離變換（前景像素到最近背景像素的距離）\n"
        "2. 在距離圖上尋找局部極大值作為標記（種子）\n"
        "3. 執行 watershed 分割\n\n"
        "常見問題：過分割（標記太多）。解決方法：合併小區域或使用更嚴格的標記生成。"
    )

    p.formula("距離變換",
              "D(x,y) = min{ dist((x,y), (i,j)) :\n"
              "              I(i,j) = 背景 }\n\n"
              "  歐式距離: dist = sqrt(dx^2 + dy^2)\n"
              "  棋盤距離: dist = max(|dx|, |dy|)\n"
              "  城市距離: dist = |dx| + |dy|",
              "距離變換後，物體中心的距離值最大。對距離圖設閾值可分離接觸的物體。")

    p.formula("標記生成",
              "markers = label(D > alpha * max(D))\n\n"
              "  alpha = 0.5 ~ 0.7（控制標記的嚴格程度）\n"
              "  alpha 越大 -> 標記越少 -> 欠分割\n"
              "  alpha 越小 -> 標記越多 -> 過分割",
              "標記的品質直接決定分水嶺分割的效果。可結合形態學腐蝕生成更穩健的標記。")

    p.code("import cv2\nimport numpy as np\n\n"
           "def watershed_segment(gray, binary):\n"
           '    """完整的 watershed 分割流程\n'
           "    gray:   灰度影像\n"
           '    binary: 二值前景遮罩"""\n'
           "    # 1. 形態學開運算去噪\n"
           "    kernel = cv2.getStructuringElement(\n"
           "        cv2.MORPH_ELLIPSE, (3,3))\n"
           "    opening = cv2.morphologyEx(\n"
           "        binary, cv2.MORPH_OPEN, kernel,\n"
           "        iterations=2)\n\n"
           "    # 2. 確定背景區域（膨脹）\n"
           "    sure_bg = cv2.dilate(\n"
           "        opening, kernel, iterations=3)\n\n"
           "    # 3. 距離變換 -> 確定前景\n"
           "    dist = cv2.distanceTransform(\n"
           "        opening, cv2.DIST_L2, 5)\n"
           "    _, sure_fg = cv2.threshold(\n"
           "        dist, 0.5 * dist.max(), 255, 0)\n"
           "    sure_fg = sure_fg.astype(np.uint8)\n\n"
           "    # 4. 未知區域\n"
           "    unknown = cv2.subtract(sure_bg, sure_fg)\n\n"
           "    # 5. 標記連通域\n"
           "    n_labels, markers = \\\n"
           "        cv2.connectedComponents(sure_fg)\n"
           "    markers = markers + 1\n"
           "    markers[unknown == 255] = 0\n\n"
           "    # 6. 執行 watershed\n"
           "    img_c = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)\n"
           "    markers = cv2.watershed(img_c, markers)\n\n"
           "    # markers == -1 為邊界\n"
           "    return markers, n_labels")

    # 21.2
    p.sec_title("GrabCut 互動分割")
    p.txt(
        "GrabCut 是一種半自動分割方法，使用高斯混合模型（GMM）分別對前景和背景建模，"
        "並通過圖切割（Graph Cut）優化進行分割。使用者僅需提供一個包含前景的矩形框或初始遮罩。\n\n"
        "原理：\n"
        "1. 使用者指定前景矩形框\n"
        "2. 框外像素初始化為背景，框內為可能的前景\n"
        "3. 用 GMM（通常 K=5）分別建模前景/背景的顏色分佈\n"
        "4. 構建圖模型，邊權重由顏色相似度和 GMM 分配決定\n"
        "5. 最小割/最大流求解最優分割\n"
        "6. 可迭代優化以改善結果"
    )

    p.code("import cv2\nimport numpy as np\n\n"
           "def grabcut_segment(img, rect=None,\n"
           "                     mask_hint=None,\n"
           "                     n_iter=5):\n"
           '    """GrabCut 分割\n'
           "    img:       BGR 彩色影像\n"
           "    rect:      前景矩形 (x,y,w,h)\n"
           '    mask_hint: 使用者提供的初始遮罩"""\n'
           "    mask = np.zeros(\n"
           "        img.shape[:2], dtype=np.uint8)\n"
           "    bgd = np.zeros((1,65), np.float64)\n"
           "    fgd = np.zeros((1,65), np.float64)\n\n"
           "    if rect is not None:\n"
           "        mode = cv2.GC_INIT_WITH_RECT\n"
           "        cv2.grabCut(img, mask, rect,\n"
           "                    bgd, fgd, n_iter, mode)\n"
           "    elif mask_hint is not None:\n"
           "        mask = mask_hint.copy()\n"
           "        mode = cv2.GC_INIT_WITH_MASK\n"
           "        cv2.grabCut(img, mask, None,\n"
           "                    bgd, fgd, n_iter, mode)\n\n"
           "    # 提取前景\n"
           "    fg_mask = np.where(\n"
           "        (mask == cv2.GC_FGD) |\n"
           "        (mask == cv2.GC_PR_FGD), 1, 0\n"
           "    ).astype(np.uint8)\n\n"
           "    result = img * fg_mask[:,:,np.newaxis]\n"
           "    return fg_mask, result\n\n"
           "# 使用範例\n"
           "img = cv2.imread('part.jpg')\n"
           "rect = (50, 50, 400, 300)  # 前景矩形\n"
           "mask, result = grabcut_segment(img, rect=rect)")

    # 21.3
    p.sec_title("超像素分割：SLIC")
    p.txt(
        "SLIC（Simple Linear Iterative Clustering）將影像分割為多個緊湊且均勻的超像素區塊。"
        "每個超像素包含一群相似的相鄰像素。超像素作為後續分析的基本單位，取代單個像素，"
        "可大幅減少計算量並保留邊界資訊。"
    )

    p.formula("SLIC 距離度量",
              "d_c = sqrt((l_i-l_j)^2 + (a_i-a_j)^2 +\n"
              "           (b_i-b_j)^2)\n\n"
              "d_s = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)\n\n"
              "D = sqrt(d_c^2 + (d_s/S)^2 * m^2)\n\n"
              "  d_c = CIELAB 色彩距離\n"
              "  d_s = 空間距離\n"
              "  S   = 超像素間距 sqrt(N/k)\n"
              "  m   = 緊湊度因子 (通常 10~40)",
              "m 越大，超像素越規則方形；m 越小，超像素越貼合邊緣。")

    p.code("import cv2\nimport numpy as np\n\n"
           "def slic_segment(img, n_segments=200,\n"
           "                  compactness=10):\n"
           '    """SLIC 超像素分割 + 後處理"""\n'
           "    # 建立 SLIC 物件\n"
           "    slic = cv2.ximgproc.createSuperpixelSLIC(\n"
           "        img, algorithm=cv2.ximgproc.SLIC,\n"
           "        region_size=int(\n"
           "            np.sqrt(img.shape[0]*img.shape[1]\n"
           "                    / n_segments)),\n"
           "        ruler=compactness)\n\n"
           "    # 迭代\n"
           "    slic.iterate(10)\n"
           "    slic.enforceLabelConnectivity(25)\n\n"
           "    # 取得標籤與邊界\n"
           "    labels = slic.getLabels()\n"
           "    mask = slic.getLabelContourMask()\n\n"
           "    # 計算每個超像素的特徵\n"
           "    n = slic.getNumberOfSuperpixels()\n"
           "    features = []\n"
           "    for i in range(n):\n"
           "        region = img[labels == i]\n"
           "        mean_color = region.mean(axis=0)\n"
           "        std_color = region.std(axis=0)\n"
           "        features.append({\n"
           "            'label': i,\n"
           "            'mean': mean_color,\n"
           "            'std': std_color,\n"
           "            'size': len(region)\n"
           "        })\n\n"
           "    return labels, mask, features")

    # 21.4
    p.sec_title("基於區域的分割：區域生長")
    p.txt(
        "區域生長從種子點開始，將相鄰像素與種子比較，如果相似度在閾值內則併入區域，"
        "不斷擴展直到沒有可合併的像素為止。適合分割具有均勻特性的區域。"
    )

    p.code("import cv2\nimport numpy as np\nfrom collections import deque\n\n"
           "def region_growing(gray, seed, threshold=15):\n"
           '    """區域生長分割\n'
           "    gray:      灰度影像\n"
           "    seed:      種子點 (x, y)\n"
           '    threshold: 相似度閾值"""\n'
           "    h, w = gray.shape\n"
           "    visited = np.zeros((h, w), dtype=bool)\n"
           "    mask = np.zeros((h, w), dtype=np.uint8)\n\n"
           "    queue = deque([seed])\n"
           "    seed_val = float(gray[seed[1], seed[0]])\n\n"
           "    while queue:\n"
           "        x, y = queue.popleft()\n"
           "        if (x < 0 or x >= w or y < 0 or y >= h\n"
           "                or visited[y, x]):\n"
           "            continue\n\n"
           "        visited[y, x] = True\n"
           "        diff = abs(float(gray[y,x]) - seed_val)\n\n"
           "        if diff <= threshold:\n"
           "            mask[y, x] = 255\n"
           "            # 8 鄰域擴展\n"
           "            for dx in [-1, 0, 1]:\n"
           "                for dy in [-1, 0, 1]:\n"
           "                    if dx == 0 and dy == 0:\n"
           "                        continue\n"
           "                    queue.append(\n"
           "                        (x+dx, y+dy))\n\n"
           "    return mask")

    # 21.5
    p.sec_title("語義分割在工業中的應用")
    p.txt(
        "深度學習語義分割（如 DeepLabV3+）為每個像素分配類別標籤，"
        "實現瑕疵的精確像素級定位和分類。在多種瑕疵共存的場景中尤其有用。\n\n"
        "DeepLabV3+ 的核心技術：\n"
        "- 空洞卷積（Atrous Convolution）：擴大感受野而不降低解析度\n"
        "- ASPP（Atrous Spatial Pyramid Pooling）：多尺度特徵融合\n"
        "- 編碼器-解碼器結構：兼顧語義與空間細節\n\n"
        "工業標註策略：\n"
        "- 背景（類別 0）、瑕疵類型 A（類別 1）、瑕疵類型 B（類別 2）...\n"
        "- 使用 CVAT 或 LabelMe 等工具進行多邊形標註\n"
        "- 標註時注意瑕疵邊界的精確性"
    )

    p.table(
        ["分割方法", "速度", "精度", "人工介入"],
        [
            ["閾值分割", "極快 (<1ms)", "低~中", "設定閾值"],
            ["分水嶺", "快 (~5ms)", "中", "標記生成"],
            ["GrabCut", "慢 (~100ms)", "中~高", "初始矩形/遮罩"],
            ["SLIC 超像素", "中 (~20ms)", "中", "參數調整"],
            ["區域生長", "中 (~30ms)", "中", "種子點選擇"],
            ["U-Net", "慢 (~50ms GPU)", "高", "標註訓練資料"],
            ["DeepLabV3+", "慢 (~80ms GPU)", "極高", "標註訓練資料"],
        ]
    )

    p.case("重疊零件分割",
           "場景：傳送帶上多個金屬墊圈重疊堆放，需要分離計數。\n\n"
           "挑戰：簡單閾值無法分離接觸/重疊的零件。\n\n"
           "解決方案：分水嶺分割\n"
           "1. 前處理：CLAHE 增強 + 中值濾波\n"
           "2. Otsu 閾值分割出前景\n"
           "3. 距離變換 -> 找到每個零件的中心\n"
           "4. 距離圖設閾值（0.6 * max）生成種子標記\n"
           "5. 分水嶺分割分離重疊區域\n"
           "6. 對每個分離的區域進行特徵分析\n\n"
           "結果：成功分離 95% 的重疊零件。失敗案例多為嚴重重疊（面積覆蓋 > 50%）的零件。\n"
           "改善：對失敗案例可結合深度學習實例分割（Mask R-CNN）。")

    p.tip(
        "分割方法選擇指南：\n"
        "1. 背景均勻、瑕疵對比明顯 -> 閾值分割（最快最簡單）\n"
        "2. 背景不均勻但前景明確 -> 自適應閾值\n"
        "3. 物體接觸/重疊需分離 -> 分水嶺分割\n"
        "4. 需要互動式精確分割 -> GrabCut\n"
        "5. 需要區域級特徵分析 -> SLIC 超像素\n"
        "6. 均勻區域的精確提取 -> 區域生長\n"
        "7. 多類瑕疵的像素級分類 -> 深度學習語義分割\n"
        "原則：從最簡單的方法開始，僅在效果不足時升級。"
    )


def write_ch22(p):
    p.ch_title("即時處理與效能優化")
    p.txt(
        "工業產線要求檢測系統在嚴格的節拍時間（takt time）內完成每件產品的檢測。"
        "典型要求為每秒 5-30 件，意味著每件的處理時間僅 33-200 毫秒。"
        "本章探討如何分析效能瓶頸、應用加速技術、設計管線架構和利用 GPU，"
        "以構建滿足即時要求的檢測系統。"
    )

    # 22.1
    p.sec_title("效能瓶頸分析")
    p.txt(
        "優化的第一步是找出瓶頸。典型瑕疵檢測流程中，各步驟的耗時差異很大。\n"
        "常見瓶頸（按耗時排序）：\n"
        "1. 深度學習推論（50-500 ms）\n"
        "2. 影像擷取與傳輸（10-50 ms）\n"
        "3. 複雜形態學運算（5-20 ms）\n"
        "4. 頻率域濾波（5-15 ms）\n"
        "5. 模板匹配（3-10 ms）\n"
        "6. 閾值分割（< 1 ms）"
    )

    p.code("import time\nimport cProfile\nimport cv2\nimport numpy as np\n\n"
           "def profile_pipeline(img):\n"
           '    """分析各步驟耗時"""\n'
           "    timings = {}\n\n"
           "    t0 = time.perf_counter()\n"
           "    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n"
           "    timings['grayscale'] = (\n"
           "        time.perf_counter() - t0) * 1000\n\n"
           "    t0 = time.perf_counter()\n"
           "    blur = cv2.GaussianBlur(gray, (5,5), 1.0)\n"
           "    timings['blur'] = (\n"
           "        time.perf_counter() - t0) * 1000\n\n"
           "    t0 = time.perf_counter()\n"
           "    _, binary = cv2.threshold(\n"
           "        blur, 0, 255,\n"
           "        cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n"
           "    timings['threshold'] = (\n"
           "        time.perf_counter() - t0) * 1000\n\n"
           "    t0 = time.perf_counter()\n"
           "    k = cv2.getStructuringElement(\n"
           "        cv2.MORPH_ELLIPSE, (5,5))\n"
           "    clean = cv2.morphologyEx(\n"
           "        binary, cv2.MORPH_OPEN, k)\n"
           "    timings['morphology'] = (\n"
           "        time.perf_counter() - t0) * 1000\n\n"
           "    t0 = time.perf_counter()\n"
           "    cnts, _ = cv2.findContours(\n"
           "        clean, cv2.RETR_EXTERNAL,\n"
           "        cv2.CHAIN_APPROX_SIMPLE)\n"
           "    timings['contours'] = (\n"
           "        time.perf_counter() - t0) * 1000\n\n"
           "    # 列印結果\n"
           "    total = sum(timings.values())\n"
           "    for step, ms in timings.items():\n"
           "        pct = ms / total * 100\n"
           "        print(f'{step:15s}: {ms:6.2f} ms '\n"
           "              f'({pct:5.1f}%)')\n"
           "    print(f'{\"TOTAL\":15s}: {total:6.2f} ms')\n"
           "    return timings")

    # 22.2
    p.sec_title("影像處理加速技術")

    p.sub_sec("ROI 縮小搜索範圍")
    p.txt(
        "如果瑕疵只可能出現在影像的特定區域，僅處理該 ROI 可大幅減少計算量。"
        "例如：僅檢測零件表面區域，跳過背景。"
    )

    p.sub_sec("影像降取樣（多解析度金字塔）")
    p.txt(
        "先在低解析度影像上快速粗篩，找到可疑區域後再在原始解析度上精細檢測。"
        "降取樣 2x 後，處理速度提升約 4 倍。"
    )

    p.sub_sec("查找表（LUT）加速")
    p.txt(
        "預計算灰度映射表，避免重複的逐像素計算。特別適合自定義的亮度變換。"
    )

    p.code("import cv2\nimport numpy as np\n\n"
           "# --- ROI 處理 ---\n"
           "def process_roi(img, roi):\n"
           "    x, y, w, h = roi\n"
           "    patch = img[y:y+h, x:x+w]\n"
           "    # 僅處理 ROI 區域\n"
           "    result = cv2.GaussianBlur(patch, (5,5), 1)\n"
           "    return result\n\n"
           "# --- 多解析度金字塔 ---\n"
           "def pyramid_detect(img, scale=0.5):\n"
           "    small = cv2.resize(\n"
           "        img, None, fx=scale, fy=scale)\n"
           "    # 粗篩：快速檢測\n"
           "    _, mask_s = cv2.threshold(\n"
           "        small, 0, 255,\n"
           "        cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n"
           "    cnts, _ = cv2.findContours(\n"
           "        mask_s, cv2.RETR_EXTERNAL,\n"
           "        cv2.CHAIN_APPROX_SIMPLE)\n"
           "    # 有可疑區域時才精細檢測\n"
           "    if len(cnts) > 0:\n"
           "        return full_inspect(img)  # 原解析度\n"
           "    return None  # 快速通過\n\n"
           "# --- LUT 加速 ---\n"
           "def apply_lut_gamma(img, gamma=1.5):\n"
           "    lut = np.array([\n"
           "        np.clip(pow(i/255.0, gamma)*255, 0, 255)\n"
           "        for i in range(256)\n"
           "    ], dtype=np.uint8)\n"
           "    return cv2.LUT(img, lut)")

    # 22.3
    p.sec_title("多執行緒與管線化")
    p.txt(
        "瑕疵檢測流程可分解為三個獨立階段：影像擷取、影像處理、結果輸出。"
        "使用多執行緒管線（pipeline），三個階段並行運作，"
        "系統吞吐量由最慢的階段決定，而非三者之和。"
    )

    p.code("import threading\nimport queue\nimport time\nimport cv2\n\n"
           "class InspectionPipeline:\n"
           '    """三級管線：擷取 -> 處理 -> 輸出"""\n\n'
           "    def __init__(self, max_queue=5):\n"
           "        self.q_capture = queue.Queue(max_queue)\n"
           "        self.q_result = queue.Queue(max_queue)\n"
           "        self.running = True\n\n"
           "    def capture_worker(self, camera):\n"
           '        """擷取執行緒"""\n'
           "        while self.running:\n"
           "            ret, frame = camera.read()\n"
           "            if ret:\n"
           "                ts = time.perf_counter()\n"
           "                self.q_capture.put(\n"
           "                    (frame, ts))\n\n"
           "    def process_worker(self, inspect_fn):\n"
           '        """處理執行緒"""\n'
           "        while self.running:\n"
           "            try:\n"
           "                frame, ts = \\\n"
           "                    self.q_capture.get(\n"
           "                        timeout=1)\n"
           "            except queue.Empty:\n"
           "                continue\n"
           "            result = inspect_fn(frame)\n"
           "            self.q_result.put(\n"
           "                (frame, result, ts))\n\n"
           "    def output_worker(self, output_fn):\n"
           '        """輸出執行緒"""\n'
           "        while self.running:\n"
           "            try:\n"
           "                frame, result, ts = \\\n"
           "                    self.q_result.get(\n"
           "                        timeout=1)\n"
           "            except queue.Empty:\n"
           "                continue\n"
           "            latency = (\n"
           "                time.perf_counter()-ts)*1000\n"
           "            output_fn(frame, result, latency)\n\n"
           "    def start(self, camera, inspect_fn,\n"
           "              output_fn):\n"
           "        threads = [\n"
           "            threading.Thread(\n"
           "                target=self.capture_worker,\n"
           "                args=(camera,)),\n"
           "            threading.Thread(\n"
           "                target=self.process_worker,\n"
           "                args=(inspect_fn,)),\n"
           "            threading.Thread(\n"
           "                target=self.output_worker,\n"
           "                args=(output_fn,)),\n"
           "        ]\n"
           "        for t in threads:\n"
           "            t.daemon = True\n"
           "            t.start()\n"
           "        return threads")

    # 22.4
    p.sec_title("GPU 加速")
    p.txt(
        "GPU 並行計算能力可大幅加速影像處理和深度學習推論。\n\n"
        "兩種方式：\n"
        "1. OpenCV CUDA 模組：加速傳統影像處理（濾波、形態學、模板匹配）\n"
        "2. TensorRT / ONNX Runtime：加速深度學習推論（通常 3-10 倍加速）"
    )

    p.code("import cv2\nimport numpy as np\n\n"
           "# --- OpenCV CUDA 加速 ---\n"
           "def gpu_process(img):\n"
           "    # 上傳到 GPU\n"
           "    gpu_img = cv2.cuda_GpuMat()\n"
           "    gpu_img.upload(img)\n\n"
           "    # GPU 上執行高斯模糊\n"
           "    gpu_blur = cv2.cuda.createGaussianFilter(\n"
           "        cv2.CV_8UC1, cv2.CV_8UC1, (5,5), 1.0)\n"
           "    gpu_result = gpu_blur.apply(gpu_img)\n\n"
           "    # 下載回 CPU\n"
           "    result = gpu_result.download()\n"
           "    return result\n\n"
           "# --- TensorRT 推論 ---\n"
           "# 先將 PyTorch 模型轉為 ONNX -> TensorRT\n"
           "# torch.onnx.export(model, dummy, 'model.onnx')\n"
           "# trtexec --onnx=model.onnx --saveEngine=m.trt\n\n"
           "# 使用 ONNX Runtime (更簡單)\n"
           "import onnxruntime as ort\n\n"
           "def onnx_infer(onnx_path, img):\n"
           "    sess = ort.InferenceSession(\n"
           "        onnx_path,\n"
           "        providers=['CUDAExecutionProvider',\n"
           "                   'CPUExecutionProvider'])\n"
           "    blob = cv2.dnn.blobFromImage(\n"
           "        img, 1/255.0, (640,640),\n"
           "        swapRB=True)\n"
           "    outputs = sess.run(None, {\n"
           "        sess.get_inputs()[0].name: blob})\n"
           "    return outputs")

    # 22.5
    p.sec_title("最佳化演算法選擇")
    p.txt(
        "不同演算法的時間複雜度差異巨大。根據即時性要求選擇合適的演算法至關重要。"
    )

    p.table(
        ["演算法", "時間複雜度", "1MP 耗時", "適用場景"],
        [
            ["閾值分割", "O(N)", "< 1 ms", "快速粗篩"],
            ["高斯模糊", "O(N*k^2)", "~2 ms", "預處理"],
            ["形態學", "O(N*k^2)", "~3 ms", "後處理"],
            ["Canny", "O(N)", "~3 ms", "邊緣檢測"],
            ["連通域", "O(N)", "~2 ms", "Blob 分析"],
            ["模板匹配", "O(N*M)", "~10 ms", "定位/比對"],
            ["FFT", "O(N*logN)", "~8 ms", "頻率域"],
            ["GrabCut", "O(N*k*iter)", "~100 ms", "精確分割"],
            ["CNN 推論", "依模型", "30-200 ms", "深度學習"],
        ]
    )

    # 22.6
    p.sec_title("即時系統設計模式")
    p.txt(
        "根據觸發方式，即時檢測系統分為兩種模式：\n"
        "1. 觸發模式：感測器偵測到零件到位時觸發拍照和檢測。適合離散零件。\n"
        "2. 連續模式：相機持續擷取，軟體連續處理。適合連續卷材或輸送帶。"
    )

    p.formula("系統吞吐量",
              "throughput = 1 / max(t_capture, t_process,\n"
              "                     t_output)\n\n"
              "  t_capture = 擷取+傳輸時間\n"
              "  t_process = 處理時間\n"
              "  t_output  = 輸出+通訊時間\n\n"
              "要求: throughput >= production_rate\n\n"
              "例: 產線速度 20 件/秒\n"
              "    => max(t) <= 50 ms",
              "管線化後吞吐量由最慢階段決定。如果處理是瓶頸，考慮多處理執行緒或 GPU 加速。")

    p.case("每秒 30 件的產線即時檢測系統",
           "場景：電子連接器高速檢測，要求每秒 30 件（33 ms/件）。\n\n"
           "硬體配置：\n"
           "- 線掃描相機（4096 像素），配頻閃光源\n"
           "- 工業 PC（i7 + RTX 3060），GigE 介面\n"
           "- IO 觸發 + PLC 通訊\n\n"
           "軟體架構：\n"
           "- 三級管線（擷取 / 處理 / IO 輸出）\n"
           "- 處理流程：ROI 裁剪(1ms) -> 閾值(0.5ms) -> 形態學(1ms) -> Blob(1ms) -> 判定(0.5ms)\n"
           "- 總處理時間：~4 ms << 33 ms（有充足餘量）\n\n"
           "效能結果：\n"
           "- 實際吞吐量：35 件/秒（餘量 17%）\n"
           "- 平均延遲：12 ms（擷取 5ms + 處理 4ms + IO 3ms）\n"
           "- 穩定運行 24 小時無丟幀")

    p.warn(
        "即時系統的常見問題：\n"
        "1. 記憶體洩漏：長時間運行後記憶體逐漸增長，最終導致系統變慢或崩潰\n"
        "2. GC 暫停：Python 垃圾回收可能造成不可預期的延遲尖峰\n"
        "3. 佇列溢出：處理速度跟不上擷取速度時，佇列無限增長\n"
        "4. 溫度降頻：GPU/CPU 長時間滿載會過熱降頻，降低效能\n"
        "5. 網路延遲：GigE 相機可能因網路擁塞丟幀\n"
        "6. 未做壓力測試：開發環境的效能不代表產線的持續效能"
    )


def write_ch23(p):
    p.ch_title("系統穩健性與邊界案例")
    p.txt(
        "一個在實驗室中表現完美的檢測系統，在實際產線上可能因為環境變化、異常輸入或邊界情況而失效。"
        "系統穩健性（Robustness）是工業檢測系統的核心要求。本章探討影響穩健性的因素、"
        "如何處理異常情況、分析假陽性/假陰性的根因，以及建立系統化的穩健性測試和日誌追溯體系。"
    )

    # 23.1
    p.sec_title("環境變化的影響")
    p.txt(
        "工業環境中的多種變化會影響檢測結果：\n\n"
        "1. 溫度漂移：溫度變化導致相機增益漂移、LED 光源亮度衰減。"
        "白天與夜間的環境光差異也可能影響結果。\n"
        "2. 光源老化：LED 光源隨使用時間逐漸衰減（通常 5000 小時後亮度降至 80%）。"
        "光源的光譜特性也可能隨老化改變。\n"
        "3. 振動：設備振動導致影像模糊或位移。高速產線上特別嚴重。\n"
        "4. 灰塵與污染：鏡頭或光源表面的灰塵積累導致影像品質下降。\n"
        "5. 材料批次差異：不同批次的原材料在顏色、紋理上存在自然變異。\n\n"
        "對策：\n"
        "- 定期校正：每班次或每天使用標準參考片驗證系統\n"
        "- 自動曝光補償：監控影像亮度並自動調整曝光\n"
        "- 環境監控：記錄溫度、濕度，與檢測結果相關分析\n"
        "- 預防性維護：定期清潔鏡頭和光源，按壽命更換 LED"
    )

    # 23.2
    p.sec_title("自動曝光與白平衡補償")
    p.txt(
        "光源亮度隨時間和溫度變化，導致影像整體亮度偏移。"
        "自動曝光補償演算法可在每次拍攝後調整相機參數，維持影像亮度穩定。"
    )

    p.code("import cv2\nimport numpy as np\n\n"
           "class AutoExposure:\n"
           '    """自動曝光調整控制器"""\n\n'
           "    def __init__(self, target_mean=128,\n"
           "                 tolerance=10,\n"
           "                 roi=None):\n"
           "        self.target = target_mean\n"
           "        self.tol = tolerance\n"
           "        self.roi = roi\n"
           "        self.exposure = 10000  # us\n"
           "        self.gain = 1.0\n"
           "        self.history = []\n\n"
           "    def compute_brightness(self, img):\n"
           '        """計算 ROI 內的平均亮度"""\n'
           "        if len(img.shape) == 3:\n"
           "            gray = cv2.cvtColor(\n"
           "                img, cv2.COLOR_BGR2GRAY)\n"
           "        else:\n"
           "            gray = img\n\n"
           "        if self.roi:\n"
           "            x, y, w, h = self.roi\n"
           "            region = gray[y:y+h, x:x+w]\n"
           "        else:\n"
           "            region = gray\n\n"
           "        return float(np.mean(region))\n\n"
           "    def adjust(self, img):\n"
           '        """根據當前影像調整曝光參數\n'
           '        回傳: (需調整, 新曝光值, 新增益)"""\n'
           "        brightness = self.compute_brightness(img)\n"
           "        self.history.append(brightness)\n\n"
           "        diff = self.target - brightness\n\n"
           "        if abs(diff) <= self.tol:\n"
           "            return False, self.exposure, self.gain\n\n"
           "        # P 控制器\n"
           "        ratio = self.target / max(\n"
           "            brightness, 1)\n"
           "        ratio = np.clip(ratio, 0.8, 1.25)\n\n"
           "        # 優先調整曝光，增益不變\n"
           "        new_exp = int(self.exposure * ratio)\n"
           "        new_exp = np.clip(new_exp, 100, 100000)\n"
           "        self.exposure = int(new_exp)\n\n"
           "        return True, self.exposure, self.gain\n\n"
           "    def check_drift(self, window=100):\n"
           '        """檢測亮度是否有持續漂移趨勢"""\n'
           "        if len(self.history) < window:\n"
           "            return False\n"
           "        recent = self.history[-window:]\n"
           "        trend = np.polyfit(\n"
           "            range(window), recent, 1)[0]\n"
           "        # 斜率大於閾值表示有漂移\n"
           "        return abs(trend) > 0.1")

    # 23.3
    p.sec_title("異常輸入處理")
    p.txt(
        "檢測系統必須能處理各種異常輸入，而非直接崩潰。常見異常：\n"
        "- 空影像或全黑/全白影像（相機故障、擷取超時）\n"
        "- 過曝或欠曝影像（光源異常、曝光設定錯誤）\n"
        "- 模糊影像（對焦失敗、振動）\n"
        "- 無零件或異物（零件缺失、錯位）"
    )

    p.formula("模糊偵測：Laplacian 方差",
              "blur_score = Var(Laplacian(I))\n\n"
              "Laplacian = d^2I/dx^2 + d^2I/dy^2\n\n"
              "blur_score 高 -> 影像清晰（邊緣多）\n"
              "blur_score 低 -> 影像模糊\n\n"
              "閾值: 通常 < 100 判定為模糊\n"
              "      (需依據具體場景校準)",
              "Laplacian 是二階導數運算子，對邊緣敏感。清晰影像有大量邊緣，方差大；模糊影像邊緣弱，方差小。")

    p.code("import cv2\nimport numpy as np\n\n"
           "def check_image_quality(img):\n"
           '    """影像品質檢查，回傳 (通過, 品質報告)"""\n'
           "    report = {}\n\n"
           "    # 1. 空影像檢查\n"
           "    if img is None or img.size == 0:\n"
           "        return False, {'error': '空影像'}\n\n"
           "    gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n"
           "            if len(img.shape) == 3 else img)\n\n"
           "    # 2. 亮度檢查\n"
           "    mean_val = float(np.mean(gray))\n"
           "    report['brightness'] = mean_val\n"
           "    if mean_val < 20:\n"
           "        return False, {\n"
           "            **report, 'error': '欠曝 (過暗)'}\n"
           "    if mean_val > 240:\n"
           "        return False, {\n"
           "            **report, 'error': '過曝 (過亮)'}\n\n"
           "    # 3. 動態範圍檢查\n"
           "    min_v, max_v = int(np.min(gray)), \\\n"
           "                   int(np.max(gray))\n"
           "    dyn_range = max_v - min_v\n"
           "    report['dynamic_range'] = dyn_range\n"
           "    if dyn_range < 30:\n"
           "        return False, {\n"
           "            **report,\n"
           "            'error': '動態範圍不足'}\n\n"
           "    # 4. 模糊偵測 (Laplacian 方差)\n"
           "    lap = cv2.Laplacian(gray, cv2.CV_64F)\n"
           "    blur_score = float(lap.var())\n"
           "    report['blur_score'] = blur_score\n"
           "    if blur_score < 100:\n"
           "        return False, {\n"
           "            **report, 'error': '影像模糊'}\n\n"
           "    # 5. 零件存在性檢查\n"
           "    _, binary = cv2.threshold(\n"
           "        gray, 0, 255,\n"
           "        cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n"
           "    fg_ratio = np.count_nonzero(binary) \\\n"
           "               / binary.size\n"
           "    report['foreground_ratio'] = fg_ratio\n"
           "    if fg_ratio < 0.05 or fg_ratio > 0.95:\n"
           "        return False, {\n"
           "            **report,\n"
           "            'error': '零件可能缺失'}\n\n"
           "    report['status'] = 'OK'\n"
           "    return True, report")

    # 23.4
    p.sec_title("假陽性與假陰性分析")
    p.txt(
        "假陽性（False Positive）：良品被判為瑕疵，導致過度報廢和返工成本增加。\n"
        "假陰性（False Negative）：瑕疵品未被檢出而流出，導致客訴和品質問題。\n\n"
        "在工業檢測中，假陰性（漏檢）通常比假陽性（誤檢）更嚴重。"
        "系統設計時需要找到兩者之間的最佳平衡點。"
    )

    p.table(
        ["類型", "常見原因", "對策"],
        [
            ["假陽性", "閾值太嚴格", "放寬閾值，加入特徵過濾"],
            ["假陽性", "光源反光/陰影", "改善打光，遮罩已知反光區"],
            ["假陽性", "材料自然紋理", "使用變異模型學習紋理"],
            ["假陽性", "對齊偏差", "改善對齊精度或加大容差"],
            ["假陰性", "閾值太寬鬆", "收緊閾值，多特徵組合"],
            ["假陰性", "瑕疵對比度低", "改善照明或使用光度立體"],
            ["假陰性", "瑕疵太小", "提高解析度或使用微距鏡頭"],
            ["假陰性", "訓練資料不足", "增加訓練樣本和資料增強"],
        ]
    )

    # 23.5
    p.sec_title("穩健性測試方法")
    p.txt(
        "Gauge R&R（量測系統分析）是評估檢測系統穩健性的標準方法。"
        "它量化系統的重複性（同一條件重複檢測的一致性）和再現性（不同條件下的一致性）。"
    )

    p.code("import numpy as np\n\n"
           "def gauge_rr(measurements):\n"
           '    """Gauge R&R 量測系統分析\n'
           "    measurements: shape (n_operators,\n"
           "                        n_parts,\n"
           '                        n_repeats)"""\n'
           "    n_op, n_parts, n_rep = measurements.shape\n\n"
           "    # 重複性 (EV - Equipment Variation)\n"
           "    # 同一操作員/同一零件的變異\n"
           "    ranges = []\n"
           "    for op in range(n_op):\n"
           "        for part in range(n_parts):\n"
           "            vals = measurements[op, part, :]\n"
           "            ranges.append(\n"
           "                np.max(vals) - np.min(vals))\n"
           "    r_bar = np.mean(ranges)\n"
           "    # d2 常數 (n_rep=2: 1.128, 3: 1.693)\n"
           "    d2 = {2: 1.128, 3: 1.693, 4: 2.059}\n"
           "    ev = r_bar / d2.get(n_rep, 1.693)\n\n"
           "    # 再現性 (AV - Appraiser Variation)\n"
           "    # 不同操作員之間的變異\n"
           "    op_means = measurements.mean(\n"
           "        axis=(1, 2))\n"
           "    x_diff = np.max(op_means) - \\\n"
           "             np.min(op_means)\n"
           "    d2_op = d2.get(n_op, 1.693)\n"
           "    av_sq = max(0,\n"
           "        (x_diff/d2_op)**2 - ev**2/(n_parts*n_rep))\n"
           "    av = np.sqrt(av_sq)\n\n"
           "    # 總 R&R\n"
           "    grr = np.sqrt(ev**2 + av**2)\n\n"
           "    # 零件變異 (PV)\n"
           "    part_means = measurements.mean(\n"
           "        axis=(0, 2))\n"
           "    pv = np.std(part_means, ddof=1)\n\n"
           "    # 總變異\n"
           "    tv = np.sqrt(grr**2 + pv**2)\n\n"
           "    # %GRR\n"
           "    pct_grr = (grr / tv * 100\n"
           "               if tv > 0 else 0)\n\n"
           "    return {\n"
           "        'EV': ev, 'AV': av,\n"
           "        'GRR': grr, 'PV': pv, 'TV': tv,\n"
           "        '%GRR': pct_grr,\n"
           "        'status': (\n"
           "            'PASS' if pct_grr < 10\n"
           "            else 'MARGINAL' if pct_grr < 30\n"
           "            else 'FAIL')\n"
           "    }")

    # 23.6
    p.sec_title("日誌記錄與追溯")
    p.txt(
        "完整的日誌系統是產線問題排查的基礎。每次檢測都應記錄：\n"
        "- 時間戳、序列號/批次號\n"
        "- 檢測結果、瑕疵類型、分數\n"
        "- 環境參數（曝光、光源電流、溫度）\n"
        "- 影像品質指標\n"
        "- 異常事件\n\n"
        "日誌應支援快速查詢和統計分析。"
    )

    p.code("import json\nimport logging\nfrom datetime import datetime\nfrom pathlib import Path\n\n"
           "class InspectionLogger:\n"
           '    """結構化檢測日誌系統"""\n\n'
           "    def __init__(self, log_dir, batch_id):\n"
           "        self.log_dir = Path(log_dir)\n"
           "        self.log_dir.mkdir(\n"
           "            parents=True, exist_ok=True)\n"
           "        self.batch_id = batch_id\n\n"
           "        # 文字日誌\n"
           "        self.logger = logging.getLogger(\n"
           "            'inspection')\n"
           "        self.logger.setLevel(logging.INFO)\n"
           "        fh = logging.FileHandler(\n"
           "            self.log_dir / f'{batch_id}.log',\n"
           "            encoding='utf-8')\n"
           "        fh.setFormatter(logging.Formatter(\n"
           "            '%(asctime)s | %(levelname)s '\n"
           "            '| %(message)s'))\n"
           "        self.logger.addHandler(fh)\n\n"
           "        # JSON 結果記錄\n"
           "        self.results = []\n\n"
           "    def log_result(self, serial, grade,\n"
           "                    score, defects,\n"
           "                    quality_info=None):\n"
           '        """記錄單件檢測結果"""\n'
           "        record = {\n"
           "            'timestamp': datetime.now()\n"
           "                .isoformat(),\n"
           "            'batch': self.batch_id,\n"
           "            'serial': serial,\n"
           "            'grade': grade,\n"
           "            'score': round(score, 2),\n"
           "            'defect_count': len(defects),\n"
           "            'defects': defects,\n"
           "        }\n"
           "        if quality_info:\n"
           "            record['quality'] = quality_info\n\n"
           "        self.results.append(record)\n"
           "        self.logger.info(\n"
           "            f'{serial} | {grade} '\n"
           "            f'| score={score:.1f} '\n"
           "            f'| defects={len(defects)}')\n\n"
           "    def log_alert(self, message):\n"
           '        """記錄警報"""\n'
           "        self.logger.warning(\n"
           "            f'ALERT: {message}')\n\n"
           "    def save_summary(self):\n"
           '        """保存批次彙總"""\n'
           "        summary_path = (\n"
           "            self.log_dir /\n"
           "            f'{self.batch_id}_summary.json')\n"
           "        with open(summary_path, 'w',\n"
           "                  encoding='utf-8') as f:\n"
           "            json.dump({\n"
           "                'batch': self.batch_id,\n"
           "                'total': len(self.results),\n"
           "                'results': self.results\n"
           "            }, f, ensure_ascii=False,\n"
           "               indent=2)")

    p.case("產線環境變化導致的檢測失效排查",
           "問題描述：某金屬零件檢測系統在連續運行 6 小時後，假陽性率從 2% 急升到 15%。\n\n"
           "排查過程：\n"
           "1. 查看日誌：假陽性集中在 14:00 後出現\n"
           "2. 查看亮度趨勢：平均亮度從 135 緩降至 118\n"
           "3. 分析原因：LED 光源隨溫度升高（車間無空調）亮度衰減\n"
           "4. 驗證：在固定閾值下，亮度降低導致更多像素被判定為異常\n\n"
           "解決方案：\n"
           "1. 短期：加入自動曝光補償（AutoExposure 類）\n"
           "2. 短期：改用自適應閾值取代固定閾值\n"
           "3. 長期：安裝恆溫光源控制器\n"
           "4. 長期：每 2 小時用參考片自動校正\n\n"
           "效果：假陽性率穩定在 1.5% 以下，全天波動 < 0.5%。")

    p.tip(
        "系統穩健性檢查清單：\n"
        "1. 連續運行 24 小時測試，監控假陽性/假陰性率是否漂移\n"
        "2. 使用參考片（已知良品和已知瑕疵品）定期驗證\n"
        "3. 記錄環境參數（溫度、濕度）並與檢測結果交叉分析\n"
        "4. 對閾值做 +/- 10% 的敏感度測試\n"
        "5. 模擬異常輸入（過曝、模糊、空影像）確認系統不會崩潰\n"
        "6. 記錄每個被誤判的案例，定期分析根因並優化\n"
        "7. 監控影像品質指標，設定預警閾值"
    )

    p.warn(
        "不要過度依賴單一條件：\n"
        "1. 不要僅靠一個閾值做判定——組合多個特徵更穩健\n"
        "2. 不要假設光源永遠穩定——必須有亮度監控\n"
        "3. 不要忽略罕見案例——邊界情況在大量生產中必然發生\n"
        "4. 不要假設新批次材料與舊批次相同——留足適應餘量\n"
        "5. 不要在沒有日誌的情況下部署——問題排查將極其困難"
    )


def write_ch24(p):
    p.ch_title("3D 檢測技術")
    p.txt(
        "傳統 2D 影像檢測無法獲取高度和深度資訊。對於翹曲、凹陷、毛邊高度等三維特徵的瑕疵，"
        "2D 方法力不從心。3D 檢測技術通過結構光、雷射三角測量或立體視覺獲取物體的三維形貌，"
        "實現高度測量和三維表面分析。本章介紹主流的 3D 檢測原理、實作方法和應用場景。"
    )

    # 24.1
    p.sec_title("為什麼需要 3D 檢測")
    p.txt(
        "2D 影像是三維場景的投影，丟失了深度維度。以下瑕疵在 2D 影像中難以或無法檢測：\n"
        "- 翹曲/彎曲：平面零件的微小翹曲在 2D 正視圖中不可見\n"
        "- 凹陷/凸起：淺凹陷在均勻照明下對比度極低\n"
        "- 毛邊高度：毛邊存在但高度是否超標需要 3D 量測\n"
        "- 焊球高度：BGA 焊球的共面性需要精確的 3D 資料\n"
        "- 密封面平面度：密封溝槽的平面度影響密封性能\n"
        "- 螺紋深度：螺紋是否完整加工需要深度測量\n\n"
        "3D 檢測的代價是更高的硬體成本和較慢的速度。僅在 2D 方法不足時使用。"
    )

    # 24.2
    p.sec_title("結構光三維重建")
    p.txt(
        "結構光方法將已知的光柵圖案（條紋、棋盤格或灰碼）投射到物體表面。"
        "圖案因物體表面的高度變化而產生形變。相機從另一個角度觀察被形變的圖案，"
        "通過三角測量原理計算每個點的三維座標。\n\n"
        "常見編碼方式：\n"
        "- 二進制碼（最簡單，需要 N 張圖案編碼 2^N 條紋）\n"
        "- 灰碼（Gray Code）：相鄰碼只差 1 位，減少邊界誤差\n"
        "- 相移法（Phase Shifting）：高精度但需要更多圖案\n"
        "- 混合方法：灰碼 + 相移，兼顧範圍和精度"
    )

    p.formula("三角測量公式",
              "Z = f * B / (x_L - x_R)\n\n"
              "  Z   = 物體深度 (距基線的距離)\n"
              "  f   = 焦距\n"
              "  B   = 基線距離 (投影儀與相機間距)\n"
              "  x_L = 投射點在投影儀座標的 x 位置\n"
              "  x_R = 觀測點在相機座標的 x 位置\n\n"
              "精度取決於: B (越大越好), f, 和像素解析度",
              "基線越長精度越高，但陰影區域也會增加。實際系統需要在精度和覆蓋率之間取捨。")

    p.code("import cv2\nimport numpy as np\n\n"
           "def decode_gray_code(images, threshold=30):\n"
           '    """灰碼結構光解碼\n'
           "    images: list, 每張為 (正碼, 反碼) 元組\n"
           '    threshold: 二值化閾值"""\n'
           "    n_bits = len(images)\n"
           "    h, w = images[0][0].shape[:2]\n"
           "    gray_code = np.zeros((h, w),\n"
           "                         dtype=np.uint32)\n\n"
           "    for bit_idx, (pos, neg) in \\\n"
           "            enumerate(images):\n"
           "        # 正碼減反碼，消除環境光\n"
           "        diff = pos.astype(np.float32) - \\\n"
           "               neg.astype(np.float32)\n"
           "        bit_map = (diff > threshold)\\\n"
           "            .astype(np.uint32)\n"
           "        gray_code |= (\n"
           "            bit_map << (n_bits - 1 - bit_idx))\n\n"
           "    # Gray Code -> Binary\n"
           "    binary_code = gray_code.copy()\n"
           "    shift = 1\n"
           "    while shift < n_bits:\n"
           "        binary_code ^= (binary_code >> shift)\n"
           "        shift <<= 1\n\n"
           "    return binary_code\n\n"
           "def triangulate(code_map, calib_params):\n"
           '    """從編碼圖計算 3D 座標\n'
           "    code_map:     解碼後的條紋編號\n"
           '    calib_params: 校準參數"""\n'
           "    f = calib_params['focal_length']\n"
           "    B = calib_params['baseline']\n"
           "    cx = calib_params['cx']\n"
           "    cy = calib_params['cy']\n"
           "    pitch = calib_params['stripe_pitch']\n\n"
           "    h, w = code_map.shape\n"
           "    points_3d = np.zeros((h, w, 3),\n"
           "                         dtype=np.float64)\n\n"
           "    for y in range(h):\n"
           "        for x in range(w):\n"
           "            x_proj = code_map[y, x] * pitch\n"
           "            disparity = x_proj - x\n"
           "            if abs(disparity) < 1:\n"
           "                continue\n"
           "            Z = f * B / disparity\n"
           "            X = (x - cx) * Z / f\n"
           "            Y = (y - cy) * Z / f\n"
           "            points_3d[y, x] = [X, Y, Z]\n\n"
           "    return points_3d")

    # 24.3
    p.sec_title("雷射三角測量")
    p.txt(
        "雷射三角測量使用一條或多條雷射線投射到物體表面。相機從另一角度觀察雷射線，"
        "物體表面的高度變化導致雷射線在影像中產生位移。通過標定的幾何關係，"
        "位移量可轉換為高度值。\n\n"
        "優點：速度快（單線掃描），精度高（可達微米級）。\n"
        "常用於線掃描場景：零件在輸送帶上移動，雷射線逐行掃描。"
    )

    p.formula("高度計算公式（雷射三角測量）",
              "h = delta_x * sin(alpha) * sin(beta)\n"
              "    / sin(alpha + beta)\n\n"
              "  h       = 表面高度變化\n"
              "  delta_x = 雷射線在影像中的位移 (像素)\n"
              "  alpha   = 雷射入射角\n"
              "  beta    = 相機觀測角\n\n"
              "簡化（垂直觀測, beta=90 度）:\n"
              "  h = delta_x * sin(alpha) * k_calib",
              "k_calib 為校準係數，通過已知高度的標準件校準。精度與像素解析度和角度有關。")

    p.code("import cv2\nimport numpy as np\n\n"
           "def extract_laser_line(img,\n"
           "                        threshold=200):\n"
           '    """提取雷射線中心位置（亞像素）\n'
           "    img: 灰度影像，雷射線為亮線\n"
           '    回傳: 每列的雷射線 y 座標"""\n'
           "    h, w = img.shape\n"
           "    centers = np.full(w, np.nan)\n\n"
           "    for col in range(w):\n"
           "        profile = img[:, col]\\\n"
           "            .astype(np.float64)\n\n"
           "        # 找最大值\n"
           "        peak_idx = np.argmax(profile)\n"
           "        if profile[peak_idx] < threshold:\n"
           "            continue\n\n"
           "        # 亞像素插值（高斯擬合）\n"
           "        if (0 < peak_idx < h - 1\n"
           "                and profile[peak_idx] > 0):\n"
           "            y_m1 = np.log(\n"
           "                max(profile[peak_idx-1], 1))\n"
           "            y_0 = np.log(\n"
           "                max(profile[peak_idx], 1))\n"
           "            y_p1 = np.log(\n"
           "                max(profile[peak_idx+1], 1))\n\n"
           "            denom = 2*(2*y_0 - y_m1 - y_p1)\n"
           "            if abs(denom) > 1e-6:\n"
           "                offset = (\n"
           "                    y_m1 - y_p1) / denom\n"
           "                centers[col] = (\n"
           "                    peak_idx + offset)\n"
           "            else:\n"
           "                centers[col] = peak_idx\n\n"
           "    return centers\n\n"
           "def compute_height_profile(centers,\n"
           "                            ref_centers,\n"
           "                            k_calib):\n"
           '    """計算高度輪廓\n'
           "    centers:     當前掃描的雷射線位置\n"
           "    ref_centers: 參考平面的雷射線位置\n"
           '    k_calib:     校準係數 (mm/pixel)"""\n'
           "    displacement = centers - ref_centers\n"
           "    height = displacement * k_calib\n"
           "    return height")

    # 24.4
    p.sec_title("立體視覺（雙目相機）")
    p.txt(
        "立體視覺使用兩個相機從不同角度同時拍攝。通過匹配兩張影像中的對應點，"
        "計算視差（同一物體點在左右影像中的 x 坐標差），進而計算深度。"
    )

    p.formula("視差與深度公式",
              "Z = f * B / d\n\n"
              "  Z = 深度 (物體到相機的距離)\n"
              "  f = 焦距 (像素)\n"
              "  B = 基線距離 (兩相機間距, mm)\n"
              "  d = 視差 (左影像x - 右影像x, 像素)\n\n"
              "  d 越大 -> Z 越小 (物體越近)\n"
              "  d 越小 -> Z 越大 (物體越遠)\n\n"
              "深度精度: dZ = Z^2 / (f*B) * delta_d\n"
              "  delta_d = 視差精度 (通常 0.1~1 像素)",
              "深度精度與距離的平方成正比。近距離時精度高，遠距離時精度急劇下降。工業應用中通常在近距離（<1m）使用。")

    p.code("import cv2\nimport numpy as np\n\n"
           "def stereo_depth(img_left, img_right,\n"
           "                  focal_length, baseline):\n"
           '    """雙目立體視覺深度計算\n'
           "    img_left/right: 已校正的灰度影像\n"
           "    focal_length: 焦距 (像素)\n"
           '    baseline: 基線距離 (mm)"""\n'
           "    # 建立 StereoSGBM 匹配器\n"
           "    min_disp = 0\n"
           "    num_disp = 128  # 必須是 16 的倍數\n"
           "    block_size = 5\n\n"
           "    stereo = cv2.StereoSGBM_create(\n"
           "        minDisparity=min_disp,\n"
           "        numDisparities=num_disp,\n"
           "        blockSize=block_size,\n"
           "        P1=8 * block_size**2,\n"
           "        P2=32 * block_size**2,\n"
           "        disp12MaxDiff=1,\n"
           "        uniquenessRatio=10,\n"
           "        speckleWindowSize=100,\n"
           "        speckleRange=32,\n"
           "        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY\n"
           "    )\n\n"
           "    # 計算視差圖\n"
           "    disparity = stereo.compute(\n"
           "        img_left, img_right)\n"
           "    disp_float = disparity.astype(\n"
           "        np.float32) / 16.0\n\n"
           "    # 視差 -> 深度\n"
           "    depth = np.zeros_like(disp_float)\n"
           "    valid = disp_float > 0\n"
           "    depth[valid] = (\n"
           "        focal_length * baseline\n"
           "        / disp_float[valid])\n\n"
           "    return depth, disp_float")

    # 24.5
    p.sec_title("點雲處理與瑕疵檢測")
    p.txt(
        "3D 掃描的結果通常是點雲（大量 3D 坐標點的集合）。"
        "對點雲進行平面擬合、曲面擬合，計算每個點到擬合面的偏差，"
        "偏差超過閾值的區域即為瑕疵（凹陷、凸起、翹曲）。"
    )

    p.code("import numpy as np\n\n"
           "# 以下為 Open3D 點雲處理範例\n"
           "# import open3d as o3d\n\n"
           "def fit_plane(points):\n"
           '    """最小二乘平面擬合\n'
           "    points: Nx3 array\n"
           '    回傳: (a, b, c, d) 使得 ax+by+cz+d=0"""\n'
           "    centroid = points.mean(axis=0)\n"
           "    centered = points - centroid\n\n"
           "    # SVD 分解\n"
           "    _, _, Vh = np.linalg.svd(centered)\n"
           "    normal = Vh[-1]  # 最小奇異值的向量\n"
           "    a, b, c = normal\n"
           "    d = -np.dot(normal, centroid)\n"
           "    return a, b, c, d\n\n"
           "def compute_deviation(points, plane):\n"
           '    """計算每個點到平面的偏差"""\n'
           "    a, b, c, d = plane\n"
           "    norm = np.sqrt(a**2 + b**2 + c**2)\n"
           "    distances = (\n"
           "        a*points[:,0] + b*points[:,1]\n"
           "        + c*points[:,2] + d) / norm\n"
           "    return distances\n\n"
           "def detect_3d_defects(points,\n"
           "                      threshold=0.05):\n"
           '    """3D 瑕疵檢測\n'
           "    points:    Nx3 點雲\n"
           '    threshold: 偏差閾值 (mm)"""\n'
           "    # 擬合參考平面\n"
           "    plane = fit_plane(points)\n"
           "    deviations = compute_deviation(\n"
           "        points, plane)\n\n"
           "    # 分類偏差\n"
           "    defects = {\n"
           "        'bumps': points[\n"
           "            deviations > threshold],\n"
           "        'dents': points[\n"
           "            deviations < -threshold],\n"
           "        'flatness': float(\n"
           "            np.max(deviations)\n"
           "            - np.min(deviations)),\n"
           "        'max_bump': float(\n"
           "            np.max(deviations)),\n"
           "        'max_dent': float(\n"
           "            np.min(deviations)),\n"
           "    }\n\n"
           "    return defects, deviations")

    # 24.6
    p.sec_title("3D 瑕疵類型與檢測方法")

    p.table(
        ["瑕疵類型", "量測指標", "檢測方法", "精度要求"],
        [
            ["翹曲/彎曲", "平面度偏差", "結構光/雷射掃描", "~10 um"],
            ["凹陷/凹坑", "深度值", "結構光/雷射", "~5 um"],
            ["毛邊高度", "邊緣高度", "雷射三角測量", "~10 um"],
            ["焊球高度", "共面性", "結構光+多角度", "~5 um"],
            ["螺紋深度", "溝槽深度", "雷射掃描", "~20 um"],
            ["表面粗糙度", "Ra/Rz", "白光干涉儀", "~0.01 um"],
            ["體積缺損", "缺失體積", "多角度掃描", "~50 um"],
        ]
    )

    p.case("BGA 焊球高度檢測",
           "場景：BGA（Ball Grid Array）封裝的焊球共面性檢測。焊球高度偏差會導致虛焊。\n\n"
           "規格要求：\n"
           "- 焊球高度：300 +/- 30 um\n"
           "- 共面性（Coplanarity）：< 50 um\n"
           "- 直徑：350 +/- 25 um\n\n"
           "檢測方案：\n"
           "1. 硬體：高精度結構光系統（Z 解析度 2 um）\n"
           "2. 投射 8 位灰碼 + 4 步相移圖案\n"
           "3. 重建焊球頂面的 3D 點雲\n"
           "4. 對每個焊球擬合球面，提取頂點高度\n"
           "5. 擬合基準平面（PCB 表面）\n"
           "6. 計算每個焊球相對基準面的高度\n"
           "7. 共面性 = max(h) - min(h)\n\n"
           "判定：\n"
           "- 任一焊球高度超出 300 +/- 30 um -> NG\n"
           "- 共面性 > 50 um -> NG\n"
           "- 焊球缺失 -> NG\n"
           "- 焊球直徑異常 -> NG\n\n"
           "結果：漏檢率 < 0.01%，檢測速度 0.5 秒/件。")

    p.tip(
        "3D 精度影響因素：\n"
        "1. 校準品質：使用高精度標準件，定期重新校準\n"
        "2. 環境振動：使用防振平台，避免量測時產生振動\n"
        "3. 光學解析度：像素尺寸和景深限制了空間解析度\n"
        "4. 材料表面特性：反光面需要使用漫射噴霧或特殊照明\n"
        "5. 溫度：精密量測需控制在恆溫環境\n"
        "6. 掃描密度：點與點之間距離越小，重建越精確但速度越慢"
    )

    p.warn(
        "3D 檢測的成本與限制：\n"
        "1. 硬體成本高：結構光系統通常比 2D 相機貴 5-20 倍\n"
        "2. 速度較慢：結構光需投射多張圖案，雷射需逐線掃描\n"
        "3. 反光面問題：金屬鏡面、透明材料的 3D 重建非常困難\n"
        "4. 校準複雜：3D 系統的校準比 2D 複雜得多，需要專業知識\n"
        "5. 資料量大：點雲資料量遠大於 2D 影像，存儲和處理成本高\n"
        "6. 不是所有瑕疵都需要 3D——先確認 2D 方法確實不足再考慮 3D"
    )


def write_ch25(p):
    p.ch_title("參數調校方法論")
    p.txt(
        "瑕疵檢測系統的效能高度依賴參數設定：閾值、核大小、面積範圍、信心分數等。"
        "錯誤的參數會導致漏檢或誤檢。本章介紹系統化的參數調校方法論，"
        "從 ROC 曲線分析到貝葉斯優化，幫助工程師科學地找到最佳參數組合，"
        "而非依賴經驗猜測。"
    )

    # 25.1
    p.sec_title("系統化參數調校流程")
    p.txt(
        "有效的參數調校遵循以下流程：\n\n"
        "1. 識別關鍵參數：列出所有可調參數，識別對檢測結果影響最大的 3-5 個\n"
        "2. 準備驗證資料：收集一組有標註的良品和瑕疵品影像（至少各 50 張）\n"
        "3. 設計實驗：系統化地掃描參數範圍\n"
        "4. 評估效果：計算 Precision, Recall, F1 等指標\n"
        "5. 選擇最佳組合：ROC 曲線、Youden's J 或業務導向的選擇\n"
        "6. 驗證穩定性：交叉驗證確認結果不是偶然\n"
        "7. 凍結參數：確定後鎖定，版本管理\n\n"
        "關鍵原則：永遠在獨立的測試集上評估，避免在調參集上過擬合。"
    )

    # 25.2
    p.sec_title("ROC 曲線與最佳閾值選擇")
    p.txt(
        "ROC（Receiver Operating Characteristic）曲線展示了不同閾值下的 TPR（真陽性率）"
        "和 FPR（假陽性率）之間的取捨關係。AUC（曲線下面積）衡量分類器的整體性能。"
    )

    p.formula("TPR, FPR, AUC",
              "TPR (Recall) = TP / (TP + FN)\n"
              "  = 實際瑕疵中被正確檢出的比例\n\n"
              "FPR = FP / (FP + TN)\n"
              "  = 實際良品中被誤判的比例\n\n"
              "AUC = 曲線下面積\n"
              "  AUC = 1.0 : 完美分類器\n"
              "  AUC = 0.5 : 隨機猜測\n\n"
              "Youden's J = TPR - FPR\n"
              "  最佳閾值 = argmax(J)",
              "Youden's J 同時最大化檢出率和最小化誤報率。"
              "但在工業檢測中，可能需要以高 Recall 為優先（寧可誤報不可漏報）。")

    p.code("import numpy as np\nfrom sklearn.metrics import roc_curve, auc\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n\n"
           "def plot_roc_find_threshold(y_true, y_scores,\n"
           "                             save_path=None):\n"
           '    """繪製 ROC 曲線並找最佳閾值\n'
           "    y_true:   真實標籤 (0=良品, 1=瑕疵)\n"
           '    y_scores: 分數（越高越可能是瑕疵）"""\n'
           "    fpr, tpr, thresholds = roc_curve(\n"
           "        y_true, y_scores)\n"
           "    roc_auc = auc(fpr, tpr)\n\n"
           "    # Youden's J 最佳閾值\n"
           "    j_scores = tpr - fpr\n"
           "    best_idx = np.argmax(j_scores)\n"
           "    best_thresh = thresholds[best_idx]\n"
           "    best_tpr = tpr[best_idx]\n"
           "    best_fpr = fpr[best_idx]\n\n"
           "    # 高 Recall 閾值 (TPR >= 0.99)\n"
           "    high_recall_idx = np.where(\n"
           "        tpr >= 0.99)[0]\n"
           "    if len(high_recall_idx) > 0:\n"
           "        hr_idx = high_recall_idx[0]\n"
           "        hr_thresh = thresholds[hr_idx]\n"
           "    else:\n"
           "        hr_thresh = thresholds[0]\n\n"
           "    # 繪圖\n"
           "    plt.figure(figsize=(8, 6))\n"
           "    plt.plot(fpr, tpr,\n"
           "             label=f'AUC={roc_auc:.4f}')\n"
           "    plt.plot(best_fpr, best_tpr, 'ro',\n"
           "             label=f'Best (J={best_thresh:.3f})')\n"
           "    plt.plot([0,1], [0,1], 'k--')\n"
           "    plt.xlabel('FPR')\n"
           "    plt.ylabel('TPR')\n"
           "    plt.title('ROC Curve')\n"
           "    plt.legend()\n"
           "    if save_path:\n"
           "        plt.savefig(save_path, dpi=150)\n"
           "    plt.close()\n\n"
           "    return {\n"
           "        'auc': roc_auc,\n"
           "        'best_threshold': float(best_thresh),\n"
           "        'best_tpr': float(best_tpr),\n"
           "        'best_fpr': float(best_fpr),\n"
           "        'high_recall_threshold': float(\n"
           "            hr_thresh)\n"
           "    }")

    # 25.3
    p.sec_title("交叉驗證")
    p.txt(
        "在小資料集上調參容易過擬合（參數恰好適合這組資料但泛化不佳）。"
        "交叉驗證將資料分成 K 份，輪流使用其中 K-1 份調參、1 份驗證，"
        "取 K 次結果的平均值作為最終評估。\n\n"
        "瑕疵檢測中的注意事項：\n"
        "- 使用分層抽樣（Stratified）確保每份中瑕疵比例一致\n"
        "- 同一零件的多張影像不能分到不同份中（避免資料洩漏）"
    )

    p.code("from sklearn.model_selection import (\n"
           "    StratifiedKFold, cross_val_score\n"
           ")\nfrom sklearn.svm import SVC\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nimport numpy as np\n\n"
           "def cross_validate_classifier(X, y, n_folds=5):\n"
           '    """K-fold 交叉驗證\n'
           "    X: 特徵矩陣 (n_samples, n_features)\n"
           '    y: 標籤 (0=OK, 1=NG)"""\n'
           "    # 建立管線\n"
           "    pipeline = Pipeline([\n"
           "        ('scaler', StandardScaler()),\n"
           "        ('svm', SVC(kernel='rbf', C=10,\n"
           "                     gamma='scale',\n"
           "                     probability=True))\n"
           "    ])\n\n"
           "    # 分層 K-fold\n"
           "    cv = StratifiedKFold(\n"
           "        n_splits=n_folds, shuffle=True,\n"
           "        random_state=42)\n\n"
           "    # 多指標評估\n"
           "    metrics = {}\n"
           "    for metric in ['accuracy', 'precision',\n"
           "                   'recall', 'f1']:\n"
           "        scores = cross_val_score(\n"
           "            pipeline, X, y, cv=cv,\n"
           "            scoring=metric)\n"
           "        metrics[metric] = {\n"
           "            'mean': float(np.mean(scores)),\n"
           "            'std': float(np.std(scores)),\n"
           "            'scores': scores.tolist()\n"
           "        }\n\n"
           "    return metrics")

    # 25.4
    p.sec_title("敏感度分析")
    p.txt(
        "敏感度分析評估每個參數對檢測結果的影響程度。了解哪些參數最敏感，"
        "有助於集中精力調校關鍵參數，並設定合理的容差範圍。"
    )

    p.code("import numpy as np\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n\n"
           "def sensitivity_analysis(evaluate_fn,\n"
           "                          params_range,\n"
           "                          base_params):\n"
           '    """單因子敏感度分析\n'
           "    evaluate_fn:  評估函數，輸入參數，回傳分數\n"
           "    params_range: dict {參數名: [值列表]}\n"
           '    base_params:  基準參數 dict"""\n'
           "    results = {}\n\n"
           "    for param_name, values in \\\n"
           "            params_range.items():\n"
           "        scores = []\n"
           "        for val in values:\n"
           "            test_params = base_params.copy()\n"
           "            test_params[param_name] = val\n"
           "            score = evaluate_fn(test_params)\n"
           "            scores.append(score)\n"
           "        results[param_name] = {\n"
           "            'values': values,\n"
           "            'scores': scores,\n"
           "            'sensitivity': (\n"
           "                max(scores) - min(scores))\n"
           "        }\n\n"
           "    # 繪製敏感度圖\n"
           "    n = len(results)\n"
           "    fig, axes = plt.subplots(\n"
           "        1, n, figsize=(4*n, 3))\n"
           "    if n == 1:\n"
           "        axes = [axes]\n\n"
           "    for ax, (name, data) in \\\n"
           "            zip(axes, results.items()):\n"
           "        ax.plot(data['values'],\n"
           "                data['scores'], 'bo-')\n"
           "        ax.set_xlabel(name)\n"
           "        ax.set_ylabel('F1 Score')\n"
           "        ax.set_title(\n"
           "            f'Sensitivity: '\n"
           "            f'{data[\"sensitivity\"]:.3f}')\n"
           "    plt.tight_layout()\n"
           "    plt.savefig('sensitivity.png', dpi=150)\n"
           "    plt.close()\n\n"
           "    return results")

    # 25.5
    p.sec_title("自動參數搜索")
    p.txt(
        "手動調參效率低且難以找到全局最優。自動搜索方法可系統化地探索參數空間。\n\n"
        "三種主要方法：\n"
        "- 網格搜索（Grid Search）：遍歷所有組合。簡單但組合爆炸。\n"
        "- 隨機搜索（Random Search）：隨機採樣。在高維空間中效率高於網格搜索。\n"
        "- 貝葉斯優化（Bayesian Optimization）：根據已有結果智能選擇下一個試驗點。最高效。"
    )

    p.code("import optuna\nimport numpy as np\n\n"
           "def optimize_parameters(images, labels):\n"
           '    """使用 Optuna 貝葉斯優化搜索最佳參數"""\n\n'
           "    def objective(trial):\n"
           "        # 定義搜索空間\n"
           "        blur_k = trial.suggest_int(\n"
           "            'blur_kernel', 3, 11, step=2)\n"
           "        thresh_c = trial.suggest_int(\n"
           "            'threshold_c', 5, 30)\n"
           "        morph_k = trial.suggest_int(\n"
           "            'morph_kernel', 3, 9, step=2)\n"
           "        min_area = trial.suggest_int(\n"
           "            'min_area', 20, 500, step=20)\n"
           "        block_size = trial.suggest_int(\n"
           "            'block_size', 11, 101, step=10)\n\n"
           "        # 用這組參數檢測所有影像\n"
           "        tp = fp = fn = tn = 0\n"
           "        for img, label in \\\n"
           "                zip(images, labels):\n"
           "            pred = detect_with_params(\n"
           "                img, blur_k, thresh_c,\n"
           "                morph_k, min_area,\n"
           "                block_size)\n"
           "            if pred and label:     tp += 1\n"
           "            elif pred and not label: fp += 1\n"
           "            elif not pred and label: fn += 1\n"
           "            else:                  tn += 1\n\n"
           "        # 計算 F1\n"
           "        prec = tp/(tp+fp) if tp+fp else 0\n"
           "        rec = tp/(tp+fn) if tp+fn else 0\n"
           "        f1 = (2*prec*rec/(prec+rec)\n"
           "              if prec+rec else 0)\n\n"
           "        return f1\n\n"
           "    # 執行優化\n"
           "    study = optuna.create_study(\n"
           "        direction='maximize',\n"
           "        sampler=optuna.samplers.TPESampler(\n"
           "            seed=42))\n"
           "    study.optimize(objective, n_trials=200,\n"
           "                   show_progress_bar=True)\n\n"
           "    print(f'Best F1: '\n"
           "          f'{study.best_value:.4f}')\n"
           "    print(f'Best params: '\n"
           "          f'{study.best_params}')\n\n"
           "    return study.best_params")

    # 25.6
    p.sec_title("參數驗證與版本管理")
    p.txt(
        "確定最佳參數後，必須凍結並納入版本管理。參數應以配置檔形式存儲，"
        "而非硬編碼在程式中。每次參數變更都應記錄原因和效果。"
    )

    p.code("import yaml\nfrom pathlib import Path\nfrom datetime import datetime\n\n"
           "class ParamManager:\n"
           '    """參數配置管理系統"""\n\n'
           "    def __init__(self, config_dir):\n"
           "        self.config_dir = Path(config_dir)\n"
           "        self.config_dir.mkdir(\n"
           "            parents=True, exist_ok=True)\n\n"
           "    def save_config(self, params, version,\n"
           "                     description=''):\n"
           '        """保存參數配置"""\n'
           "        config = {\n"
           "            'version': version,\n"
           "            'created': datetime.now()\n"
           "                .isoformat(),\n"
           "            'description': description,\n"
           "            'parameters': params\n"
           "        }\n"
           "        path = (self.config_dir /\n"
           "                f'config_v{version}.yaml')\n"
           "        with open(path, 'w',\n"
           "                  encoding='utf-8') as f:\n"
           "            yaml.dump(\n"
           "                config, f,\n"
           "                allow_unicode=True,\n"
           "                default_flow_style=False)\n"
           "        return path\n\n"
           "    def load_config(self, version=None):\n"
           '        """載入參數配置\n'
           '        version=None 時載入最新版本"""\n'
           "        if version:\n"
           "            path = (self.config_dir /\n"
           "                    f'config_v{version}.yaml')\n"
           "        else:\n"
           "            files = sorted(\n"
           "                self.config_dir.glob(\n"
           "                    'config_v*.yaml'))\n"
           "            if not files:\n"
           "                raise FileNotFoundError(\n"
           "                    'No config found')\n"
           "            path = files[-1]\n\n"
           "        with open(path, 'r',\n"
           "                  encoding='utf-8') as f:\n"
           "            config = yaml.safe_load(f)\n"
           "        return config\n\n"
           "# 使用範例\n"
           "pm = ParamManager('./configs')\n\n"
           "# 保存最佳參數\n"
           "best_params = {\n"
           "    'preprocessing': {\n"
           "        'blur_kernel': 5,\n"
           "        'clahe_clip': 2.0,\n"
           "        'clahe_grid': 8\n"
           "    },\n"
           "    'segmentation': {\n"
           "        'method': 'adaptive',\n"
           "        'block_size': 51,\n"
           "        'threshold_c': 10\n"
           "    },\n"
           "    'morphology': {\n"
           "        'kernel_size': 5,\n"
           "        'open_iter': 1,\n"
           "        'close_iter': 1\n"
           "    },\n"
           "    'classification': {\n"
           "        'min_area': 100,\n"
           "        'max_area': 50000,\n"
           "        'min_circularity': 0.3,\n"
           "        'score_threshold': 40\n"
           "    }\n"
           "}\n\n"
           "pm.save_config(\n"
           "    best_params, version='1.2.0',\n"
           "    description='Optuna 優化後參數')")

    p.table(
        ["參數", "推薦搜索範圍", "步進", "敏感度"],
        [
            ["高斯核大小", "3 ~ 11", "2", "中"],
            ["自適應閾值 C", "5 ~ 30", "1", "高"],
            ["自適應 blockSize", "11 ~ 101", "10", "高"],
            ["形態學核大小", "3 ~ 9", "2", "中"],
            ["最小面積", "20 ~ 500", "20", "高"],
            ["圓度閾值", "0.2 ~ 0.9", "0.05", "中"],
            ["差分閾值", "10 ~ 80", "5", "極高"],
            ["z-score 倍數 k", "2.0 ~ 5.0", "0.5", "高"],
            ["SSIM 閾值", "0.8 ~ 0.99", "0.01", "中"],
        ]
    )

    p.case("Otsu 閾值 vs 自適應閾值的系統化比較",
           "場景：金屬零件表面刮痕檢測，照明存在輕微不均勻。\n\n"
           "實驗設計：\n"
           "- 驗證資料集：200 張（100 良品 + 100 瑕疵品，標註人員交叉驗證）\n"
           "- 方法 A：Otsu 全域閾值\n"
           "- 方法 B：自適應高斯閾值（掃描 blockSize 和 C）\n"
           "- 評估指標：F1、Recall、Precision\n\n"
           "結果：\n"
           "- Otsu 最佳 F1 = 0.82 (Recall=0.78, Precision=0.87)\n"
           "- 自適應閾值最佳 F1 = 0.93 (blockSize=41, C=12)\n"
           "  (Recall=0.95, Precision=0.91)\n\n"
           "分析：\n"
           "- Otsu 在照明均勻區域表現好，但不均勻區域漏檢嚴重\n"
           "- 自適應閾值在所有區域都穩定\n"
           "- blockSize 對結果影響最大（敏感度 0.15）\n"
           "- C 值的影響次之（敏感度 0.08）\n\n"
           "決策：選用自適應閾值，參數 blockSize=41, C=12，經 5-fold 交叉驗證確認穩定。")

    p.tip(
        "參數調校的經驗法則：\n"
        "1. 先固定大部分參數，一次只調一個——理解每個參數的獨立影響\n"
        "2. 從粗搜索開始（大步進），縮小範圍後再細搜索\n"
        "3. 始終在獨立測試集上評估——不要在調參集上看結果\n"
        "4. 記錄每次實驗的參數和結果——方便回溯\n"
        "5. 最終參數需通過交叉驗證確認穩定性\n"
        "6. 留安全餘量：如果最佳閾值為 42，設 40 或 45 測試結果是否仍可接受\n"
        "7. 考慮業務需求：是寧可誤報（高 Recall）還是寧可漏報（高 Precision）"
    )

    p.warn(
        "過度擬合參數的風險：\n"
        "1. 參數完美匹配當前資料集，但換一批材料就失效\n"
        "2. 同時優化太多參數導致交互作用複雜，難以理解和維護\n"
        "3. 使用太少的驗證資料（< 30 張），結果不具統計意義\n"
        "4. 忽略極端案例：調校時遇到的最難樣本決定了實際表現\n"
        "5. 不記錄參數版本：無法追溯哪個版本在產線上運行\n"
        "6. 調校一次後永不更新：產品和環境在變化，參數也需要定期驗證"
    )
