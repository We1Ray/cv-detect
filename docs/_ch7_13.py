# -*- coding: utf-8 -*-
"""Chapter 7-13 content"""


def write_ch7(p):
    p.ch_title("模板匹配")
    p.txt("模板匹配將測試影像與已知良品參考（模板）進行比較。兩者之間的差異指示潛在瑕疵。對於外觀一致且可重複的產品（PCB 板、印刷標籤等）特別有效。")

    # 7.1
    p.sec_title("歸一化互相關（NCC）")

    p.formula("歸一化互相關",
              "NCC(x,y) =\n"
              " SUM[(T(i,j)-T_mean)*(I(x+i,y+j)-I_mean)]\n"
              " / sqrt[SUM(T-T_mean)^2 * SUM(I-I_mean)^2]\n\n"
              "結果範圍: [-1, +1]\n"
              "  +1 = 完美匹配\n"
              "   0 = 無相關\n"
              "  -1 = 反向匹配",
              "用於在測試影像中定位模板。對線性亮度變化具有不變性。")

    p.code("# 模板匹配\n"
           "template = cv2.imread('golden_ref.jpg', 0)\n"
           "result = cv2.matchTemplate(\n"
           "    gray, template, cv2.TM_CCOEFF_NORMED)\n"
           "_, max_val, _, max_loc = cv2.minMaxLoc(result)")

    # 7.2
    p.sec_title("基於形狀的匹配")
    p.txt("基於形狀的匹配使用邊緣/梯度資訊而非原始像素值，對照明變化和輕微表面變化更具魯棒性。這是工業視覺 find_shape_model 使用的方法。\n\n"
          "流程：\n"
          "1. 從參考零件建立形狀模型（提取邊緣）\n"
          "2. 對每個測試影像搜尋形狀模型\n"
          "3. 獲取找到實例的姿態（位置、角度、縮放）\n"
          "4. 使用姿態將測試影像對齊到參考\n"
          "5. 計算逐像素差異以檢測瑕疵")

    # 7.3
    p.sec_title("差分影像法")

    p.formula("絕對差分",
              "Diff(x,y) = |I_test(x,y) - I_ref(x,y)|\n"
              "瑕疵遮罩: Diff(x,y) > 閾值",
              "對齊準確時簡單有效。對未對齊敏感——即使 1 像素偏移也會產生假邊緣。")

    p.code("# 差分瑕疵檢測\n"
           "# 1. 對齊測試影像到參考\n"
           "# 2. 計算絕對差分\n"
           "diff = cv2.absdiff(aligned_test, reference)\n\n"
           "# 3. 閾值分割\n"
           "_, mask = cv2.threshold(\n"
           "    diff, 30, 255, cv2.THRESH_BINARY)\n\n"
           "# 4. 形態學清理\n"
           "k = cv2.getStructuringElement(\n"
           "    cv2.MORPH_ELLIPSE, (5,5))\n"
           "mask = cv2.morphologyEx(\n"
           "    mask, cv2.MORPH_OPEN, k)\n"
           "mask = cv2.morphologyEx(\n"
           "    mask, cv2.MORPH_CLOSE, k)\n\n"
           "# 5. 分析瑕疵區域\n"
           "contours, _ = cv2.findContours(\n"
           "    mask, cv2.RETR_EXTERNAL,\n"
           "    cv2.CHAIN_APPROX_SIMPLE)")


def write_ch8(p):
    p.ch_title("光度立體法")
    p.txt("光度立體法使用從不同方向照明拍攝的多張影像來重建表面法向量。這揭示了在單方向照明下不可見或模糊的表面瑕疵（凹痕、凸起、刮痕）。")

    # 8.1
    p.sec_title("原理與打光配置")
    p.txt("配置：相機固定在物體上方。多個光源（通常 4~8 個）位於物體周圍不同角度。每次開啟一個光源並拍攝一張影像。每個像素的亮度隨該點的表面法向量而變化。")

    p.formula("朗伯反射模型",
              "I(x,y) = rho(x,y) * max(0, n(x,y) . L)\n\n"
              "  I     = 觀測亮度\n"
              "  rho   = 表面反照率（反射率）\n"
              "  n     = 表面法向量 [nx, ny, nz]\n"
              "  L     = 光源方向向量 [lx, ly, lz]\n"
              "  (.)   = 點積",
              "每張影像為每個像素提供一個方程式。3+ 張不同光源方向的影像即可求解法向量 n(x,y)。")

    # 8.2
    p.sec_title("表面法向量估計")

    p.formula("最小二乘法向量估計",
              "對 k 張影像，光源方向 L1..Lk:\n\n"
              "[I1]   [L1x L1y L1z] [rho*nx]\n"
              "[I2] = [L2x L2y L2z] [rho*ny]\n"
              "[..]   [..  ..  .. ] [rho*nz]\n\n"
              "=> I = L * g\n"
              "解: g = (L^T * L)^(-1) * L^T * I\n"
              "反照率: rho = |g|,  法向量: n = g/|g|",
              "需要至少 3 個非共面的光源方向。更多光源 = 更穩健的估計。")

    p.code("# 光度立體法實作\n"
           "# images: k 張灰度影像列表\n"
           "# lights: kx3 光源方向矩陣\n"
           "H, W = images[0].shape\n"
           "I = np.vstack([img.flatten().astype(np.float64)\n"
           "               for img in images])  # (k, N)\n"
           "L = np.array(lights)  # (k, 3)\n\n"
           "# 求解\n"
           "g = np.linalg.inv(L.T @ L) @ L.T @ I\n\n"
           "# 反照率和法向量\n"
           "albedo = np.linalg.norm(g, axis=0)\n"
           "albedo = albedo.reshape(H, W)\n"
           "normals = g / (np.linalg.norm(\n"
           "    g, axis=0, keepdims=True) + 1e-8)")

    # 8.3
    p.sec_title("利用法向量檢測表面瑕疵")
    p.txt("表面瑕疵導致法向量局部變化：\n"
          "- 刮痕：法向量跨越刮痕處急劇偏轉\n"
          "- 凹痕：法向量向凹痕中心傾斜\n"
          "- 凸起：法向量向外傾斜\n\n"
          "檢測方法：計算法向量場的梯度（曲率），閾值分割以找到異常曲率區域。")


def write_ch9(p):
    p.ch_title("量測與擬合")
    p.txt("基於量測的瑕疵檢測驗證幾何屬性：尺寸、直線度、圓度、角度和距離。如果測量值超出公差範圍，則零件為不良品。")

    # 9.1
    p.sec_title("卡尺工具與亞像素邊緣檢測")
    p.txt("卡尺工具在邊緣上投射一條線（或弧線），沿該線精確找到邊緣位置。沿著邊緣放置多個卡尺，可以亞像素精度量測邊緣輪廓。")

    p.formula("亞像素邊緣檢測（一維梯度插值）",
              "給定 3 個連續梯度值 g[i-1], g[i], g[i+1]\n"
              "其中 g[i] 為最大值:\n\n"
              "offset = (g[i-1]-g[i+1]) /\n"
              "         (2*(g[i-1]-2*g[i]+g[i+1]))\n"
              "edge_pos = i + offset",
              "梯度峰值附近的拋物線插值。可達到約 0.1 像素精度。")

    # 9.2
    p.sec_title("直線 / 圓 / 橢圓擬合")

    p.formula("最小二乘圓擬合",
              "最小化: SUM[(sqrt((xi-cx)^2+(yi-cy)^2)-R)^2]\n\n"
              "線性化後可用最小二乘法求解 (cx, cy, R)",
              "將圓擬合到邊緣點。最大/最小半徑偏差指示圓度誤差。用於檢測孔洞、銷釘和墊圈。")

    p.code("# OpenCV 擬合\n"
           "pts = np.array(edge_pts, dtype=np.float32)\n\n"
           "# 最小外接圓\n"
           "(cx,cy), r = cv2.minEnclosingCircle(pts)\n\n"
           "# 擬合橢圓（需 >= 5 點）\n"
           "if len(pts) >= 5:\n"
           "    ellipse = cv2.fitEllipse(pts)\n\n"
           "# 擬合直線\n"
           "line = cv2.fitLine(\n"
           "    pts, cv2.DIST_L2, 0, 0.01, 0.01)")

    # 9.3
    p.sec_title("尺寸驗證")
    p.txt("擬合幾何基元後，量測關鍵尺寸並與規格比較：\n"
          "- 兩邊緣間距（寬度、間隙）\n"
          "- 孔洞/銷釘直徑\n"
          "- 表面間角度\n"
          "- 直線度、平面度、圓度\n\n"
          "任何尺寸超出公差範圍 → 不良品。")


def write_ch10(p):
    p.ch_title("機器學習瑕疵分類")
    p.txt("當基於規則的方法（閾值 + 特徵）變得過於複雜或脆弱時，機器學習分類器可以從訓練數據中學習區分瑕疵類型。兩種方法：傳統 ML（手工特徵）和深度學習（自動學習特徵）。")

    # 10.1
    p.sec_title("特徵工程")
    p.txt("對於傳統 ML，特徵品質決定分類準確度。常用瑕疵分類特徵：")
    p.blist([
        "幾何特徵：面積、周長、圓度、長寬比、實心度、Hu 矩",
        "亮度特徵：均值、標準差、最小值、最大值、偏度、峰度",
        "紋理特徵：GLCM 的 Haralick 特徵（對比度、相關性、能量、均勻性）",
        "梯度特徵：方向梯度直方圖（HOG）",
    ])

    p.formula("GLCM（灰度共生矩陣）",
              "P(i,j|d,theta): 距離 d、角度 theta 處\n"
              "  像素對亮度為 (i,j) 的計數\n\n"
              "對比度   = SUM[(i-j)^2 * P(i,j)]\n"
              "相關性   = SUM[(i-mu_i)(j-mu_j)*P/(s_i*s_j)]\n"
              "能量     = SUM[P(i,j)^2]\n"
              "均勻性   = SUM[P(i,j) / (1+|i-j|)]",
              "這 4 個 Haralick 特徵捕捉紋理屬性。對比度衡量局部變化；能量衡量均勻性。")

    # 10.2
    p.sec_title("SVM 分類器")

    p.formula("SVM 決策函數",
              "f(x) = sign(SUM[a_i*y_i*K(x_i,x)] + b)\n\n"
              "常用核函數:\n"
              "  線性: K(x,y) = x . y\n"
              "  RBF:  K(x,y) = exp(-gamma*|x-y|^2)\n"
              "  多項式: K(x,y) = (gamma*x.y + r)^d",
              "SVM 找到最大化類間間距的最佳超平面。RBF 核是瑕疵分類的預設選擇。")

    p.code("from sklearn.svm import SVC\n"
           "from sklearn.preprocessing import StandardScaler\n\n"
           "# 正規化特徵\n"
           "scaler = StandardScaler()\n"
           "X_s = scaler.fit_transform(features)\n\n"
           "# 訓練 SVM\n"
           "svm = SVC(kernel='rbf', C=10, gamma='scale')\n"
           "svm.fit(X_train, y_train)\n"
           "acc = svm.score(X_test, y_test)\n"
           "print(f'準確率: {acc:.2%}')")

    # 10.3
    p.sec_title("深度學習：CNN 異常檢測")

    p.sub_sec("1. 分類 CNN")
    p.txt("訓練 CNN 將影像區塊分類為「OK」或「瑕疵」。需要標註訓練數據。架構：通常是小型網路（3~5 個卷積層 + 全連接層），在裁剪區塊上訓練。")

    p.sub_sec("2. 異常檢測（自編碼器）")
    p.txt("僅用良品樣本訓練自編碼器。網路學習重建正常影像。當輸入瑕疵影像時，瑕疵區域的重建誤差會很高。")

    p.formula("自編碼器異常分數",
              "anomaly(x,y) = |I_input - I_reconstructed|^2\n"
              "瑕疵判定: anomaly > 閾值",
              "不需要瑕疵訓練樣本！僅需要一組良品影像用於訓練。")

    p.code("# 異常檢測自編碼器（PyTorch）\n"
           "class DefectAE(nn.Module):\n"
           "    def __init__(self):\n"
           "        super().__init__()\n"
           "        self.enc = nn.Sequential(\n"
           "          nn.Conv2d(1,32,3,stride=2,padding=1),\n"
           "          nn.ReLU(),\n"
           "          nn.Conv2d(32,64,3,stride=2,padding=1),\n"
           "          nn.ReLU())\n"
           "        self.dec = nn.Sequential(\n"
           "          nn.ConvTranspose2d(64,32,3,\n"
           "            stride=2,padding=1,output_padding=1),\n"
           "          nn.ReLU(),\n"
           "          nn.ConvTranspose2d(32,1,3,\n"
           "            stride=2,padding=1,output_padding=1),\n"
           "          nn.Sigmoid())\n"
           "    def forward(self, x):\n"
           "        return self.dec(self.enc(x))\n\n"
           "# 訓練：僅對良品影像最小化 MSE\n"
           "# 推論：高 MSE 區域 = 瑕疵")


def write_ch11(p):
    p.ch_title("變異模型（統計背景）")
    p.txt("變異模型是工業視覺系統使用的統計方法。利用多張訓練影像計算每個位置像素值的均值和標準差來建模。檢測時，顯著偏離模型的像素被標記為瑕疵。")

    # 11.1
    p.sec_title("訓練階段：均值與變異數模型")

    p.formula("變異模型訓練",
              "對每個像素位置 (x,y)，從 N 張訓練影像:\n\n"
              "mean(x,y) = (1/N) * SUM(k=1..N)[I_k(x,y)]\n\n"
              "var(x,y) = (1/(N-1)) *\n"
              "  SUM(k=1..N)[(I_k(x,y) - mean(x,y))^2]\n\n"
              "std(x,y) = sqrt(var(x,y))",
              "需要 15~50 張良品影像。影像必須在訓練前對齊（相同位置/方向）。")

    p.code("# 變異模型訓練\n"
           "import glob\n"
           "train_imgs = []\n"
           "for f in sorted(glob.glob('train/*.png')):\n"
           "    img = cv2.imread(f, 0).astype(np.float64)\n"
           "    train_imgs.append(img)\n\n"
           "stack = np.array(train_imgs)  # (N, H, W)\n"
           "mean_m = np.mean(stack, axis=0)\n"
           "std_m = np.std(stack, axis=0, ddof=1)\n"
           "std_m = np.maximum(std_m, 5.0)  # 避免除零")

    # 11.2
    p.sec_title("檢測階段：統計比較")

    p.formula("變異模型瑕疵檢測",
              "z(x,y) = |I_test(x,y) - mean(x,y)| / std(x,y)\n\n"
              "瑕疵判定: z(x,y) > k  (通常 k = 3.0~5.0)\n\n"
              "k=3: 99.7% 正常像素在範圍內\n"
              "k=4: 99.99% 正常像素在範圍內",
              "k 越高 = 假陽性越少但可能遺漏細微瑕疵。k 越低 = 越敏感但假陽性越多。")

    p.code("# 變異模型檢測\n"
           "test = cv2.imread('test.png', 0)\n"
           "test = test.astype(np.float64)\n\n"
           "# 計算 z-score\n"
           "z = np.abs(test - mean_m) / std_m\n\n"
           "# 閾值分割\n"
           "k = 3.5\n"
           "mask = (z > k).astype(np.uint8) * 255\n\n"
           "# 形態學清理\n"
           "kern = cv2.getStructuringElement(\n"
           "    cv2.MORPH_ELLIPSE, (5,5))\n"
           "mask = cv2.morphologyEx(\n"
           "    mask, cv2.MORPH_OPEN, kern)\n"
           "mask = cv2.morphologyEx(\n"
           "    mask, cv2.MORPH_CLOSE, kern)")


def write_ch12(p):
    p.ch_title("完整瑕疵檢測案例")
    p.txt("本章展示 10 個實際瑕疵檢測場景，每個都包含從影像擷取到瑕疵分類的完整流程。")

    # Case 1
    p.sec_title("案例 1：餅乾破損檢測（Blob 分析）")
    p.case("餅乾品質檢測",
           "檢測輸送帶上破碎、缺角或裂開的餅乾。\n\n"
           "瑕疵特徵：缺損材料改變形狀/面積。\n\n"
           "流程：\n"
           "1. 前處理：灰度轉換，高斯模糊（5x5）\n"
           "2. 分割：Otsu 閾值分離餅乾與輸送帶\n"
           "3. 形態學：開運算（去碎屑），閉運算（填小隙）\n"
           "4. 特徵提取：面積、圓度、實心度\n"
           "5. 判定規則：\n"
           "   OK：面積>8000 且 圓度>0.75 且 實心度>0.90\n"
           "   破損：否則\n\n"
           "關鍵：實心度是最具區分力的特徵——破損邊緣產生凹陷，降低面積/凸包比。")

    # Case 2
    p.sec_title("案例 2：瓶蓋裂紋檢測（邊緣檢測）")
    p.case("瓶蓋完整性檢查",
           "檢測塑膠瓶蓋上的裂紋和斷裂。\n\n"
           "流程：\n"
           "1. ROI 提取：用霍夫圓檢測定位瓶蓋\n"
           "2. 前處理：CLAHE 增強對比，中值濾波降噪\n"
           "3. 邊緣檢測：Canny（低閾值 30, 80）\n"
           "4. 遮罩已知邊緣區域（瓶蓋邊緣、文字）\n"
           "5. 計算瓶蓋本體區域的殘餘邊緣像素\n"
           "6. 邊緣數 > 閾值 → 裂紋\n\n"
           "關鍵：使用良品瓶蓋的期望邊緣遮罩，抑制正常邊緣，僅隔離裂紋邊緣。")

    # Case 3
    p.sec_title("案例 3：PCB 缺陷檢測（模板差分 + 形態學）")
    p.case("PCB 板品質控制",
           "檢測缺件、焊橋、斷線。\n\n"
           "方法 A — 模板差分：\n"
           "1. 形狀匹配對齊測試 PCB 到黃金參考\n"
           "2. 仿射變換對齊\n"
           "3. 差分：diff = |test - ref|\n"
           "4. 自適應閾值 + 形態學清理\n"
           "5. 按位置分類瑕疵\n\n"
           "方法 B — 形態學開閉差分（無需參考影像）：\n"
           "1. 開運算差分 → 亮色缺陷（毛刺、短路）\n"
           "2. 閉運算差分 → 暗色缺陷（斷路、缺口）\n"
           "3. 合併並閾值分割（T=58）\n"
           "4. 15x15 閉運算遮罩去除非線路區假陽性\n\n"
           "PCB 缺陷類別：缺孔、鼠咬痕、斷路、短路、毛刺、多餘銅。")

    # Case 4
    p.sec_title("案例 4：LCD 雲狀缺陷（頻率域）")
    p.case("LCD 面板 Mura 檢測",
           "檢測 LCD 螢幕上的非均勻亮度區域。\n\n"
           "流程：\n"
           "1. 顯示均勻白色測試圖案\n"
           "2. FFT 計算\n"
           "3. 高通濾波移除直流分量\n"
           "4. IFFT 轉回空間域\n"
           "5. 正規化 + 閾值 + Blob 分析\n\n"
           "替代：擬合 2D 多項式曲面模擬理想背光，殘差即為瑕疵。")

    # Case 5
    p.sec_title("案例 5：藥丸檢測（形狀 + 顏色）")
    p.case("製藥藥丸品質檢測",
           "檢測缺角、顏色錯誤、形狀異常、污染。\n\n"
           "流程：\n"
           "1. HSV 閾值分割藥丸顏色\n"
           "2. 形狀驗證：橢圓擬合 + 圓度/長寬比\n"
           "3. 表面檢測：灰度均值/標準差異常\n"
           "4. 印字驗證：NCC < 0.7 → 印字缺失/模糊\n\n"
           "判定矩陣：\n"
           "- 顏色錯誤：H 通道超出範圍\n"
           "- 缺角：圓度 < 0.85 或面積過小\n"
           "- 污染：局部亮度異常 > 3 std\n"
           "- 印字異常：NCC < 0.7")

    # Case 6
    p.sec_title("案例 6：金屬表面刮痕（光度立體法）")
    p.case("拋光金屬表面檢測",
           "檢測拋光金屬零件上的細微刮痕。\n\n"
           "挑戰：刮痕在普通照明下幾乎無對比度。\n\n"
           "流程：\n"
           "1. 拍攝 4 張方向光影像（0/90/180/270 度）\n"
           "2. 光度立體法計算表面法向量\n"
           "3. 曲率圖：|d(nx)/dx| + |d(ny)/dy|\n"
           "4. 自適應閾值分割曲率圖\n"
           "5. Blob 分析：長度和方向\n\n"
           "嚴重程度分級：\n"
           "- 輕微：長度<2mm, 深度<0.01mm\n"
           "- 重大：長度>5mm 或 深度>0.05mm\n"
           "- 致命：A 級表面區域中的任何刮痕")

    # Case 7
    p.sec_title("案例 7：塑膠網孔驗證（Blob + 形態學）")
    p.case("塑膠網格檢測",
           "驗證網格中所有孔洞是否存在且形狀正確。\n\n"
           "流程：\n"
           "1. 前處理：高斯模糊 + CLAHE\n"
           "2. Otsu 閾值（孔洞為暗色）\n"
           "3. 反轉使孔洞為前景\n"
           "4. 開運算移除雜訊\n"
           "5. 連通域標記每個孔洞\n"
           "6. 每孔特徵：面積、圓度、質心\n"
           "7. 瑕疵判定：\n"
           "   - 缺孔：期望位置無 blob\n"
           "   - 堵孔：面積遠小於預期\n"
           "   - 變形孔：圓度低於閾值\n"
           "   - 多餘孔：非預期位置出現 blob")

    # Case 8
    p.sec_title("案例 8：吸盤瑕疵（變異模型）")
    p.case("橡膠吸盤品質檢測",
           "檢測氣泡、撕裂、變形。\n\n"
           "為何用變異模型：橡膠有自然紋理變化，固定閾值不可靠。\n\n"
           "流程：\n"
           "1. 訓練：拍攝 30 張對齊良品影像\n"
           "2. 計算 mean_model 和 std_model\n"
           "3. 測試：\n"
           "   a. 形狀匹配對齊\n"
           "   b. 計算 z-score\n"
           "   c. 閾值 k=3.5\n"
           "   d. 形態學清理\n"
           "   e. 按區域分類：\n"
           "      邊緣區 → 變形/撕裂\n"
           "      本體區 → 氣泡/污染\n"
           "      中心區 → 凹陷缺陷")

    # Case 9
    p.sec_title("案例 9：織布紋理缺陷（頻率域）")
    p.case("編織布料檢測",
           "檢測破洞、斷線、污漬和編織不規則。\n\n"
           "挑戰：規律編織模式主導影像，瑕疵隱藏其中。\n\n"
           "流程：\n"
           "1. FFT 分析\n"
           "2. 識別編織頻率的主要峰值\n"
           "3. 建立陷波濾波器抑制這些峰值\n"
           "4. IFFT：結果僅含非週期性內容\n"
           "5. 閾值 + Blob 分析\n\n"
           "缺陷類型：\n"
           "- 破洞：濾波後的大暗色 blob\n"
           "- 污漬：大亮色 blob\n"
           "- 斷線：細長 blob\n"
           "- 編織偏移：帶狀高能量")

    # Case 10
    p.sec_title("案例 10：焊縫檢測（卡尺量測）")
    p.case("自動焊縫品質驗證",
           "量測焊縫寬度、檢測間隙、驗證焊道輪廓。\n\n"
           "流程：\n"
           "1. ROI：提取焊縫區域\n"
           "2. Canny 邊緣檢測焊縫兩側\n"
           "3. 卡尺量測：\n"
           "   - 50~100 條垂直卡尺線\n"
           "   - 每條量測左右邊緣間距\n"
           "4. 輪廓分析：均值、標準差、最小/最大寬度\n"
           "5. 邊緣直線擬合：直線度偏差\n\n"
           "缺陷分類：\n"
           "- 底切：寬度 < 最小規格\n"
           "- 過量：寬度 > 最大規格\n"
           "- 間隙：寬度=0（缺焊）\n"
           "- 波浪：邊緣直線度 > 閾值\n"
           "- 氣孔：焊縫中的小暗點")


def write_ch13(p):
    p.ch_title("方法選擇決策指南")
    p.txt("選擇正確的方法取決於缺陷類型、產品特性和可用硬體。本章提供系統化的方法選擇指南。")

    # 13.1
    p.sec_title("缺陷類型 vs 方法矩陣")

    p.table(
        ["缺陷類型", "主要方法", "替代方法", "關鍵特徵"],
        [
            ["缺損材料", "Blob 分析", "模板差分", "面積/實心度"],
            ["裂紋/斷裂", "邊緣檢測", "光度立體", "邊緣密度"],
            ["刮痕", "頂帽/光度立體", "頻率域", "圓度低"],
            ["污漬/變色", "顏色閾值", "Blob 分析", "色相/飽和度"],
            ["表面凹坑", "Blob+圓度", "光度立體", "圓度>0.8"],
            ["尺寸偏差", "卡尺/擬合", "Blob 分析", "量測值"],
            ["紋理異常", "頻率域", "變異模型", "頻譜變化"],
            ["週期性圖案", "FFT+陷波", "模板差分", "頻率峰值"],
            ["表面拓撲", "光度立體", "結構光", "法向量"],
            ["隨機變異", "變異模型", "自編碼器", "Z-score"],
            ["複雜/多種", "ML 分類器", "深度學習", "學習特徵"],
        ]
    )

    # 13.2
    p.sec_title("打光與設置建議")

    p.table(
        ["產品類型", "打光方式", "相機", "方法"],
        [
            ["平面均勻", "漫射圓頂", "面掃描", "閾值+Blob"],
            ["反射金屬", "暗場照明", "面掃描", "光度/邊緣"],
            ["透明材料", "背光照明", "面掃描", "Blob 分析"],
            ["紋理表面", "漫射+角度", "面掃描", "頻率/變異"],
            ["3D 特徵", "結構光", "面掃描", "量測"],
            ["移動輸送", "頻閃+線光", "線掃描", "任何+同步"],
            ["多面檢測", "多方向光", "面掃描", "光度立體"],
            ["精細細節", "同軸/環形", "微距鏡頭", "邊緣/模板"],
        ]
    )

    # 13.3
    p.sec_title("完整決策流程圖")
    p.txt("按照以下流程選擇最佳方法：")

    p.code("步驟 1：瑕疵是形狀/尺寸異常嗎？\n"
           "  是 -> 已知良品參考？\n"
           "        是 -> 模板匹配 + 差分\n"
           "        否 -> Blob 分析 + 特徵分類\n"
           "  否 -> 步驟 2\n\n"
           "步驟 2：瑕疵是表面/紋理異常嗎？\n"
           "  是 -> 紋理是週期性/規律的？\n"
           "        是 -> 頻率域（FFT + 陷波）\n"
           "        否 -> 表面反光？\n"
           "              是 -> 光度立體法\n"
           "              否 -> 變異模型\n"
           "  否 -> 步驟 3\n\n"
           "步驟 3：瑕疵與尺寸/幾何相關嗎？\n"
           "  是 -> 卡尺 + 擬合 + 量測\n"
           "  否 -> 步驟 4\n\n"
           "步驟 4：瑕疵是顏色/污漬異常嗎？\n"
           "  是 -> HSV 色彩分割 + Blob\n"
           "  否 -> 步驟 5\n\n"
           "步驟 5：瑕疵類型多且難以定義規則？\n"
           "  是 -> 有標註數據？\n"
           "        是 -> CNN / SVM\n"
           "        否 -> 異常檢測（自編碼器）\n"
           "  否 -> 結合上述多種方法",
           lang="決策流程圖")

    p.txt("通用流程（適用大多數場景）：")

    p.code("1. 影像擷取\n"
           "   - 選擇適合缺陷類型的打光\n"
           "   - 設定曝光、增益、對焦\n\n"
           "2. 前處理\n"
           "   - 降噪（高斯/中值/雙邊）\n"
           "   - 對比度增強（CLAHE）\n"
           "   - ROI 提取 + 對齊\n\n"
           "3. 分割\n"
           "   - 閾值 / 邊緣檢測 /\n"
           "     頻率濾波 / 統計比較\n\n"
           "4. 後處理\n"
           "   - 形態學（先開後閉）\n"
           "   - 連通域標記 + 小區域移除\n\n"
           "5. 特徵提取\n"
           "   - 幾何/亮度/位置/形狀\n\n"
           "6. 分類決策\n"
           "   - 規則式 / ML式\n"
           "   - OK / 輕微 / 重大 / 報廢\n\n"
           "7. 報告\n"
           "   - 瑕疵位置標記\n"
           "   - 類型與嚴重程度\n"
           "   - 統計分析",
           lang="通用流程")

    # 快速參考表
    p.sec_title("快速參考：方法選擇總覽")

    p.table(
        ["情境", "推薦方法", "原因"],
        [
            ["有良品參考影像", "模板差分", "直接比較"],
            ["僅形狀瑕疵", "Blob + 特徵", "快速且規則簡單"],
            ["細微表面瑕疵", "光度立體法", "揭示表面拓撲"],
            ["週期性紋理", "FFT 濾波", "移除背景模式"],
            ["自然變異產品", "變異模型", "統計自適應"],
            ["多種瑕疵類型", "ML 分類器", "自動學習規則"],
            ["無瑕疵樣本", "自編碼器", "僅學習正常"],
            ["尺寸檢查", "卡尺 + 擬合", "精確量測"],
            ["顏色缺陷", "HSV 分割", "隔離顏色"],
            ["未知缺陷", "異常檢測", "捕捉任何異常"],
        ]
    )

    p.tip("實務上，最佳結果來自結合 2~3 種方法。例如：Blob 分析檢測大瑕疵 + 變異模型檢測細微瑕疵 + 量測驗證尺寸。永遠從最簡單的方法開始（閾值 + Blob），只在需要時增加複雜度。")


# ============================================================
# 第 14 章：比對、評分與瑕疵篩選
# ============================================================
def write_ch14(p):
    p.ch_title("比對、評分與瑕疵影像篩選")
    p.txt(
        "前面章節介紹了如何檢測瑕疵。但在實際產線中，檢測只是第一步。"
        "完成檢測後，還需要：\n"
        "1. 將測試件與良品進行比對，量化差異程度\n"
        "2. 根據差異計算瑕疵分數\n"
        "3. 依據分數和規則篩選出不良品\n"
        "4. 標記瑕疵位置並保存分類影像\n"
        "5. 產生檢測報告供後續分析\n\n"
        "本章涵蓋這整個「後處理 → 判定 → 輸出」的完整流程。"
    )

    # ==================================================================
    # 14.1 良品比對方法總覽
    # ==================================================================
    p.sec_title("良品比對方法總覽")

    p.txt(
        "比對的核心思想是：建立「良品應該長什麼樣」的基準，然後量化測試件與基準的偏離程度。"
        "根據基準的表示方式，分為三個層次："
    )

    p.sub_sec("層次一：像素級比對（Pixel-level）")
    p.txt(
        "直接將測試影像與參考影像逐像素相減。差異大的區域即為瑕疵。\n"
        "適用於外觀高度一致的產品（PCB、印刷標籤、IC 晶片）。\n"
        "前提：測試件必須先精確對齊到參考影像的座標系。"
    )

    p.formula("像素級差異分數",
              "diff(x,y) = |I_test(x,y) - I_ref(x,y)|\n\n"
              "defect_map = (diff > T)\n\n"
              "SSIM(x,y) = (2*mu_x*mu_y + C1)(2*sigma_xy + C2)\n"
              "          / (mu_x^2+mu_y^2+C1)(sig_x^2+sig_y^2+C2)\n\n"
              "  SSIM = 1 : 完全相同\n"
              "  SSIM ~ 0 : 完全不同",
              "SSIM（結構相似性指標）綜合考慮亮度、對比度和結構三個面向，"
              "比單純的像素差分更能反映人眼感知到的品質差異。")

    p.code("import cv2\nimport numpy as np\n"
           "from skimage.metrics import structural_similarity\n\n"
           "def pixel_compare(test, ref, thresh=30):\n"
           '    """像素級比對，返回差異圖和分數"""\n'
           "    # 絕對差分\n"
           "    diff = cv2.absdiff(test, ref)\n\n"
           "    # 二值化差異區域\n"
           "    _, mask = cv2.threshold(\n"
           "        diff, thresh, 255, cv2.THRESH_BINARY)\n\n"
           "    # SSIM 分數（越高越好）\n"
           "    score, ssim_map = structural_similarity(\n"
           "        test, ref, full=True)\n\n"
           "    # 差異面積百分比\n"
           "    diff_ratio = np.count_nonzero(mask) / mask.size\n\n"
           "    return {\n"
           "        'diff_img': diff,\n"
           "        'mask': mask,\n"
           "        'ssim': score,\n"
           "        'ssim_map': ssim_map,\n"
           "        'diff_ratio': diff_ratio\n"
           "    }")

    p.sub_sec("層次二：特徵級比對（Feature-level）")
    p.txt(
        "不直接比像素，而是提取特徵（面積、圓度、顏色等），"
        "與良品特徵的允許範圍進行比較。\n"
        "適用於產品有自然變異、無法逐像素比對的場景。"
    )

    p.formula("特徵偏離度",
              "deviation_i = |feature_i - mean_i| / std_i\n\n"
              "偏離度 > k (如 k=3) => 該特徵異常\n\n"
              "綜合偏離度 = sqrt(SUM(deviation_i^2) / N)",
              "每個特徵的偏離度類似 z-score。多特徵的綜合偏離度使用歐式距離。")

    p.code("def feature_compare(features, golden_stats):\n"
           '    """特徵級比對\n'
           "    features: dict, 測試件的特徵值\n"
           "    golden_stats: dict, 良品特徵的\n"
           '                 {name: (mean, std)} """\n'
           "    deviations = {}\n"
           "    for name, val in features.items():\n"
           "        mean, std = golden_stats[name]\n"
           "        std = max(std, 1e-6)  # 避免除零\n"
           "        dev = abs(val - mean) / std\n"
           "        deviations[name] = dev\n\n"
           "    # 綜合偏離度\n"
           "    vals = list(deviations.values())\n"
           "    overall = np.sqrt(\n"
           "        sum(v**2 for v in vals) / len(vals))\n\n"
           "    return deviations, overall")

    p.sub_sec("層次三：模型級比對（Model-level）")
    p.txt(
        "使用統計模型（變異模型）或神經網路（自編碼器）建立良品模型。"
        "測試件輸入模型後，模型輸出重建結果或統計分數，"
        "偏離越大表示越可能有瑕疵。\n\n"
        "- 變異模型：z-score 圖（第 11 章）\n"
        "- 自編碼器：重建誤差圖（第 10 章）\n"
        "- 這些方法的優勢是自動學習「正常」的定義。"
    )

    p.formula("自編碼器異常圖",
              "anomaly_map(x,y) = (I(x,y) - AE(I)(x,y))^2\n"
              "anomaly_score = mean(anomaly_map)\n\n"
              "歸一化: score_norm = anomaly_score / threshold",
              "score_norm > 1 表示該影像整體異常程度超過基準閾值。")

    p.table(
        ["比對層次", "適用場景", "精度", "對齊要求"],
        [
            ["像素級", "PCB/印刷品/IC", "最高", "必須精確對齊"],
            ["特徵級", "零件/食品/藥品", "中等", "粗略對齊即可"],
            ["模型級", "紋理/橡膠/複雜", "高", "需要對齊"],
        ]
    )

    # ==================================================================
    # 14.2 瑕疵評分機制
    # ==================================================================
    p.sec_title("瑕疵評分機制")

    p.txt(
        "檢測到瑕疵後，需要量化其嚴重程度。單一的「有/無」判定不足以支撐品質管控決策。"
        "評分機制將瑕疵轉換為可比較的數值分數。"
    )

    p.sub_sec("單一缺陷分數")
    p.formula("缺陷嚴重度分數",
              "score_i = w_area * (area_i / area_total)\n"
              "        + w_contrast * (mean_diff_i / 255)\n"
              "        + w_position * position_weight_i\n\n"
              "  w_area     = 面積權重 (如 0.4)\n"
              "  w_contrast = 對比度權重 (如 0.3)\n"
              "  w_position = 位置權重 (如 0.3)\n"
              "  position_weight: 關鍵區域=1, 邊緣=0.3",
              "不同維度加權。位置權重反映「瑕疵出現在哪裡」的重要性——"
              "A 級表面瑕疵比邊角瑕疵嚴重得多。")

    p.sub_sec("多缺陷聚合分數")
    p.formula("整件評分",
              "Score_part = max(score_i)           -- 最嚴重的\n"
              "  或\n"
              "Score_part = SUM(score_i)            -- 累加\n"
              "  或\n"
              "Score_part = SUM(score_i * w_type_i) -- 類型加權\n\n"
              "歸一化到 [0, 100]:\n"
              "Score_norm = min(Score_part / S_max * 100, 100)",
              "max 策略：一個致命瑕疵即判死。sum 策略：允許多個輕微瑕疵。根據品質標準選擇。")

    p.code("def defect_score(contours, gray, diff_img,\n"
           "                  position_map=None):\n"
           '    """計算每個缺陷和整件的瑕疵分數\n'
           "    contours:     瑕疵輪廓列表\n"
           "    gray:         原始灰度影像\n"
           "    diff_img:     差異影像\n"
           '    position_map: 位置權重圖 (可選)"""\n'
           "    total_area = gray.shape[0] * gray.shape[1]\n"
           "    scores = []\n\n"
           "    for cnt in contours:\n"
           "        area = cv2.contourArea(cnt)\n"
           "        mask_i = np.zeros_like(gray)\n"
           "        cv2.drawContours(\n"
           "            mask_i, [cnt], -1, 255, -1)\n\n"
           "        # 面積比\n"
           "        area_ratio = area / total_area\n\n"
           "        # 對比度（差異均值）\n"
           "        mean_diff = cv2.mean(\n"
           "            diff_img, mask=mask_i)[0] / 255\n\n"
           "        # 位置權重\n"
           "        if position_map is not None:\n"
           "            pos_w = cv2.mean(\n"
           "                position_map, mask=mask_i)[0]\n"
           "        else:\n"
           "            pos_w = 1.0\n\n"
           "        s = (0.4*area_ratio + 0.3*mean_diff\n"
           "             + 0.3*pos_w)\n"
           "        scores.append({\n"
           "            'area': area,\n"
           "            'area_ratio': area_ratio,\n"
           "            'contrast': mean_diff,\n"
           "            'pos_weight': pos_w,\n"
           "            'score': s\n"
           "        })\n\n"
           "    # 整件分數（取最嚴重）\n"
           "    part_score = max(\n"
           "        (d['score'] for d in scores),\n"
           "        default=0)\n"
           "    score_100 = min(part_score * 1000, 100)\n\n"
           "    return scores, score_100")

    # ==================================================================
    # 14.3 篩選判定與分級
    # ==================================================================
    p.sec_title("篩選判定與分級")

    p.txt(
        "有了分數後，需要設定判定規則將產品分類。常見策略："
    )

    p.sub_sec("二元判定（OK / NG）")
    p.formula("簡單閾值判定",
              "result = 'OK'  if score < T_ng  else 'NG'\n\n"
              "或基於缺陷數量:\n"
              "result = 'OK'  if defect_count == 0  else 'NG'",
              "最簡單的判定。適用於零容忍的高精度產品。")

    p.sub_sec("多級分類")

    p.table(
        ["等級", "分數範圍", "定義", "處置"],
        [
            ["OK", "0 ~ 10", "無瑕疵或極輕微", "正常出貨"],
            ["輕微 (Minor)", "10 ~ 40", "可接受的小瑕疵", "降級出貨/標記"],
            ["重大 (Major)", "40 ~ 70", "明顯瑕疵", "返工/降級"],
            ["報廢 (Scrap)", "70 ~ 100", "嚴重瑕疵", "報廢處理"],
        ]
    )

    p.code("def classify_part(score, defects,\n"
           "                   rules=None):\n"
           '    """多級品質判定\n'
           "    score:   整件分數 [0,100]\n"
           "    defects: 缺陷列表\n"
           '    rules:   自訂規則 (可選)"""\n'
           "    if rules is None:\n"
           "        rules = {\n"
           "            'scrap_score': 70,\n"
           "            'major_score': 40,\n"
           "            'minor_score': 10,\n"
           "            'max_defects': 5,\n"
           "            'max_single_area': 500,\n"
           "        }\n\n"
           "    # 檢查致命條件\n"
           "    for d in defects:\n"
           "        if d['area'] > rules['max_single_area']:\n"
           "            return 'SCRAP', '單一瑕疵面積過大'\n\n"
           "    if len(defects) > rules['max_defects']:\n"
           "        return 'MAJOR', '瑕疵數量過多'\n\n"
           "    # 分數判定\n"
           "    if score >= rules['scrap_score']:\n"
           "        return 'SCRAP', f'分數={score:.1f}'\n"
           "    elif score >= rules['major_score']:\n"
           "        return 'MAJOR', f'分數={score:.1f}'\n"
           "    elif score >= rules['minor_score']:\n"
           "        return 'MINOR', f'分數={score:.1f}'\n"
           "    else:\n"
           "        return 'OK', f'分數={score:.1f}'")

    p.sub_sec("動態閾值（SPC 統計製程控制）")
    p.txt(
        "固定閾值可能不適合所有批次。使用 SPC 控制圖動態調整閾值：\n"
        "- 計算近 N 批次的平均分數和標準差\n"
        "- UCL（上控制限）= mean + 3*std\n"
        "- 超過 UCL 的批次觸發警報\n"
        "- 連續多個批次趨近 UCL 表示製程飄移"
    )

    p.formula("SPC 控制限",
              "UCL = mu + 3*sigma    (上控制限)\n"
              "CL  = mu              (中心線)\n"
              "LCL = mu - 3*sigma    (下控制限)\n\n"
              "mu, sigma 來自近 N 個批次的統計",
              "基於常態分佈，99.7% 的正常值落在 UCL/LCL 之間。超出即為異常。")

    # ==================================================================
    # 14.4 瑕疵影像標記與輸出
    # ==================================================================
    p.sec_title("瑕疵影像標記與輸出")

    p.txt(
        "檢測結果需要可視化，供品管人員複核。標記方式包括：\n"
        "- 外接矩形框（Bounding Box）：最常用，快速定位\n"
        "- 輪廓描繪：精確顯示瑕疵形狀\n"
        "- 熱力圖疊加：顯示差異強度分佈\n"
        "- 瑕疵區域裁剪：放大瑕疵細節\n"
        "- 綜合報告圖：原圖 + 標記 + 文字資訊"
    )

    p.code("def annotate_defects(img, contours, scores,\n"
           "                      grade):\n"
           '    """在影像上標記所有瑕疵\n'
           "    img:      原始彩色影像\n"
           "    contours: 瑕疵輪廓列表\n"
           "    scores:   每個瑕疵的分數 dict 列表\n"
           '    grade:    整件判定等級"""\n'
           "    result = img.copy()\n\n"
           "    # 等級顏色\n"
           "    colors = {\n"
           "        'OK':    (0, 200, 0),\n"
           "        'MINOR': (0, 200, 255),\n"
           "        'MAJOR': (0, 100, 255),\n"
           "        'SCRAP': (0, 0, 255),\n"
           "    }\n"
           "    color = colors.get(grade, (0,0,255))\n\n"
           "    for cnt, sc in zip(contours, scores):\n"
           "        # 畫輪廓\n"
           "        cv2.drawContours(\n"
           "            result, [cnt], -1, color, 2)\n\n"
           "        # 畫外接矩形\n"
           "        x,y,w,h = cv2.boundingRect(cnt)\n"
           "        cv2.rectangle(result,\n"
           "            (x-3,y-3), (x+w+3,y+h+3),\n"
           "            color, 1)\n\n"
           "        # 標記分數\n"
           "        label = f\"{sc['score']:.3f}\"\n"
           "        cv2.putText(result, label,\n"
           "            (x, y-8),\n"
           "            cv2.FONT_HERSHEY_SIMPLEX,\n"
           "            0.4, color, 1)\n\n"
           "    # 整件判定標記（左上角）\n"
           "    cv2.putText(result, grade,\n"
           "        (10, 30),\n"
           "        cv2.FONT_HERSHEY_SIMPLEX,\n"
           "        1.0, color, 2)\n\n"
           "    return result")

    p.sub_sec("熱力圖疊加")
    p.code("def overlay_heatmap(img, diff_img, alpha=0.5):\n"
           '    """將差異圖以熱力圖形式疊加到原圖"""\n'
           "    # 歸一化差異圖到 0-255\n"
           "    norm = cv2.normalize(\n"
           "        diff_img, None, 0, 255,\n"
           "        cv2.NORM_MINMAX).astype(np.uint8)\n\n"
           "    # 套用色彩映射\n"
           "    heatmap = cv2.applyColorMap(\n"
           "        norm, cv2.COLORMAP_JET)\n\n"
           "    # 確保原圖是彩色\n"
           "    if len(img.shape) == 2:\n"
           "        img_c = cv2.cvtColor(\n"
           "            img, cv2.COLOR_GRAY2BGR)\n"
           "    else:\n"
           "        img_c = img.copy()\n\n"
           "    # 疊加\n"
           "    overlay = cv2.addWeighted(\n"
           "        img_c, 1-alpha, heatmap, alpha, 0)\n"
           "    return overlay")

    p.sub_sec("瑕疵區域裁剪")
    p.code("def crop_defects(img, contours, margin=20):\n"
           '    """裁剪每個瑕疵區域並保存"""\n'
           "    crops = []\n"
           "    h, w = img.shape[:2]\n"
           "    for i, cnt in enumerate(contours):\n"
           "        x,y,bw,bh = cv2.boundingRect(cnt)\n"
           "        # 加邊距\n"
           "        x1 = max(0, x - margin)\n"
           "        y1 = max(0, y - margin)\n"
           "        x2 = min(w, x + bw + margin)\n"
           "        y2 = min(h, y + bh + margin)\n"
           "        crop = img[y1:y2, x1:x2]\n"
           "        crops.append(crop)\n"
           "    return crops")

    p.sub_sec("綜合檢測報告影像")
    p.code("def make_report_image(original, annotated,\n"
           "                       diff_img, heatmap,\n"
           "                       grade, score, info):\n"
           '    """拼接 4 宮格報告影像"""\n'
           "    h, w = original.shape[:2]\n"
           "    # 統一轉為彩色\n"
           "    imgs = []\n"
           "    for im in [original, annotated,\n"
           "               diff_img, heatmap]:\n"
           "        if len(im.shape) == 2:\n"
           "            im = cv2.cvtColor(\n"
           "                im, cv2.COLOR_GRAY2BGR)\n"
           "        im = cv2.resize(im, (w//2, h//2))\n"
           "        imgs.append(im)\n\n"
           "    # 2x2 拼接\n"
           "    top = np.hstack([imgs[0], imgs[1]])\n"
           "    bot = np.hstack([imgs[2], imgs[3]])\n"
           "    report = np.vstack([top, bot])\n\n"
           "    # 加標題\n"
           "    cv2.putText(report,\n"
           "        f'{grade} | Score:{score:.1f}',\n"
           "        (10, 25),\n"
           "        cv2.FONT_HERSHEY_SIMPLEX,\n"
           "        0.7, (255,255,255), 2)\n\n"
           "    return report")

    # ==================================================================
    # 14.5 批次篩選流程
    # ==================================================================
    p.sec_title("批次篩選流程（完整範例）")

    p.txt(
        "以下是完整的端到端批次篩選流程，將所有步驟整合為一個可直接運行的函數。"
        "流程：遍歷資料夾中的影像 → 前處理 → 檢測 → 評分 → 分級 → 標記 → 分類保存 → 輸出報告。"
    )

    p.code("import os, csv, glob, shutil\n\n"
           "def batch_inspect(input_dir, ref_img_path,\n"
           "                   output_dir, thresh=30):\n"
           '    """完整的批次瑕疵篩選流程\n'
           "    input_dir:    待檢影像資料夾\n"
           "    ref_img_path: 良品參考影像路徑\n"
           "    output_dir:   輸出根目錄\n"
           '    thresh:       差異閾值"""\n\n'
           "    # 讀取參考影像\n"
           "    ref = cv2.imread(ref_img_path, 0)\n\n"
           "    # 建立輸出目錄結構\n"
           "    for grade in ['OK','MINOR','MAJOR','SCRAP']:\n"
           "        os.makedirs(\n"
           "            os.path.join(output_dir, grade),\n"
           "            exist_ok=True)\n"
           "    os.makedirs(\n"
           "        os.path.join(output_dir, 'reports'),\n"
           "        exist_ok=True)\n\n"
           "    # CSV 報告\n"
           "    csv_path = os.path.join(\n"
           "        output_dir, 'inspection_report.csv')\n"
           "    csv_f = open(csv_path, 'w', newline='')\n"
           "    writer = csv.writer(csv_f)\n"
           "    writer.writerow([\n"
           "        'filename', 'grade', 'score',\n"
           "        'defect_count', 'max_area', 'reason'\n"
           "    ])\n\n"
           "    # 統計\n"
           "    stats = {'OK':0, 'MINOR':0,\n"
           "             'MAJOR':0, 'SCRAP':0}\n\n"
           "    files = sorted(glob.glob(\n"
           "        os.path.join(input_dir, '*.png'))\n"
           "        + glob.glob(\n"
           "            os.path.join(input_dir, '*.jpg')))\n\n"
           "    for fpath in files:\n"
           "        fname = os.path.basename(fpath)\n"
           "        img = cv2.imread(fpath)\n"
           "        gray = cv2.cvtColor(\n"
           "            img, cv2.COLOR_BGR2GRAY)\n\n"
           "        # --- 1. 對齊（此處簡化） ---\n"
           "        aligned = gray  # 實際需做配準\n\n"
           "        # --- 2. 差異比對 ---\n"
           "        diff = cv2.absdiff(aligned, ref)\n"
           "        _, mask = cv2.threshold(\n"
           "            diff, thresh, 255,\n"
           "            cv2.THRESH_BINARY)\n\n"
           "        # --- 3. 形態學清理 ---\n"
           "        k = cv2.getStructuringElement(\n"
           "            cv2.MORPH_ELLIPSE, (5,5))\n"
           "        mask = cv2.morphologyEx(\n"
           "            mask, cv2.MORPH_OPEN, k)\n"
           "        mask = cv2.morphologyEx(\n"
           "            mask, cv2.MORPH_CLOSE, k)\n\n"
           "        # --- 4. 找輪廓 + 評分 ---\n"
           "        cnts, _ = cv2.findContours(\n"
           "            mask, cv2.RETR_EXTERNAL,\n"
           "            cv2.CHAIN_APPROX_SIMPLE)\n"
           "        cnts = [c for c in cnts\n"
           "                if cv2.contourArea(c) > 30]\n\n"
           "        scores, score_100 = defect_score(\n"
           "            cnts, gray, diff)\n\n"
           "        # --- 5. 分級 ---\n"
           "        grade, reason = classify_part(\n"
           "            score_100, scores)\n\n"
           "        # --- 6. 標記 ---\n"
           "        annotated = annotate_defects(\n"
           "            img, cnts, scores, grade)\n\n"
           "        # --- 7. 保存到對應目錄 ---\n"
           "        dst = os.path.join(\n"
           "            output_dir, grade, fname)\n"
           "        cv2.imwrite(dst, annotated)\n\n"
           "        # --- 8. 報告影像 ---\n"
           "        heat = overlay_heatmap(img, diff)\n"
           "        report = make_report_image(\n"
           "            img, annotated, diff, heat,\n"
           "            grade, score_100, '')\n"
           "        rpt_path = os.path.join(\n"
           "            output_dir, 'reports',\n"
           "            fname.replace('.','_rpt.'))\n"
           "        cv2.imwrite(rpt_path, report)\n\n"
           "        # --- 9. CSV 記錄 ---\n"
           "        max_a = max(\n"
           "            (s['area'] for s in scores),\n"
           "            default=0)\n"
           "        writer.writerow([\n"
           "            fname, grade,\n"
           "            f'{score_100:.1f}',\n"
           "            len(cnts), max_a, reason\n"
           "        ])\n"
           "        stats[grade] += 1\n\n"
           "    csv_f.close()\n"
           "    return stats",
           lang="python")

    p.case("PCB 批次篩選實例",
           "場景：100 張 PCB 影像需要自動篩選分類。\n\n"
           "1. 準備良品參考影像（golden_ref.png）\n"
           "2. 將待檢影像放入 input/ 資料夾\n"
           "3. 執行 batch_inspect('input/', 'golden_ref.png', 'output/')\n"
           "4. 輸出結構：\n"
           "   output/OK/     -> 良品影像（帶綠色標記）\n"
           "   output/MINOR/  -> 輕微瑕疵（帶黃色標記）\n"
           "   output/MAJOR/  -> 重大瑕疵（帶橙色標記）\n"
           "   output/SCRAP/  -> 報廢品（帶紅色標記）\n"
           "   output/reports/-> 4 宮格報告影像\n"
           "   output/inspection_report.csv -> 彙總表\n\n"
           "CSV 範例：\n"
           "filename, grade, score, defect_count, max_area, reason\n"
           "pcb_001.png, OK, 2.3, 0, 0, 分數=2.3\n"
           "pcb_042.png, MAJOR, 55.1, 3, 820, 分數=55.1\n"
           "pcb_099.png, SCRAP, 88.7, 1, 2400, 單一瑕疵面積過大")

    # ==================================================================
    # 14.6 常見判定策略與陷阱
    # ==================================================================
    p.sec_title("常見判定策略與陷阱")

    p.table(
        ["場景", "推薦策略", "閾值設定建議"],
        [
            ["零容忍高精密", "二元 OK/NG", "defect_count=0"],
            ["消費電子外觀", "多級分級", "分數 10/40/70"],
            ["食品/藥品", "二元 + 法規", "面積+位置"],
            ["汽車零件", "嚴格多級", "依 IATF 標準"],
            ["半導體晶圓", "分區域判定", "各區不同閾值"],
            ["紡織品", "允許輕微", "分數 20/60/85"],
        ]
    )

    p.tip(
        "閾值設定方法：\n"
        "1. 收集 100+ 張已知良品和已知瑕疵品的影像\n"
        "2. 對所有影像計算分數\n"
        "3. 繪製分數分佈直方圖\n"
        "4. 找到良品和瑕疵品的分數分界點\n"
        "5. 設定閾值在分界點附近，留適當安全餘量\n"
        "6. 計算 Precision / Recall，調整至平衡\n"
        "7. 定期用新數據驗證並更新閾值"
    )

    p.warn(
        "常見錯誤：\n"
        "1. 閾值設太嚴：大量良品被誤判為瑕疵（假陽性高），增加返工成本\n"
        "2. 閾值設太鬆：真正的瑕疵品流出（假陰性高），影響客戶品質\n"
        "3. 忽略位置資訊：同樣大小的瑕疵在不同位置嚴重程度可能完全不同\n"
        "4. 使用固定閾值不更新：材料批次變化或設備老化會導致閾值失效\n"
        "5. 僅依賴單一特徵：用多特徵組合判定才穩健"
    )

    p.sub_sec("評估指標")
    p.formula("Precision 與 Recall",
              "Precision = TP / (TP + FP)\n"
              "  檢出的瑕疵中真正有瑕疵的比例\n\n"
              "Recall = TP / (TP + FN)\n"
              "  所有瑕疵中被成功檢出的比例\n\n"
              "F1 = 2 * Precision * Recall\n"
              "   / (Precision + Recall)",
              "TP=真陽性, FP=假陽性, FN=假陰性。工業檢測通常優先確保高 Recall（不漏檢），允許適度的低 Precision（可接受少量誤檢）。")

    p.code("def evaluate(predictions, ground_truth):\n"
           '    """評估檢測系統表現"""\n'
           "    tp = sum(1 for p, g in\n"
           "             zip(predictions, ground_truth)\n"
           "             if p == 'NG' and g == 'NG')\n"
           "    fp = sum(1 for p, g in\n"
           "             zip(predictions, ground_truth)\n"
           "             if p == 'NG' and g == 'OK')\n"
           "    fn = sum(1 for p, g in\n"
           "             zip(predictions, ground_truth)\n"
           "             if p == 'OK' and g == 'NG')\n\n"
           "    precision = tp/(tp+fp) if tp+fp else 0\n"
           "    recall = tp/(tp+fn) if tp+fn else 0\n"
           "    f1 = (2*precision*recall /\n"
           "          (precision+recall)\n"
           "          if precision+recall else 0)\n\n"
           "    return {\n"
           "        'precision': precision,\n"
           "        'recall': recall,\n"
           "        'f1': f1,\n"
           "        'escape_rate': fn/(tp+fn)\n"
           "                       if tp+fn else 0\n"
           "    }")
