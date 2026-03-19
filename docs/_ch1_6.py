# -*- coding: utf-8 -*-
"""Chapter 1-6 content, cover, TOC"""

def write_cover_and_toc(p):
    p.add_page()
    p.ln(30)
    p.set_font(p.F, "B", 28)
    p.set_text_color(0, 50, 140)
    p.cell(0, 14, "電腦視覺", align="C", new_x="LMARGIN", new_y="NEXT")
    p.cell(0, 14, "瑕疵檢測教程", align="C", new_x="LMARGIN", new_y="NEXT")
    p.ln(3)
    p.set_font(p.F, "B", 18)
    p.set_text_color(0, 90, 150)
    p.cell(0, 12, "完整技術手冊", align="C", new_x="LMARGIN", new_y="NEXT")
    p.ln(4)
    p.set_draw_color(0, 90, 180)
    p.set_line_width(0.8)
    p.line(50, p.get_y(), 160, p.get_y())
    p.ln(8)
    p.set_font(p.F, "", 11)
    p.set_text_color(80, 80, 80)
    p.cell(0, 7, "方法原理 | 數學公式 | 處理流程 | 實際案例", align="C", new_x="LMARGIN", new_y="NEXT")
    p.cell(0, 7, "從基礎到工業應用的完整指南", align="C", new_x="LMARGIN", new_y="NEXT")
    p.ln(15)
    p.set_font(p.F, "", 9)
    p.set_text_color(130, 130, 130)
    p.cell(0, 6, "涵蓋：影像前處理、相機校正、光源選型、色彩檢測、Blob 分析、", align="C", new_x="LMARGIN", new_y="NEXT")
    p.cell(0, 6, "邊緣檢測、頻率域、深度學習、3D 檢測、即時優化、25 章完整指南", align="C", new_x="LMARGIN", new_y="NEXT")
    p.ln(8)
    p.cell(0, 5, "參考：CSDN 工業視覺瑕疵檢測系列 / PCB 形態學檢測方法", align="C", new_x="LMARGIN", new_y="NEXT")

    # TOC
    p.add_page()
    p.set_font(p.F, "B", 17)
    p.set_text_color(0, 50, 140)
    p.cell(0, 11, "目錄", new_x="LMARGIN", new_y="NEXT")
    p.set_draw_color(0, 90, 180)
    p.line(10, p.get_y(), 200, p.get_y())
    p.ln(5)

    toc = [
        ("第 1 章", "影像前處理基礎"),
        ("  1.1", "灰度轉換與色彩空間"), ("  1.2", "降噪與濾波"), ("  1.3", "對比度增強"), ("  1.4", "幾何變換（仿射）"),
        ("第 2 章", "閾值分割"),
        ("  2.1", "全域閾值（Otsu 法）"), ("  2.2", "自適應閾值"), ("  2.3", "多層閾值"),
        ("第 3 章", "形態學運算"),
        ("  3.1", "腐蝕與膨脹"), ("  3.2", "開運算與閉運算"), ("  3.3", "開閉差分缺陷檢測"), ("  3.4", "頂帽與黑帽變換"),
        ("第 4 章", "邊緣檢測"),
        ("  4.1", "Sobel / Prewitt / Roberts"), ("  4.2", "Canny 邊緣檢測器"), ("  4.3", "高斯拉普拉斯（LoG）"),
        ("第 5 章", "Blob 分析與連通域"),
        ("  5.1", "連通域標記"), ("  5.2", "區域特徵提取"), ("  5.3", "基於特徵的分類"),
        ("第 6 章", "頻率域分析"),
        ("  6.1", "傅立葉變換（DFT/FFT）"), ("  6.2", "高通 / 低通 / 帶通濾波"), ("  6.3", "紋理缺陷檢測"),
        ("第 7 章", "模板匹配"),
        ("  7.1", "歸一化互相關（NCC）"), ("  7.2", "基於形狀的匹配"), ("  7.3", "差分影像法"),
        ("第 8 章", "光度立體法"),
        ("  8.1", "原理與打光配置"), ("  8.2", "表面法向量估計"), ("  8.3", "表面缺陷檢測"),
        ("第 9 章", "量測與擬合"),
        ("  9.1", "卡尺工具與亞像素邊緣"), ("  9.2", "直線/圓/橢圓擬合"), ("  9.3", "尺寸驗證"),
        ("第 10 章", "機器學習瑕疵分類"),
        ("  10.1", "特徵工程"), ("  10.2", "SVM 分類器"), ("  10.3", "深度學習異常檢測"),
        ("第 11 章", "變異模型（統計背景）"),
        ("  11.1", "訓練階段：均值與變異數模型"), ("  11.2", "檢測階段：統計比較"),
        ("第 12 章", "完整瑕疵檢測案例"),
        ("  12.1-12.10", "10 個工業案例"),
        ("第 13 章", "方法選擇決策指南"),
        ("  13.1", "缺陷類型 vs 方法矩陣"), ("  13.2", "打光建議"), ("  13.3", "完整決策流程圖"),
        ("第 14 章", "比對、評分與瑕疵影像篩選"),
        ("  14.1", "良品比對方法總覽"), ("  14.2", "瑕疵評分機制"), ("  14.3", "篩選判定與分級"),
        ("  14.4", "瑕疵影像標記與輸出"), ("  14.5", "批次篩選流程（完整範例）"), ("  14.6", "常見判定策略與陷阱"),
        ("第 15 章", "相機校正與像素-毫米轉換"),
        ("  15.1", "為什麼需要相機校正"), ("  15.2", "內部參數與畸變係數"), ("  15.3", "棋盤格校正流程"),
        ("  15.4", "像素到毫米轉換"), ("  15.5", "畸變校正與影像矯正"),
        ("第 16 章", "影像對齊與配準"),
        ("  16.1", "為什麼需要對齊"), ("  16.2", "基於特徵點的對齊"), ("  16.3", "基於輪廓的對齊"),
        ("  16.4", "基於相位相關的對齊"), ("  16.5", "多尺度配準策略"),
        ("第 17 章", "光源原理與選型"),
        ("  17.1", "光源的核心角色"), ("  17.2", "照明幾何詳解"), ("  17.3", "光源類型與特性"),
        ("  17.4", "瑕疵類型最佳打光"), ("  17.5", "打光實驗方法論"),
        ("第 18 章", "色彩瑕疵檢測"),
        ("  18.1", "色彩空間選擇"), ("  18.2", "色差計算與閾值"), ("  18.3", "色彩區域分割"),
        ("  18.4", "色彩一致性檢測"), ("  18.5", "多通道融合檢測"),
        ("第 19 章", "進階輪廓分析"),
        ("  19.1", "輪廓層次結構"), ("  19.2", "輪廓近似與形狀描述"), ("  19.3", "最小外接形狀"),
        ("  19.4", "輪廓距離與相似度"), ("  19.5", "缺陷輪廓分類策略"),
        ("第 20 章", "深度學習瑕疵檢測進階"),
        ("  20.1", "YOLO 目標檢測"), ("  20.2", "U-Net 語義分割"), ("  20.3", "PatchCore / PADIM"),
        ("  20.4", "遷移學習與微調"), ("  20.5", "資料增強策略"), ("  20.6", "模型評估與部署"),
        ("第 21 章", "進階分割方法"),
        ("  21.1", "分水嶺分割"), ("  21.2", "GrabCut 互動分割"), ("  21.3", "超像素分割（SLIC）"),
        ("  21.4", "區域生長分割"), ("  21.5", "語義分割應用"),
        ("第 22 章", "即時處理與效能優化"),
        ("  22.1", "效能瓶頸分析"), ("  22.2", "影像處理加速"), ("  22.3", "多執行緒管線化"),
        ("  22.4", "GPU 加速"), ("  22.5", "演算法選擇"), ("  22.6", "即時系統設計"),
        ("第 23 章", "系統穩健性與邊界案例"),
        ("  23.1", "環境變化的影響"), ("  23.2", "自動曝光與白平衡"), ("  23.3", "異常輸入處理"),
        ("  23.4", "假陽性與假陰性分析"), ("  23.5", "穩健性測試"), ("  23.6", "日誌與追溯"),
        ("第 24 章", "3D 檢測技術"),
        ("  24.1", "3D 檢測的必要性"), ("  24.2", "結構光三維重建"), ("  24.3", "雷射三角測量"),
        ("  24.4", "立體視覺"), ("  24.5", "點雲處理"), ("  24.6", "3D 瑕疵檢測"),
        ("第 25 章", "參數調校方法論"),
        ("  25.1", "系統化調校流程"), ("  25.2", "ROC 曲線與閾值選擇"), ("  25.3", "交叉驗證"),
        ("  25.4", "敏感度分析"), ("  25.5", "自動參數搜索"), ("  25.6", "參數版本管理"),
    ]
    for num, title in toc:
        if num.startswith("第"):
            if p._space_left() < 14:
                p.add_page()
            p.set_font(p.F, "B", 9.5)
            p.set_text_color(0, 50, 140)
            p.cell(0, 6, f"{num}：{title}", new_x="LMARGIN", new_y="NEXT")
        else:
            if p._space_left() < 8:
                p.add_page()
            p.set_font(p.F, "", 8.5)
            p.set_text_color(80, 80, 80)
            p.cell(0, 5, f"    {num}  {title}", new_x="LMARGIN", new_y="NEXT")


def write_ch1(p):
    p.ch_title("影像前處理基礎")
    p.txt("影像前處理是所有瑕疵檢測流程的基礎。工業相機拍攝的原始影像包含雜訊、不均勻照明和無關資訊。前處理的目標是將原始影像轉換為能最大化瑕疵可見性、最小化誤檢的形式。")

    # 1.1
    p.sec_title("灰度轉換與色彩空間")
    p.txt("多數工業瑕疵檢測使用灰度影像，因為瑕疵通常表現為亮度異常。從彩色轉灰度可將資料從 3 通道減為 1 通道，加速後續處理。")

    p.formula("RGB 轉灰度（亮度法）",
              "Y = 0.299 * R + 0.587 * G + 0.114 * B",
              "權重反映人眼感知靈敏度。綠色權重最高，因為人眼對綠光最為敏感。")

    p.formula("HSV 色彩空間轉換",
              "H = arctan2(sqrt(3)*(G-B), 2R-G-B)\n"
              "S = 1 - 3*min(R,G,B)/(R+G+B)\n"
              "V = max(R,G,B)",
              "HSV 將顏色（H）與亮度（V）分離，當瑕疵具有獨特顏色特徵時非常有用（如金屬上的鏽蝕、食品上的變色）。")

    p.code("import cv2\nimport numpy as np\n\n"
           "# 讀取彩色影像\nimg = cv2.imread('part.jpg')\n\n"
           "# 轉為灰度\ngray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n\n"
           "# 轉為 HSV 用於顏色缺陷檢測\nhsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)\n"
           "h, s, v = cv2.split(hsv)")

    p.tip("何時使用彩色 vs 灰度：如果瑕疵主要是顏色差異（污漬、鏽蝕、變色），使用 HSV 或 Lab。如果瑕疵是形狀、紋理或亮度差異，灰度即可且更快。")

    # 1.2
    p.sec_title("降噪與濾波")
    p.txt("相機感測器會引入雜訊，可能被誤判為瑕疵。濾波在降低雜訊的同時保留邊緣（瑕疵邊界）。關鍵取捨：濾波太少則雜訊殘留；濾波太多則模糊真正的瑕疵。")

    p.sub_sec("高斯濾波")
    p.formula("二維高斯核",
              "G(x,y) = (1/(2*pi*sigma^2)) * exp(-(x^2+y^2)/(2*sigma^2))",
              "sigma 控制模糊強度。sigma 越大平滑越多。核大小通常為 3x3、5x5 或 7x7。適用於一般高斯雜訊。")

    p.sub_sec("中值濾波")
    p.formula("中值濾波",
              "output(x,y) = median{ I(x+i, y+j) : (i,j) in 鄰域 }",
              "非線性濾波器。對椒鹽雜訊效果極佳，邊緣保留能力優於高斯。工業檢測中脈衝雜訊常見時首選。")

    p.sub_sec("雙邊濾波")
    p.formula("雙邊濾波",
              "I_f(x) = (1/W) * SUM[ I(xi)*f_r(|I(xi)-I(x)|)*g_s(|xi-x|) ]",
              "結合空間鄰近度（g_s）和亮度相似度（f_r）。在平坦區域平滑，同時保留邊緣。速度較慢但邊緣保留效果最佳。")

    p.code("# 降噪比較\n"
           "gaussian  = cv2.GaussianBlur(gray, (5,5), sigmaX=1.0)\n"
           "median    = cv2.medianBlur(gray, 5)\n"
           "bilateral = cv2.bilateralFilter(gray, d=9,\n"
           "                                sigmaColor=75, sigmaSpace=75)\n\n"
           "# 工業應用選擇：\n"
           "# 椒鹽雜訊   -> 中值濾波\n"
           "# 高斯雜訊   -> 高斯濾波\n"
           "# 需保留邊緣 -> 雙邊濾波")

    # 1.3
    p.sec_title("對比度增強（直方圖均衡化）")
    p.txt("低對比度影像難以區分瑕疵與背景。直方圖均衡化重新分配像素亮度以覆蓋完整範圍 [0, 255]，提升細微瑕疵的可見性。")

    p.formula("直方圖均衡化",
              "s_k = (L-1) * SUM(j=0..k)[ n_j / N ]\n\n"
              "  s_k = 輸入級別 k 的輸出亮度\n"
              "  L   = 亮度級別數 (256)\n"
              "  n_j = 亮度為 j 的像素數\n"
              "  N   = 總像素數",
              "本質上是直方圖的累積分佈函數（CDF），縮放到 [0, L-1]。")

    p.sub_sec("CLAHE - 對比度受限自適應直方圖均衡化")
    p.txt("全域直方圖均衡化可能過度放大雜訊。CLAHE 將影像分成小塊分別均衡化，並設定裁剪限制以防雜訊放大。這是工業檢測中首選的方法。")

    p.code("# 全域直方圖均衡化\n"
           "equalized = cv2.equalizeHist(gray)\n\n"
           "# CLAHE（推薦用於瑕疵檢測）\n"
           "clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))\n"
           "enhanced = clahe.apply(gray)")

    # 1.4
    p.sec_title("幾何變換（仿射變換）")
    p.txt("產線上的零件很少在完全相同的位置/方向。仿射變換將被檢測件對齊到參考座標系，使得能與模板進行逐像素比較。")

    p.formula("仿射變換矩陣",
              "[x']   [a  b  tx] [x]\n"
              "[y'] = [c  d  ty] [y]\n"
              "[1 ]   [0  0  1 ] [1]\n\n"
              "包含：旋轉、縮放、平移、剪切",
              "用於將測試影像對齊（配準）到黃金參考影像。通過匹配特徵點或已知幾何估計。")

    p.code("# 仿射對齊範例\n"
           "pts_src = np.float32([[50,50],[200,50],[50,200]])\n"
           "pts_dst = np.float32([[60,40],[210,55],[55,210]])\n"
           "M = cv2.getAffineTransform(pts_src, pts_dst)\n"
           "aligned = cv2.warpAffine(img, M,\n"
           "                         (img.shape[1], img.shape[0]))")


def write_ch2(p):
    p.ch_title("閾值分割")
    p.txt("閾值分割將灰度影像轉換為二值影像（瑕疵 vs 背景）。這是最基本的分割技術，也是大多數瑕疵檢測流程的第一步。選擇正確的閾值方法至關重要。")

    # 2.1
    p.sec_title("全域閾值 - Otsu 法")
    p.txt("Otsu 法通過最大化類間變異數自動找到最佳閾值。假設直方圖呈雙峰分佈（背景和前景/瑕疵兩個峰值）。")

    p.formula("Otsu 類間變異數",
              "sigma_B^2(t) = w0(t)*w1(t)*[mu0(t)-mu1(t)]^2\n\n"
              "  w0(t) = SUM(i=0..t)[p_i]       (類別 0 權重)\n"
              "  w1(t) = SUM(i=t+1..L-1)[p_i]   (類別 1 權重)\n"
              "  mu0   = SUM(i=0..t)[i*p_i]/w0   (類別 0 均值)\n"
              "  mu1   = SUM(i=t+1..L-1)[i*p_i]/w1\n\n"
              "  最佳閾值：t* = argmax{ sigma_B^2(t) }",
              "p_i 是亮度 i 的機率。使類間變異數最大的閾值最能分離兩組。")

    p.code("# Otsu 閾值分割\n"
           "ret, binary = cv2.threshold(gray, 0, 255,\n"
           "    cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n"
           "print(f'Otsu 閾值: {ret}')\n\n"
           "# 反向二值化（瑕疵為暗色時）\n"
           "ret, binary_inv = cv2.threshold(gray, 0, 255,\n"
           "    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)")

    p.warn("Otsu 法在直方圖非雙峰時會失效（如均勻紋理，或瑕疵佔影像比例極小時）。此時應使用自適應閾值或手動選擇閾值。")

    # 2.2
    p.sec_title("自適應 / 動態閾值")
    p.txt("當影像照明不均勻時（工業環境中非常常見），單一全域閾值會失敗。自適應閾值根據每個像素的鄰域計算局部閾值。")

    p.formula("自適應閾值",
              "T(x,y) = mean(鄰域(x,y)) - C\n\n"
              "Binary(x,y) = 255  若 I(x,y) > T(x,y)\n"
              "              0    否則",
              "C 是常數偏移量。鄰域通常為方形窗口。高斯加權均值優於簡單均值。")

    p.code("# 自適應閾值\n"
           "adapt_gauss = cv2.adaptiveThreshold(\n"
           "    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\n"
           "    cv2.THRESH_BINARY, blockSize=51, C=10)\n\n"
           "# 關鍵參數：\n"
           "# blockSize: 局部鄰域大小（須為奇數）\n"
           "#   小 blockSize -> 檢測細小瑕疵，雜訊多\n"
           "#   大 blockSize -> 檢測大瑕疵，可能遺漏小的\n"
           "# C: 偏移常數\n"
           "#   較大 C -> 靈敏度降低，誤檢減少")

    # 2.3
    p.sec_title("多層閾值")
    p.txt("當存在多個類別時（如背景、正常表面、輕微瑕疵、嚴重瑕疵），單一閾值不足。可使用多層 Otsu 或 K-means 聚類。")

    p.code("# 使用 K-means 多層分割\n"
           "Z = gray.reshape((-1,1)).astype(np.float32)\n"
           "criteria = (cv2.TERM_CRITERIA_EPS +\n"
           "            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)\n"
           "K = 3\n"
           "_, labels, centers = cv2.kmeans(\n"
           "    Z, K, None, criteria, 10,\n"
           "    cv2.KMEANS_RANDOM_CENTERS)\n"
           "seg = centers[labels.flatten()]\n"
           "seg = seg.reshape(gray.shape).astype(np.uint8)")


def write_ch3(p):
    p.ch_title("形態學運算")
    p.txt("閾值分割後的二值影像常包含雜訊（小的假陽性區域）和瑕疵區域中的空洞。形態學運算使用結構元素清理二值影像，移除雜訊、填補空洞、分離或連接區域。")

    # 3.1
    p.sec_title("腐蝕與膨脹")

    p.formula("腐蝕",
              "output(x,y) = min{ I(x+i,y+j) : (i,j) in B }",
              "縮小前景物體。移除小雜訊區域。可斷開弱連接的物體。")

    p.formula("膨脹",
              "output(x,y) = max{ I(x+i,y+j) : (i,j) in B }",
              "擴大前景物體。填補小空洞。連接鄰近的斷開區域。")

    p.code("# 結構元素\n"
           "k_rect = cv2.getStructuringElement(\n"
           "    cv2.MORPH_RECT, (5,5))\n"
           "k_ellipse = cv2.getStructuringElement(\n"
           "    cv2.MORPH_ELLIPSE, (5,5))\n\n"
           "# 腐蝕：移除小雜訊\n"
           "eroded = cv2.erode(binary, k_rect, iterations=1)\n"
           "# 膨脹：填補小間隙\n"
           "dilated = cv2.dilate(binary, k_rect, iterations=1)")

    # 3.2
    p.sec_title("開運算與閉運算")

    p.formula("開運算 = 先腐蝕後膨脹",
              "A o B = (A erode B) dilate B",
              "移除小的前景雜訊，同時保持較大物體的大小和形狀。用於清理閾值分割結果。")

    p.formula("閉運算 = 先膨脹後腐蝕",
              "A . B = (A dilate B) erode B",
              "填補前景物體中的小空洞和間隙。當瑕疵區域有內部空洞需要填補時使用。")

    p.code("# 開運算：移除雜訊\n"
           "opened = cv2.morphologyEx(binary,\n"
           "    cv2.MORPH_OPEN, k_rect)\n"
           "# 閉運算：填補空洞\n"
           "closed = cv2.morphologyEx(binary,\n"
           "    cv2.MORPH_CLOSE, k_rect)\n"
           "# 常見流程：先開後閉\n"
           "cleaned = cv2.morphologyEx(binary,\n"
           "    cv2.MORPH_OPEN, k_rect)\n"
           "cleaned = cv2.morphologyEx(cleaned,\n"
           "    cv2.MORPH_CLOSE, k_rect)")

    # 3.3
    p.sec_title("開閉差分缺陷檢測（PCB 方法）")
    p.txt("利用原始影像與形態學處理結果的差異來檢測瑕疵。此方法廣泛用於 PCB 檢測：\n"
          "- 開運算差分 = 原圖 - 開運算(原圖) -> 檢測亮色突出物（毛刺、短路、多餘銅）\n"
          "- 閉運算差分 = 閉運算(原圖) - 原圖 -> 檢測暗色缺口（斷路、缺損、鼠咬痕）\n"
          "- 所有缺陷 = 開運算缺陷 + 閉運算缺陷")

    p.formula("形態學缺陷提取",
              "Defect_bright = I - Opening(I, B)   (毛刺/短路)\n"
              "Defect_dark   = Closing(I, B) - I   (斷路/缺口)\n"
              "Defect_all    = Defect_bright + Defect_dark\n\n"
              "然後: threshold(Defect_all, T) -> 二值缺陷遮罩",
              "開運算會平滑掉亮色突出物，閉運算會填補暗色缺口。差異恰好隔離這些異常。")

    p.code("# PCB 開閉差分缺陷檢測\n"
           "el = cv2.getStructuringElement(\n"
           "    cv2.MORPH_RECT, (5,5))\n\n"
           "# 開運算差分 -> 亮色缺陷\n"
           "opened = cv2.morphologyEx(gray,\n"
           "    cv2.MORPH_OPEN, el)\n"
           "defect_bright = cv2.subtract(gray, opened)\n\n"
           "# 閉運算差分 -> 暗色缺陷\n"
           "closed = cv2.morphologyEx(gray,\n"
           "    cv2.MORPH_CLOSE, el)\n"
           "defect_dark = cv2.subtract(closed, gray)\n\n"
           "# 合併所有缺陷\n"
           "defect_all = cv2.add(defect_bright, defect_dark)\n"
           "_, mask = cv2.threshold(defect_all, 58, 255,\n"
           "    cv2.THRESH_BINARY)")

    p.tip("最佳化：原始方法可能在非線路區域產生假陽性。解決方案：用較大閉運算核（15x15）提取非線路區域，反轉後膨脹（11x11），然後從開運算結果中扣除，可大幅減少誤檢。")

    # 3.4
    p.sec_title("頂帽與黑帽變換")

    p.formula("頂帽變換",
              "TopHat(I) = I - Opening(I)",
              "提取暗背景上的亮色細節（小亮點瑕疵）。用於檢測均勻暗色表面上的亮色刮痕或斑點。")

    p.formula("黑帽變換",
              "BlackHat(I) = Closing(I) - I",
              "提取亮背景上的暗色細節（小暗點瑕疵）。用於檢測亮色表面上的暗色斑點、凹坑或孔洞。")

    p.code("# 頂帽：暗背景上的亮色缺陷\n"
           "tophat = cv2.morphologyEx(gray,\n"
           "    cv2.MORPH_TOPHAT,\n"
           "    cv2.getStructuringElement(\n"
           "        cv2.MORPH_RECT, (15,15)))\n\n"
           "# 黑帽：亮背景上的暗色缺陷\n"
           "blackhat = cv2.morphologyEx(gray,\n"
           "    cv2.MORPH_BLACKHAT,\n"
           "    cv2.getStructuringElement(\n"
           "        cv2.MORPH_RECT, (15,15)))")


def write_ch4(p):
    p.ch_title("邊緣檢測")
    p.txt("邊緣是不同亮度區域之間的邊界。在瑕疵檢測中，邊緣揭示裂紋、刮痕、缺損邊界和形狀變形。邊緣檢測計算影像亮度的梯度。")

    # 4.1
    p.sec_title("Sobel / Prewitt / Roberts 運算子")

    p.formula("Sobel 運算子（3x3）",
              "Gx = [-1  0 +1]   Gy = [-1 -2 -1]\n"
              "     [-2  0 +2]        [ 0  0  0]\n"
              "     [-1  0 +1]        [+1 +2 +1]\n\n"
              "梯度幅值: G = sqrt(Gx^2 + Gy^2)\n"
              "梯度方向: theta = arctan(Gy / Gx)",
              "Gx 檢測垂直邊緣，Gy 檢測水平邊緣。中心行/列的 2x 加權提供輕微平滑。")

    p.code("# Sobel 邊緣檢測\n"
           "sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)\n"
           "sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)\n"
           "mag = np.sqrt(sobelx**2 + sobely**2)\n"
           "mag = np.uint8(np.clip(mag, 0, 255))")

    # 4.2
    p.sec_title("Canny 邊緣檢測器")
    p.txt("Canny 檢測器是邊緣檢測的黃金標準。通過多階段處理產生細而連續的邊緣：\n"
          "1. 高斯平滑（降噪）\n"
          "2. 計算梯度幅值和方向（Sobel）\n"
          "3. 非極大值抑制（邊緣細化為 1 像素寬）\n"
          "4. 雙閾值分類（強邊緣/弱邊緣/無邊緣）\n"
          "5. 滯後邊緣追蹤（將弱邊緣連接到強邊緣）")

    p.formula("Canny 雙閾值",
              "強邊緣: G(x,y) >= T_high\n"
              "弱邊緣: T_low <= G(x,y) < T_high\n"
              "無邊緣: G(x,y) < T_low\n\n"
              "建議比例: T_high / T_low = 2:1 或 3:1",
              "強邊緣永遠保留。弱邊緣僅在連接到強邊緣時保留（滯後追蹤）。")

    p.code("# Canny 邊緣檢測\n"
           "edges = cv2.Canny(gray, 50, 150)\n\n"
           "# 自動閾值（基於中位數）\n"
           "v = np.median(gray)\n"
           "lo = int(max(0, 0.67 * v))\n"
           "hi = int(min(255, 1.33 * v))\n"
           "edges_auto = cv2.Canny(gray, lo, hi)")

    # 4.3
    p.sec_title("高斯拉普拉斯（LoG）")

    p.formula("高斯拉普拉斯",
              "LoG = Laplacian( Gaussian(I) )\n"
              "    = d^2G/dx^2 + d^2G/dy^2",
              "在零交叉點檢測邊緣。各向同性（檢測所有方向的邊緣）。sigma 控制檢測的邊緣尺度。")


def write_ch5(p):
    p.ch_title("Blob 分析與連通域")
    p.txt("Blob 分析是工業瑕疵檢測中最常用的方法。閾值分割後，連通的白色（或黑色）區域被識別為 'blob'。計算每個 blob 的幾何和亮度特徵，與驗收標準比較以分類為瑕疵或正常。")

    # 5.1
    p.sec_title("連通域標記")
    p.txt("連通域標記為每組連通的前景像素分配唯一標籤。\n"
          "- 4-連通：水平/垂直方向連接的像素\n"
          "- 8-連通：所有 8 個方向連接（含對角線）\n"
          "8-連通更常用，因為它更好地保留對角特徵。")

    p.code("# 連通域分析\n"
           "num, labels, stats, centroids = \\\n"
           "    cv2.connectedComponentsWithStats(\n"
           "        binary, connectivity=8)\n\n"
           "# stats 欄位: [left, top, w, h, area]\n"
           "# 按面積過濾以移除雜訊\n"
           "min_area = 100\n"
           "for i in range(1, num):  # 跳過 0（背景）\n"
           "    area = stats[i, cv2.CC_STAT_AREA]\n"
           "    if area < min_area:\n"
           "        labels[labels == i] = 0")

    # 5.2
    p.sec_title("區域特徵提取")
    p.txt("識別 blob 後，提取特徵用於分類。瑕疵檢測中的關鍵特徵：")

    p.formula("面積與周長",
              "面積 = 區域中的像素數\n"
              "周長 = 邊界像素數\n\n"
              "Area = cv2.contourArea(contour)\n"
              "Perimeter = cv2.arcLength(contour, True)",
              "面積區分大瑕疵與雜訊。周長有助於識別細長瑕疵（刮痕）。")

    p.formula("圓度（緊密度）",
              "Circularity = 4 * pi * Area / Perimeter^2\n\n"
              "完美圓形: Circularity = 1.0\n"
              "細長形狀: Circularity << 1.0",
              "圓度區分圓形瑕疵（凹坑、孔洞）和細長瑕疵（刮痕、裂紋）。")

    p.formula("長寬比、填充度、實心度",
              "AspectRatio = Width / Height\n"
              "Extent      = Area / BoundingRect_Area\n"
              "Solidity    = Area / ConvexHull_Area",
              "AspectRatio > 3：可能是刮痕或裂紋。低 Solidity：不規則形狀（破損邊緣）。")

    p.code("# 輪廓特徵提取\n"
           "contours, _ = cv2.findContours(\n"
           "    binary, cv2.RETR_EXTERNAL,\n"
           "    cv2.CHAIN_APPROX_SIMPLE)\n\n"
           "for cnt in contours:\n"
           "    area = cv2.contourArea(cnt)\n"
           "    peri = cv2.arcLength(cnt, True)\n"
           "    if peri == 0: continue\n"
           "    circ = 4*np.pi*area / (peri**2)\n"
           "    x,y,w,h = cv2.boundingRect(cnt)\n"
           "    ar = w/h if h > 0 else 0\n"
           "    hull = cv2.convexHull(cnt)\n"
           "    sol = area / cv2.contourArea(hull)\n\n"
           "    # 分類規則\n"
           "    if area < 50:      typ = '雜訊'\n"
           "    elif circ > 0.8:   typ = '凹坑/孔洞'\n"
           "    elif ar > 4:       typ = '刮痕'\n"
           "    else:              typ = '不規則瑕疵'")

    # 5.3
    p.sec_title("基於特徵的瑕疵分類")

    p.table(
        ["特徵", "低值含義", "高值含義", "對應瑕疵類型"],
        [
            ["面積", "< 50 px（雜訊）", "> 5000 px（重大）", "雜訊 / 重大瑕疵"],
            ["圓度", "< 0.3（細長）", "> 0.8（圓形）", "刮痕 / 孔洞"],
            ["長寬比", "~1.0（方形）", "> 5.0（細長）", "圓形 / 線性"],
            ["實心度", "< 0.5（不規則）", "> 0.9（實心）", "不規則 / 實心"],
        ]
    )


def write_ch6(p):
    p.ch_title("頻率域分析")
    p.txt("頻率域揭示空間域中不可見的週期性模式。織物、網格或電路板走線等紋理會產生強烈的頻率峰值。瑕疵破壞這些模式，在頻譜中表現為異常。")

    # 6.1
    p.sec_title("傅立葉變換（DFT/FFT）")

    p.formula("二維離散傅立葉變換",
              "F(u,v) = SUM_x SUM_y f(x,y) *\n"
              "         e^(-j*2*pi*(ux/M + vy/N))\n\n"
              "幅值: |F(u,v)| = sqrt(Re^2 + Im^2)\n"
              "相位: phi(u,v) = arctan(Im / Re)",
              "低頻（靠近中心）= 平滑區域、整體照明。高頻（遠離中心）= 邊緣、雜訊、細節。週期性模式 = 頻譜中的明顯峰值。")

    p.code("# 傅立葉變換用於紋理分析\n"
           "dft = cv2.dft(np.float32(gray),\n"
           "              flags=cv2.DFT_COMPLEX_OUTPUT)\n"
           "dft_shift = np.fft.fftshift(dft)\n"
           "mag = cv2.magnitude(\n"
           "    dft_shift[:,:,0], dft_shift[:,:,1])\n"
           "mag_log = np.log1p(mag)  # 對數尺度可視化")

    # 6.2
    p.sec_title("高通 / 低通 / 帶通濾波")

    p.formula("高斯低通濾波器（頻率域）",
              "H_LP(u,v) = exp(-D(u,v)^2 / (2*D0^2))\n"
              "D(u,v) = sqrt((u-M/2)^2 + (v-N/2)^2)\n"
              "D0 = 截止頻率",
              "移除高頻雜訊，保留整體結構。")

    p.formula("高斯高通濾波器",
              "H_HP(u,v) = 1 - H_LP(u,v)",
              "增強邊緣和細節。用於檢測紋理表面上的刮痕。")

    p.formula("帶阻/帶通濾波器",
              "帶阻：移除特定頻率帶\n"
              "帶通：H_BP = 1 - H_BR（僅保留該頻帶）",
              "帶阻用於移除週期性模式（如網格），以揭示隱藏在模式中的瑕疵。")

    p.code("# 頻率域高通濾波\n"
           "rows, cols = gray.shape\n"
           "crow, ccol = rows//2, cols//2\n"
           "D0 = 30  # 截止頻率\n"
           "u = np.arange(rows).reshape(-1,1) - crow\n"
           "v = np.arange(cols).reshape(1,-1) - ccol\n"
           "D = np.sqrt(u**2 + v**2)\n"
           "H_hp = 1 - np.exp(-D**2 / (2*D0**2))\n\n"
           "# 套用濾波\n"
           "dft_f = dft_shift.copy()\n"
           "dft_f[:,:,0] *= H_hp\n"
           "dft_f[:,:,1] *= H_hp\n"
           "# 反傅立葉\n"
           "f_is = np.fft.ifftshift(dft_f)\n"
           "result = cv2.idft(f_is)\n"
           "result = cv2.magnitude(\n"
           "    result[:,:,0], result[:,:,1])")

    # 6.3
    p.sec_title("紋理缺陷檢測應用")

    p.case("LCD 面板雲狀缺陷（Mura）檢測",
           "LCD 面板具有均勻背光。雲狀亮度不均（Mura 缺陷）是低頻變化。\n\n"
           "流程：\n"
           "1. 顯示均勻白色測試圖案\n"
           "2. 計算 FFT\n"
           "3. 套用高通濾波器（D0=5~10）移除直流分量\n"
           "4. IFFT 轉回空間域\n"
           "5. 正規化並拉伸對比度\n"
           "6. 閾值分割 + Blob 分析\n\n"
           "替代方案：擬合二維多項式曲面（模擬理想背光），從原始影像中減去，殘差即為 Mura 缺陷。")

    p.case("織布紋理缺陷檢測",
           "織布具有規律的編織模式，在頻譜中產生強週期峰值。瑕疵（破洞、斷線、污漬）破壞此模式。\n\n"
           "流程：\n"
           "1. 計算織布影像的 FFT\n"
           "2. 識別編織頻率的主要峰值\n"
           "3. 建立陷波濾波器抑制這些峰值\n"
           "4. 套用 IFFT\n"
           "5. 結果僅包含非週期性內容（瑕疵）\n"
           "6. 閾值分割 + Blob 分析")
