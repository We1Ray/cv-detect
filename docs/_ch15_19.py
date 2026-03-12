# -*- coding: utf-8 -*-
"""Chapter 15-19 content"""


def write_ch15(p):
    p.ch_title("相機校正與像素-毫米轉換")
    p.txt("在工業瑕疵檢測中，僅僅知道瑕疵的像素大小是不夠的。客戶的品質規格通常以毫米為單位（如：刮痕長度不得超過 0.5mm）。要將像素量測值轉換為真實世界的物理尺寸，必須進行相機校正。此外，所有光學鏡頭都會引入畸變，使直線變彎、圓形變橢圓，導致量測誤差。本章涵蓋完整的相機校正流程、畸變矯正以及像素到毫米的精確轉換。")

    # 15.1
    p.sec_title("為什麼需要相機校正")
    p.txt("相機並非完美的成像設備。從三維世界到二維影像的投影過程中，存在兩類主要問題：\n"
          "1. 透視投影畸變：針孔相機模型將三維點投影到二維平面，遠處物體看起來更小\n"
          "2. 鏡頭光學畸變：真實鏡頭的製造缺陷和光學特性導致影像變形\n\n"
          "針孔相機模型是最基本的相機數學模型。它假設所有光線通過一個無限小的孔（光心），直線投影到感測器平面上。真實相機使用透鏡聚焦光線，但在數學上可以近似為針孔模型加上畸變參數。")

    p.sub_sec("畸變類型")
    p.txt("鏡頭畸變分為兩大類：\n"
          "- 徑向畸變（Radial Distortion）：沿著離光心的徑向方向，直線變彎曲。桶形畸變使邊緣向外膨脹，枕形畸變使邊緣向內收縮。廣角鏡頭的桶形畸變尤為嚴重。\n"
          "- 切向畸變（Tangential Distortion）：由於透鏡與感測器平面不完全平行導致。影像某些區域看起來比實際更近或更遠。通常比徑向畸變小得多，但在高精度量測中不可忽略。")

    # 15.2
    p.sec_title("內部參數與畸變係數")
    p.txt("相機內部參數（Intrinsic Parameters）描述相機自身的光學特性，獨立於外部場景。核心參數包括：\n"
          "- fx, fy：焦距，以像素為單位。fx = f / pixel_width, fy = f / pixel_height。理想情況下 fx = fy，但感測器像素可能非正方形。\n"
          "- cx, cy：主點座標，光軸與感測器平面的交點。理想情況下在影像中心，但製造誤差會導致偏移。\n"
          "- k1, k2, k3：徑向畸變係數。k1 影響最大，k3 通常可忽略。\n"
          "- p1, p2：切向畸變係數。")

    p.formula("相機內部參數矩陣",
              "K = [fx   0  cx]\n"
              "    [ 0  fy  cy]\n"
              "    [ 0   0   1]\n\n"
              "3D 點投影到像素座標:\n"
              "  u = fx * (X/Z) + cx\n"
              "  v = fy * (Y/Z) + cy",
              "K 矩陣將歸一化相機座標轉換為像素座標。fx/fy 的單位是像素。cx/cy 通常接近影像中心 (W/2, H/2)。")

    p.formula("徑向畸變模型",
              "r^2 = x'^2 + y'^2\n\n"
              "x_d = x'*(1 + k1*r^2 + k2*r^4 + k3*r^6)\n"
              "y_d = y'*(1 + k1*r^2 + k2*r^4 + k3*r^6)\n\n"
              "(x', y') = 無畸變歸一化座標\n"
              "(x_d, y_d) = 有畸變歸一化座標\n"
              "r = 到光心的徑向距離",
              "k1 > 0 產生枕形畸變，k1 < 0 產生桶形畸變。大多數工業鏡頭 k1 為負值（桶形畸變）。")

    p.formula("切向畸變模型",
              "x_d = x' + [2*p1*x'*y' + p2*(r^2 + 2*x'^2)]\n"
              "y_d = y' + [p1*(r^2 + 2*y'^2) + 2*p2*x'*y']",
              "切向畸變來自透鏡與感測器不平行。在高精度量測（< 0.01mm）中必須考慮。")

    p.code("import cv2\nimport numpy as np\nimport glob\n\n"
           "# 畸變係數向量\n"
           "# dist_coeffs = [k1, k2, p1, p2, k3]\n"
           "# 例如：\n"
           "# dist = np.array([-0.2413, 0.0998,\n"
           "#                   0.0001, -0.0003, 0.0])\n\n"
           "# 完整投影流程（3D -> 2D 像素）\n"
           "def project_point(P_3d, K, dist, rvec, tvec):\n"
           '    """將 3D 點投影到像素座標"""\n'
           "    pts_2d, _ = cv2.projectPoints(\n"
           "        P_3d.reshape(-1,1,3).astype(np.float64),\n"
           "        rvec, tvec, K, dist)\n"
           "    return pts_2d.reshape(-1, 2)")

    # 15.3
    p.sec_title("棋盤格校正流程")
    p.txt("棋盤格（Checkerboard）校正是最常用的相機校正方法。原理：棋盤格的角點位置在物體座標系中是已知的（等間距網格），通過檢測多張不同角度的棋盤格影像中的角點，建立 3D-2D 對應關係，使用最小二乘法求解內部參數和畸變係數。")

    p.sub_sec("校正步驟")
    p.txt("1. 準備棋盤格校正板（建議 9x6 或 11x8 內角點，方格尺寸已知）\n"
          "2. 從不同角度、距離拍攝 15~30 張棋盤格影像\n"
          "3. 對每張影像檢測角點（findChessboardCorners）\n"
          "4. 精細化角點位置到亞像素精度（cornerSubPix）\n"
          "5. 建立 3D 物體點與 2D 影像點的對應\n"
          "6. 使用 calibrateCamera 求解所有參數\n"
          "7. 評估重投影誤差，驗證校正品質")

    p.code("# === 完整棋盤格相機校正流程 ===\nimport cv2\nimport numpy as np\nimport glob\n\n"
           "# 棋盤格參數\n"
           "BOARD_SIZE = (9, 6)  # 內角點數 (列, 行)\n"
           "SQUARE_SIZE = 25.0   # 方格邊長 (mm)\n\n"
           "# 建立物體座標點 (z=0 的平面)\n"
           "objp = np.zeros(\n"
           "    (BOARD_SIZE[0]*BOARD_SIZE[1], 3),\n"
           "    dtype=np.float32)\n"
           "objp[:, :2] = np.mgrid[\n"
           "    0:BOARD_SIZE[0],\n"
           "    0:BOARD_SIZE[1]].T.reshape(-1, 2)\n"
           "objp *= SQUARE_SIZE  # 換算為 mm\n\n"
           "# 收集所有影像的角點\n"
           "obj_points = []  # 3D 物體點\n"
           "img_points = []  # 2D 影像點\n\n"
           "images = glob.glob('calib_images/*.jpg')\n"
           "for fname in images:\n"
           "    img = cv2.imread(fname)\n"
           "    gray = cv2.cvtColor(\n"
           "        img, cv2.COLOR_BGR2GRAY)\n\n"
           "    # 尋找棋盤格角點\n"
           "    ret, corners = cv2.findChessboardCorners(\n"
           "        gray, BOARD_SIZE,\n"
           "        cv2.CALIB_CB_ADAPTIVE_THRESH +\n"
           "        cv2.CALIB_CB_NORMALIZE_IMAGE)\n\n"
           "    if ret:\n"
           "        obj_points.append(objp)\n\n"
           "        # 亞像素精細化\n"
           "        criteria = (\n"
           "            cv2.TERM_CRITERIA_EPS +\n"
           "            cv2.TERM_CRITERIA_MAX_ITER,\n"
           "            30, 0.001)\n"
           "        corners2 = cv2.cornerSubPix(\n"
           "            gray, corners, (11,11),\n"
           "            (-1,-1), criteria)\n"
           "        img_points.append(corners2)\n\n"
           "# 執行校正\n"
           "ret, K, dist, rvecs, tvecs = \\\n"
           "    cv2.calibrateCamera(\n"
           "        obj_points, img_points,\n"
           "        gray.shape[::-1], None, None)\n\n"
           "print(f'RMS 重投影誤差: {ret:.4f} pixels')\n"
           "print(f'內部參數矩陣 K:\\n{K}')\n"
           "print(f'畸變係數: {dist.ravel()}')\n\n"
           "# 計算每張影像的重投影誤差\n"
           "errors = []\n"
           "for i in range(len(obj_points)):\n"
           "    proj, _ = cv2.projectPoints(\n"
           "        obj_points[i], rvecs[i],\n"
           "        tvecs[i], K, dist)\n"
           "    err = cv2.norm(\n"
           "        img_points[i], proj,\n"
           "        cv2.NORM_L2) / len(proj)\n"
           "    errors.append(err)\n"
           "print(f'平均誤差: {np.mean(errors):.4f} px')\n"
           "print(f'最大誤差: {np.max(errors):.4f} px')\n\n"
           "# 保存校正結果\n"
           "np.savez('calibration.npz',\n"
           "         K=K, dist=dist,\n"
           "         rvecs=rvecs, tvecs=tvecs)")

    # 15.4
    p.sec_title("像素到毫米轉換")
    p.txt("校正完成後，需要建立像素與毫米之間的對應關係。轉換方式取決於應用場景：\n"
          "- 已知工作距離和感測器尺寸：直接計算像素物理尺寸\n"
          "- 使用校正結果：從焦距和工作距離推算\n"
          "- 實測法：拍攝已知尺寸的物體，計算比例因子\n\n"
          "在工業檢測中，通常物體到相機的距離固定（固定安裝），因此轉換比例是常數。")

    p.formula("基於感測器的像素尺寸",
              "pixel_size = sensor_size / resolution\n\n"
              "  sensor_size: 感測器物理尺寸 (mm)\n"
              "  resolution:  解析度 (像素)\n\n"
              "例: 1/2\" 感測器 (6.4mm x 4.8mm),\n"
              "    1280x960 解析度:\n"
              "    pixel_w = 6.4/1280 = 0.005 mm/pixel\n"
              "    pixel_h = 4.8/960  = 0.005 mm/pixel",
              "這是感測器上每個像素的物理大小。實際場景中物體上的對應尺寸還需考慮放大倍率。")

    p.formula("像素到毫米轉換（含放大倍率）",
              "mm = pixels * pixel_size / magnification\n\n"
              "magnification = focal_length / working_distance\n\n"
              "或更實用的方式：\n"
              "mm_per_pixel = FOV_mm / image_width_pixels\n\n"
              "  FOV_mm: 視場範圍 (mm)\n"
              "  image_width_pixels: 影像寬度 (像素)",
              "mm_per_pixel 是最常用的轉換係數。在固定安裝的工業相機中，此值為常數。推薦使用已知尺寸的校正件實測此值。")

    p.code("# === 像素到毫米轉換 ===\n\n"
           "# 方法一：基於感測器規格\n"
           "sensor_w_mm = 6.4     # 感測器寬度 (mm)\n"
           "img_w_px = 1280       # 影像寬度 (像素)\n"
           "focal_mm = 12.0       # 焦距 (mm)\n"
           "work_dist_mm = 300.0  # 工作距離 (mm)\n\n"
           "pixel_size = sensor_w_mm / img_w_px\n"
           "magnification = focal_mm / work_dist_mm\n"
           "mm_per_pixel = pixel_size / magnification\n"
           "print(f'mm/pixel = {mm_per_pixel:.4f}')\n\n"
           "# 方法二：實測法（推薦）\n"
           "# 拍攝已知尺寸的標準件\n"
           "known_length_mm = 50.0    # 標準件長度\n"
           "measured_length_px = 628  # 影像中像素數\n"
           "mm_per_pixel_real = (\n"
           "    known_length_mm / measured_length_px)\n"
           "print(f'實測 mm/pixel = '\n"
           "      f'{mm_per_pixel_real:.4f}')\n\n"
           "# 方法三：從校正結果計算\n"
           "# K[0,0] = fx (像素為單位的焦距)\n"
           "# mm_per_pixel = working_dist / fx\n"
           "fx = K[0, 0]\n"
           "mm_per_px_calib = work_dist_mm / fx\n"
           "print(f'校正 mm/pixel = '\n"
           "      f'{mm_per_px_calib:.4f}')\n\n"
           "# 應用：量測瑕疵的真實尺寸\n"
           "def measure_defect_mm(contour, mm_per_px):\n"
           '    """量測瑕疵的真實尺寸 (mm)"""\n'
           "    area_px = cv2.contourArea(contour)\n"
           "    peri_px = cv2.arcLength(contour, True)\n"
           "    rect = cv2.minAreaRect(contour)\n"
           "    w_px, h_px = rect[1]\n\n"
           "    return {\n"
           "        'area_mm2': area_px * mm_per_px**2,\n"
           "        'perimeter_mm': peri_px * mm_per_px,\n"
           "        'width_mm': w_px * mm_per_px,\n"
           "        'height_mm': h_px * mm_per_px,\n"
           "    }")

    # 15.5
    p.sec_title("畸變校正與影像矯正")
    p.txt("校正得到畸變係數後，可以將畸變影像矯正為無畸變影像。這對精確量測至關重要。OpenCV 提供兩種方式：直接 undistort 和使用映射表（remap）。映射表方式在批次處理時效率更高，因為映射表只需計算一次。")

    p.code("# === 畸變矯正 ===\n\n"
           "# 載入校正結果\n"
           "calib = np.load('calibration.npz')\n"
           "K = calib['K']\n"
           "dist = calib['dist']\n\n"
           "img = cv2.imread('test_image.jpg')\n"
           "h, w = img.shape[:2]\n\n"
           "# 方法一：直接矯正\n"
           "undist = cv2.undistort(img, K, dist)\n\n"
           "# 方法二：使用最佳新相機矩陣\n"
           "# alpha=0: 裁剪掉所有黑邊\n"
           "# alpha=1: 保留所有像素（有黑邊）\n"
           "new_K, roi = cv2.getOptimalNewCameraMatrix(\n"
           "    K, dist, (w, h), alpha=0)\n"
           "undist2 = cv2.undistort(\n"
           "    img, K, dist, None, new_K)\n\n"
           "# 裁剪有效區域\n"
           "x, y, rw, rh = roi\n"
           "undist2 = undist2[y:y+rh, x:x+rw]\n\n"
           "# 方法三：映射表方式（批次處理推薦）\n"
           "map1, map2 = cv2.initUndistortRectifyMap(\n"
           "    K, dist, None, new_K,\n"
           "    (w, h), cv2.CV_32FC1)\n\n"
           "# 對每張影像快速矯正\n"
           "for fname in glob.glob('images/*.jpg'):\n"
           "    img_i = cv2.imread(fname)\n"
           "    fixed = cv2.remap(\n"
           "        img_i, map1, map2,\n"
           "        cv2.INTER_LINEAR)\n"
           "    cv2.imwrite(\n"
           "        fname.replace('.jpg', '_fixed.jpg'),\n"
           "        fixed)")

    p.case("精密零件量測中的校正實例",
           "場景：精密沖壓件的孔徑公差檢測，規格為 5.00 +/- 0.05 mm。\n\n"
           "問題：未校正時量測結果為 5.12 mm，一直判定為不良品，但用千分尺實測為 5.01 mm。\n\n"
           "根因分析：\n"
           "1. 鏡頭桶形畸變導致影像邊緣區域的孔洞被放大\n"
           "2. 像素到毫米的轉換係數使用了理論值而非實測值\n\n"
           "解決流程：\n"
           "1. 使用 9x6 棋盤格拍攝 25 張校正影像\n"
           "2. 校正得到 RMS 重投影誤差 = 0.12 pixels\n"
           "3. 對所有檢測影像進行畸變矯正\n"
           "4. 使用 5mm 標準量規實測 mm/pixel = 0.01563\n"
           "5. 校正後量測結果：5.02 mm（與千分尺一致）\n\n"
           "效果：量測精度從 +/- 0.15mm 提升至 +/- 0.02mm，誤判率從 18% 降至 0.3%。")

    p.tip("校正板選擇和拍攝注意事項：\n"
          "1. 校正板應為高精度印刷或蝕刻，平面度 < 0.1mm\n"
          "2. 拍攝時棋盤格應覆蓋影像的各個區域（中心、四角、邊緣）\n"
          "3. 至少拍攝 15 張不同角度的影像，建議 20~30 張\n"
          "4. 棋盤格傾斜角度不應超過 45 度，否則角點檢測容易失敗\n"
          "5. 確保所有角點清晰可見，避免反光和陰影\n"
          "6. 校正環境的照明條件應與實際檢測環境一致")

    p.warn("常見校正失敗原因：\n"
           "1. 校正板不平整：紙質列印的棋盤格容易彎曲，應貼附在平整硬板上\n"
           "2. 拍攝角度太少或太集中：僅在正面拍攝無法準確估計畸變係數\n"
           "3. 角點檢測失敗卻未過濾：應檢查 findChessboardCorners 的返回值\n"
           "4. 重投影誤差過大（> 1.0 pixel）：表示校正結果不可靠，需重新拍攝\n"
           "5. 工作距離改變後未重新校正：mm/pixel 隨距離變化\n"
           "6. 溫度變化導致焦距漂移：精密量測應在恆溫環境中進行")


def write_ch16(p):
    p.ch_title("影像對齊與配準")
    p.txt("影像對齊（Image Registration / Alignment）是將測試影像轉換到與參考影像相同的座標系中。在瑕疵檢測中，對齊是模板比對、差分檢測和變異模型的前提條件。如果對齊不準確，正常的紋理邊緣會產生大量假陽性，使得瑕疵檢測完全失效。本章介紹從粗對齊到精對齊的完整方法體系。")

    # 16.1
    p.sec_title("為什麼需要對齊")
    p.txt("在工業產線上，被檢測產品進入視場時的位置和角度每次都不同。即使使用定位治具，仍然存在 0.5~2mm 的位置偏差和 0.5~3 度的旋轉偏差。對於逐像素比較的檢測方法（如差分影像法、變異模型），即使 1 個像素的錯位也會產生邊緣處的假差異。\n\n"
          "對齊方法分為兩大類：\n"
          "- 手動/固定對齊：使用機械治具和觸發器精確定位。成本高但延遲低。\n"
          "- 軟體自動對齊：通過影像處理計算變換矩陣。靈活且成本低，但需要計算時間。\n\n"
          "軟體對齊的精度通常可達 0.1~0.3 像素（使用亞像素方法），足以滿足大多數瑕疵檢測需求。")

    # 16.2
    p.sec_title("基於特徵點的對齊")
    p.txt("特徵點對齊是最通用的方法。流程：\n"
          "1. 在參考影像和測試影像中分別檢測特徵點（角點/斑點）\n"
          "2. 計算每個特徵點的描述子\n"
          "3. 匹配兩組描述子，找到對應點對\n"
          "4. 使用 RANSAC 剔除錯誤匹配（外點）\n"
          "5. 從正確的匹配點對估計變換矩陣（仿射或單應性）\n"
          "6. 使用變換矩陣將測試影像對齊到參考影像")

    p.formula("單應性矩陣（Homography）",
              "        [h11 h12 h13] [x]\n"
              "[x']    [h21 h22 h23] [y]\n"
              "[y'] ~= [h31 h32 h33] [1]\n"
              "[1 ]\n\n"
              "  x' = (h11*x + h12*y + h13) /\n"
              "       (h31*x + h32*y + h33)\n"
              "  y' = (h21*x + h22*y + h23) /\n"
              "       (h31*x + h32*y + h33)\n\n"
              "8 個自由度，需至少 4 對匹配點",
              "單應性矩陣描述兩個平面之間的投影變換。對於平面物體（如 PCB、標籤），Homography 是精確的變換模型。")

    p.code("import cv2\nimport numpy as np\n\n"
           "def align_by_features(test, ref,\n"
           "                       max_features=500,\n"
           "                       good_match_pct=0.15):\n"
           '    """基於 ORB 特徵的影像對齊\n'
           "    test: 測試影像 (灰度)\n"
           "    ref:  參考影像 (灰度)\n"
           '    返回: 對齊後的影像, 變換矩陣"""\n\n'
           "    # 1. 建立 ORB 偵測器\n"
           "    orb = cv2.ORB_create(\n"
           "        nfeatures=max_features)\n\n"
           "    # 2. 偵測特徵點和描述子\n"
           "    kp1, des1 = orb.detectAndCompute(\n"
           "        test, None)\n"
           "    kp2, des2 = orb.detectAndCompute(\n"
           "        ref, None)\n\n"
           "    # 3. 暴力匹配 (Hamming 距離)\n"
           "    matcher = cv2.BFMatcher(\n"
           "        cv2.NORM_HAMMING,\n"
           "        crossCheck=True)\n"
           "    matches = matcher.match(des1, des2)\n\n"
           "    # 4. 按距離排序，取最佳匹配\n"
           "    matches = sorted(\n"
           "        matches, key=lambda x: x.distance)\n"
           "    n_good = int(\n"
           "        len(matches) * good_match_pct)\n"
           "    n_good = max(n_good, 10)\n"
           "    good = matches[:n_good]\n\n"
           "    # 5. 提取匹配點座標\n"
           "    pts_test = np.float32(\n"
           "        [kp1[m.queryIdx].pt for m in good]\n"
           "    ).reshape(-1, 1, 2)\n"
           "    pts_ref = np.float32(\n"
           "        [kp2[m.trainIdx].pt for m in good]\n"
           "    ).reshape(-1, 1, 2)\n\n"
           "    # 6. RANSAC 估計 Homography\n"
           "    H, mask = cv2.findHomography(\n"
           "        pts_test, pts_ref,\n"
           "        cv2.RANSAC, 5.0)\n\n"
           "    # 7. 應用變換\n"
           "    h, w = ref.shape[:2]\n"
           "    aligned = cv2.warpPerspective(\n"
           "        test, H, (w, h))\n\n"
           "    inliers = mask.ravel().sum()\n"
           "    print(f'內點數: {inliers}/{len(good)}')\n\n"
           "    return aligned, H")

    # 16.3
    p.sec_title("基於輪廓/形狀的對齊")
    p.txt("當產品是單一明確形狀（零件輪廓）且背景簡單時，基於輪廓的對齊更加穩健快速。流程：\n"
          "1. 分割出目標物體的輪廓\n"
          "2. 找到最大輪廓的最小外接矩形（minAreaRect）\n"
          "3. 計算中心點和旋轉角度\n"
          "4. 使用平移和旋轉將輪廓對齊到標準位置\n\n"
          "這種方法不依賴紋理特徵，對於無紋理的均勻表面零件非常有效。")

    p.code("def align_by_contour(test, ref):\n"
           '    """基於輪廓的影像對齊\n'
           '    適用於背景簡單的單一零件"""\n\n'
           "    def get_pose(gray):\n"
           '        """提取物體中心和角度"""\n'
           "        _, binary = cv2.threshold(\n"
           "            gray, 0, 255,\n"
           "            cv2.THRESH_BINARY + cv2.THRESH_OTSU)\n"
           "        cnts, _ = cv2.findContours(\n"
           "            binary, cv2.RETR_EXTERNAL,\n"
           "            cv2.CHAIN_APPROX_SIMPLE)\n"
           "        # 取最大輪廓\n"
           "        cnt = max(cnts, key=cv2.contourArea)\n"
           "        rect = cv2.minAreaRect(cnt)\n"
           "        center = rect[0]  # (cx, cy)\n"
           "        angle = rect[2]   # 旋轉角度\n"
           "        return center, angle, cnt\n\n"
           "    # 取得參考和測試的姿態\n"
           "    c_ref, a_ref, _ = get_pose(ref)\n"
           "    c_test, a_test, _ = get_pose(test)\n\n"
           "    h, w = ref.shape[:2]\n"
           "    img_center = (w / 2.0, h / 2.0)\n\n"
           "    # 計算旋轉差\n"
           "    delta_angle = a_ref - a_test\n\n"
           "    # 旋轉矩陣（圍繞測試件中心）\n"
           "    M_rot = cv2.getRotationMatrix2D(\n"
           "        c_test, delta_angle, 1.0)\n\n"
           "    # 加上平移（旋轉後的中心對齊到參考中心）\n"
           "    cos_a = np.cos(np.radians(delta_angle))\n"
           "    sin_a = np.sin(np.radians(delta_angle))\n"
           "    new_cx = (cos_a*c_test[0]\n"
           "              - sin_a*c_test[1] + M_rot[0,2])\n"
           "    new_cy = (sin_a*c_test[0]\n"
           "              + cos_a*c_test[1] + M_rot[1,2])\n"
           "    M_rot[0, 2] += c_ref[0] - new_cx\n"
           "    M_rot[1, 2] += c_ref[1] - new_cy\n\n"
           "    aligned = cv2.warpAffine(\n"
           "        test, M_rot, (w, h))\n"
           "    return aligned, M_rot")

    # 16.4
    p.sec_title("基於相位相關的對齊")
    p.txt("相位相關（Phase Correlation）利用傅立葉變換計算兩張影像之間的平移偏移。它在頻率域中運作，對亮度變化具有魯棒性，且速度快（O(N log N)）。適用於僅有平移偏移的情況（無旋轉/縮放），如傳送帶上移動的零件。")

    p.formula("相位相關公式",
              "互功率譜：R(u,v) = F1(u,v) * conj(F2(u,v))\n"
              "                   / |F1 * conj(F2)|\n\n"
              "反傅立葉：r(x,y) = IFFT(R(u,v))\n\n"
              "偏移量 = argmax{r(x,y)} 的位置 (dx, dy)",
              "相位相關的峰值位置直接對應兩張影像的平移偏移。峰值的銳度指示對齊的可信度。")

    p.code("def align_by_phase(test, ref):\n"
           '    """基於相位相關的平移對齊\n'
           '    返回: 對齊後的影像, (dx, dy)"""\n\n'
           "    # 轉為 float\n"
           "    t = test.astype(np.float32)\n"
           "    r = ref.astype(np.float32)\n\n"
           "    # 漢寧窗減少邊界效應\n"
           "    h, w = t.shape\n"
           "    win_r = np.hanning(h).reshape(-1,1)\n"
           "    win_c = np.hanning(w).reshape(1,-1)\n"
           "    window = (win_r * win_c).astype(\n"
           "        np.float32)\n"
           "    t_w = t * window\n"
           "    r_w = r * window\n\n"
           "    # 相位相關\n"
           "    (dx, dy), response = cv2.phaseCorrelate(\n"
           "        t_w, r_w)\n"
           "    print(f'偏移: dx={dx:.2f}, dy={dy:.2f}')\n"
           "    print(f'相關峰值: {response:.4f}')\n\n"
           "    # 應用平移\n"
           "    M = np.float32([\n"
           "        [1, 0, -dx],\n"
           "        [0, 1, -dy]])\n"
           "    aligned = cv2.warpAffine(\n"
           "        test, M, (w, h))\n\n"
           "    return aligned, (dx, dy)")

    # 16.5
    p.sec_title("多尺度/多步驟配準策略")
    p.txt("在實際應用中，單一對齊方法可能不夠穩健或精確。多步驟策略結合不同方法的優勢：\n\n"
          "粗對齊（Coarse Alignment）：\n"
          "- 目標：消除大偏移和旋轉（> 5 pixels, > 2 度）\n"
          "- 方法：輪廓對齊或低解析度特徵匹配\n"
          "- 速度快，精度要求低\n\n"
          "精對齊（Fine Alignment）：\n"
          "- 目標：達到亞像素精度（< 0.5 pixels）\n"
          "- 方法：相位相關或高解析度模板匹配\n"
          "- 在粗對齊結果上進行微調")

    p.code("def multi_step_align(test, ref):\n"
           '    """多步驟配準：粗對齊 + 精對齊"""\n\n'
           "    # 步驟 1：粗對齊（輪廓法）\n"
           "    coarse, M1 = align_by_contour(\n"
           "        test, ref)\n\n"
           "    # 步驟 2：精對齊（相位相關）\n"
           "    fine, (dx, dy) = align_by_phase(\n"
           "        coarse, ref)\n\n"
           "    # 步驟 3：驗證對齊品質\n"
           "    diff = cv2.absdiff(fine, ref)\n"
           "    mean_diff = np.mean(diff)\n"
           "    print(f'對齊後平均差異: {mean_diff:.2f}')\n\n"
           "    if mean_diff > 30:  # 品質不佳\n"
           "        print('警告: 對齊品質不佳')\n"
           "        # 嘗試特徵點方法\n"
           "        fine, _ = align_by_features(\n"
           "            test, ref)\n\n"
           "    return fine")

    p.table(
        ["對齊方法", "速度", "精度", "適用場景"],
        [
            ["特徵點 (ORB)", "中等", "1~3 px", "有紋理的平面物體"],
            ["特徵點 (SIFT)", "慢", "0.5~2 px", "多尺度/旋轉變化"],
            ["輪廓/形狀", "快", "2~5 px", "背景簡單的零件"],
            ["相位相關", "快", "0.1~0.5 px", "僅平移偏移"],
            ["模板匹配", "慢", "1~2 px", "小 ROI 精確定位"],
            ["多步驟融合", "中等", "< 0.5 px", "高精度需求"],
        ]
    )

    p.case("PCB 對齊到黃金參考",
           "場景：PCB 板瑕疵檢測需要將測試影像與黃金參考精確對齊後差分。\n\n"
           "挑戰：\n"
           "- PCB 在傳送帶上位移可達 +/- 5mm\n"
           "- 旋轉偏差可達 +/- 2 度\n"
           "- 需要亞像素精度以避免邊緣處假陽性\n\n"
           "解決方案（三步法）：\n"
           "1. 粗定位：ORB 特徵匹配 + Homography，消除大偏移和旋轉\n"
           "   - 匹配 500 個特徵點，RANSAC 剔除外點\n"
           "   - 精度約 2 像素\n"
           "2. 精對齊：相位相關，消除殘餘平移\n"
           "   - 精度約 0.2 像素\n"
           "3. 局部微調：在關鍵 ROI 區域使用 NCC 模板匹配\n"
           "   - 精度約 0.1 像素\n\n"
           "結果：\n"
           "- 對齊成功率 > 99.5%（500 張測試影像僅 2 張失敗）\n"
           "- 假陽性率從未對齊的 45% 降至 < 2%\n"
           "- 整體對齊耗時約 50ms（1280x960 影像）")

    p.warn("對齊失敗的常見原因和解決方案：\n"
           "1. 特徵點太少：物體表面無紋理。解決方案：改用輪廓對齊或增加表面標記。\n"
           "2. 重複紋理導致錯誤匹配：週期性圖案（如網格）使匹配混亂。解決方案：限制搜尋範圍或使用 SIFT。\n"
           "3. 遮擋或缺損改變外觀：嚴重瑕疵使特徵消失。解決方案：使用 RANSAC 並提高外點容忍度。\n"
           "4. 光照變化過大：不同時間拍攝的亮度差異大。解決方案：先做直方圖均衡化再對齊。\n"
           "5. 變換模型錯誤：使用仿射變換對齊有透視變形的物體。解決方案：改用 Homography。")


def write_ch17(p):
    p.ch_title("光源原理與選型")
    p.txt("在工業視覺瑕疵檢測中，有一句經典名言：「打光佔 70%，演算法佔 30%」。正確的照明能讓瑕疵變得顯而易見，使簡單的閾值分割就能檢測；錯誤的照明則讓再先進的演算法也無能為力。本章深入探討各種照明方式的物理原理、光源類型選擇以及針對不同瑕疵的最佳打光策略。")

    # 17.1
    p.sec_title("光源在瑕疵檢測中的核心角色")
    p.txt("光源的作用是在瑕疵區域和正常區域之間產生最大的亮度對比度。沒有對比度，就沒有可檢測的信號。好的照明方案能夠：\n"
          "- 最大化瑕疵信號（亮度或顏色差異）\n"
          "- 最小化背景雜訊（表面紋理、反光）\n"
          "- 提供均勻一致的照明（避免假陽性）\n"
          "- 在產線速度下穩定工作\n\n"
          "為什麼光源比演算法更重要？考慮以下情境：\n"
          "- 金屬表面的細微刮痕在正面照明下幾乎看不到，但低角度暗場照明可使其清晰可見\n"
          "- 透明薄膜中的氣泡在反射光下不可見，但背光下立即顯現\n"
          "- 不均勻的照明會在均勻表面產生虛假的亮度梯度，被誤判為瑕疵\n\n"
          "經驗法則：如果無法用肉眼在檢測影像中看到瑕疵，演算法也幾乎無法可靠地檢測它。")

    # 17.2
    p.sec_title("照明幾何（打光方式詳解）")

    p.sub_sec("正面照明")
    p.txt("光源從物體表面法線方向照射。適用於檢測平面上的印刷、標記和顏色差異。\n\n"
          "同軸光（Coaxial Light）：\n"
          "光線通過半透明鏡從鏡頭光軸方向照射，消除陰影。鏡面反射回到相機。適用於高反射率表面（拋光金屬、玻璃、晶圓）的缺陷檢測。\n"
          "原理：平坦表面將光直接反射回相機（亮），缺陷處法線偏轉導致光偏離（暗）。\n\n"
          "環形光（Ring Light）：\n"
          "圍繞鏡頭排列的 LED 環。角度可調（0~75 度）。提供均勻無陰影的照明。是最通用的工業照明方式。\n"
          "高角度環形光：類似正面照明，凸顯顏色和紋理差異。\n"
          "低角度環形光：產生側射效果，凸顯表面凸起和刮痕。\n\n"
          "穹頂光（Dome Light）：\n"
          "半球形的漫射照明。光線從所有方向均勻照射。完全消除反射和陰影。適用於高反光曲面物體（如金屬殼體、圓柱零件）的表面檢測。\n"
          "缺點：消除了所有表面拓撲資訊，不適合檢測凹凸類瑕疵。")

    p.sub_sec("側面照明（暗場照明）")
    p.txt("光源以極低角度（5~30 度）從側面照射表面。正常的平坦區域將光反射到遠離相機的方向（暗），而凸起、刮痕、邊緣等表面不規則處會將光散射向相機（亮）。\n\n"
          "原理：暗場照明利用的是散射光而非鏡面反射。在暗背景上，瑕疵表現為明亮的特徵，產生極高的對比度。\n\n"
          "最適瑕疵類型：刮痕、毛邊、凸起、裂紋、表面粗糙度變化。\n"
          "限制：對表面輕微顏色變化不敏感。\n\n"
          "技巧：使用 4 個方向的低角度光源（0/90/180/270 度）逐一照射，可實現光度立體法並檢測所有方向的刮痕。")

    p.sub_sec("背光照明")
    p.txt("光源放在物體背面，相機在物體前面。物體遮擋光線形成輪廓/剪影。\n\n"
          "適用場景：\n"
          "- 外形尺寸量測（高對比度輪廓）\n"
          "- 透明/半透明材料中的缺陷（氣泡、異物、裂紋）\n"
          "- 孔洞、缺口的檢測（通光 vs 遮光）\n"
          "- 瓶子內的液位、異物檢測\n\n"
          "背光類型：直射背光（平行光，高對比度邊緣）vs 漫射背光（均勻照明，適合透明件）。")

    p.sub_sec("結構光")
    p.txt("投射特定的光圖案（線、網格、條紋）到物體表面。表面形狀的變化會導致光圖案變形，通過分析變形可以重建三維表面形貌。\n\n"
          "常見類型：\n"
          "- 單線結構光：用於線掃描場景，逐行重建高度\n"
          "- 多線結構光：加速掃描速度\n"
          "- 相移條紋投影：最高精度的 3D 測量\n\n"
          "適用於需要三維表面形貌的檢測：焊縫高度、零件平面度、翹曲變形。")

    # 17.3
    p.sec_title("光源類型與特性")

    p.table(
        ["光源類型", "壽命", "波長", "適用場景"],
        [
            ["白光 LED", ">50000h", "寬頻", "通用檢測"],
            ["單色 LED", ">50000h", "特定波長", "增強特定對比"],
            ["UV LED", ">20000h", "365~405nm", "螢光/膠水檢測"],
            ["IR LED", ">50000h", "850~940nm", "穿透/熱成像"],
            ["鹵素燈", "~2000h", "寬頻", "高亮度照明"],
            ["螢光燈", "~8000h", "寬頻", "均勻面照明"],
            ["雷射", ">20000h", "單波長", "結構光/干涉"],
        ]
    )

    p.txt("LED 光源是現代工業視覺的主流選擇。原因：\n"
          "- 壽命長（50000+ 小時）\n"
          "- 可瞬間開關（頻閃模式）\n"
          "- 波長可選（紅/綠/藍/白/紅外/紫外）\n"
          "- 亮度穩定且可調節\n"
          "- 不產生熱量（相比鹵素燈）\n\n"
          "頻閃控制：在高速產線上，持續照明會導致運動模糊。頻閃模式在短時間內（1~100 微秒）提供高亮度脈衝，有效「凍結」運動。")

    p.sub_sec("波長選擇")
    p.txt("不同波長與材料的交互不同。關鍵原則：\n"
          "- 使用與瑕疵顏色互補的波長增強對比：紅色瑕疵用藍光、藍色瑕疵用紅光\n"
          "- 紅色光穿透性較好，適合半透明材料的內部檢測\n"
          "- 藍色光波長短，聚焦更精確，適合高解析度檢測\n"
          "- UV 光可激發螢光材料（膠水、油脂、標記），使其在暗背景中發光\n"
          "- IR 光可穿透某些不透明材料（矽晶圓、塑膠），檢測內部缺陷")

    # 17.4
    p.sec_title("不同瑕疵類型的最佳打光方案")

    p.table(
        ["瑕疵類型", "推薦光源", "原理", "備註"],
        [
            ["刮痕", "低角度暗場", "散射光凸顯凸起", "多方向照明更佳"],
            ["凹坑/凹痕", "穹頂光/同軸", "均勻照明顯陰影", "深凹坑用側光"],
            ["毛邊/毛刺", "背光", "輪廓清晰可見", "最簡單有效"],
            ["污漬/變色", "漫射正光", "顏色差異最大化", "選合適波長"],
            ["裂紋", "低角度側光", "散射光增強細縫", "螢光滲透增強"],
            ["氣泡", "背光/暗場", "透射光凸顯氣泡", "漫射背光更佳"],
            ["缺件/多件", "正面環光", "整體外觀對比", "穩定均勻即可"],
            ["印刷缺陷", "同軸光", "消除紋理反射", "UV 光檢螢光"],
            ["焊接缺陷", "多角度光", "3D 形貌信息", "結構光最佳"],
            ["表面粗糙", "暗場對比", "粗糙區散射強", "量化粗糙度"],
        ]
    )

    # 17.5
    p.sec_title("打光實驗方法論")
    p.txt("選擇光源不應憑經驗猜測，而應系統化測試。建議流程：\n"
          "1. 準備含有已知瑕疵的樣品（包括良品和各類瑕疵品）\n"
          "2. 收集可用的光源和打光配件\n"
          "3. 固定相機參數（曝光、增益、對焦）\n"
          "4. 逐一測試每種打光方式，拍攝同一組樣品\n"
          "5. 對每組影像計算瑕疵的信噪比（SNR）\n"
          "6. 選擇 SNR 最高的打光方案\n"
          "7. 微調光源角度、亮度、距離")

    p.code("import cv2\nimport numpy as np\nimport os\n\n"
           "def evaluate_lighting(images_dir, roi=None):\n"
           '    """評估不同打光方案的信噪比\n'
           "    images_dir: 包含多組打光影像的目錄\n"
           "    每個子目錄為一種打光方案\n"
           '    roi: 已知瑕疵區域 (x,y,w,h) 或 None"""\n\n'
           "    results = {}\n"
           "    for light_dir in sorted(\n"
           "            os.listdir(images_dir)):\n"
           "        path = os.path.join(\n"
           "            images_dir, light_dir)\n"
           "        if not os.path.isdir(path):\n"
           "            continue\n\n"
           "        snr_list = []\n"
           "        for f in sorted(os.listdir(path)):\n"
           "            img = cv2.imread(\n"
           "                os.path.join(path, f), 0)\n"
           "            if img is None:\n"
           "                continue\n\n"
           "            if roi:\n"
           "                x,y,w,h = roi\n"
           "                defect = img[y:y+h, x:x+w]\n"
           "            else:\n"
           "                # 自動：中心 vs 邊緣\n"
           "                ch, cw = img.shape\n"
           "                defect = img[\n"
           "                    ch//4:3*ch//4,\n"
           "                    cw//4:3*cw//4]\n\n"
           "            # 計算 SNR\n"
           "            signal = float(\n"
           "                np.mean(defect))\n"
           "            noise = float(\n"
           "                np.std(img)) + 1e-6\n"
           "            snr = abs(signal\n"
           "                - np.mean(img)) / noise\n"
           "            snr_list.append(snr)\n\n"
           "        avg_snr = np.mean(snr_list)\n"
           "        results[light_dir] = avg_snr\n"
           "        print(f'{light_dir}: '\n"
           "              f'SNR={avg_snr:.2f}')\n\n"
           "    # 推薦最佳方案\n"
           "    best = max(results, key=results.get)\n"
           "    print(f'\\n推薦方案: {best} '\n"
           "          f'(SNR={results[best]:.2f})')\n"
           "    return results")

    p.case("金屬表面刮痕的光源選擇對比",
           "場景：拋光不銹鋼面板上的細微刮痕檢測（長度 1~10mm，深度 < 0.01mm）。\n\n"
           "測試了 5 種打光方案：\n"
           "1. 正面環形光（45 度）：SNR = 1.2（刮痕幾乎不可見）\n"
           "2. 同軸光：SNR = 2.8（部分刮痕可見，但對比度低）\n"
           "3. 穹頂光：SNR = 0.8（完全消除了表面拓撲信息）\n"
           "4. 單方向低角度暗場（10 度）：SNR = 8.5（垂直方向刮痕清晰，平行方向不可見）\n"
           "5. 四方向低角度暗場：SNR = 12.3（所有方向的刮痕都清晰可見）\n\n"
           "結論：四方向低角度暗場是最佳方案。每個方向的光源逐一點亮，拍攝 4 張影像後取最大值合成。\n\n"
           "後處理：\n"
           "- 合成影像 = max(img_0, img_90, img_180, img_270)\n"
           "- 簡單的 Otsu 閾值即可分割出刮痕\n"
           "- 檢出率 > 98%，假陽性率 < 1%")

    p.tip("打光的黃金法則：\n"
          "1. 先看樣品，用手電筒從各角度照射，觀察什麼時候瑕疵最明顯\n"
          "2. 暗場照明是表面瑕疵（刮痕、凸起、裂紋）的首選\n"
          "3. 背光是輪廓/透明件檢測的首選\n"
          "4. 穹頂光是高反光曲面的首選\n"
          "5. 同軸光是鏡面反射物體的首選\n"
          "6. 如果一種光源不夠，組合多種光源（多通道照明）\n"
          "7. 始終使用可控光源（LED），避免依賴環境光")

    p.warn("環境光干擾和解決方案：\n"
          "1. 問題：窗戶自然光隨時間變化，導致檢測結果不穩定\n"
          "   解決：用遮光罩或暗箱完全隔離環境光\n"
          "2. 問題：鄰近工作站的照明干擾\n"
          "   解決：使用窄頻 LED + 對應波長的帶通濾光片\n"
          "3. 問題：LED 光源亮度隨溫度衰減\n"
          "   解決：使用恆流驅動器，或定期自動校正亮度\n"
          "4. 問題：高速產線需要短曝光時間，亮度不足\n"
          "   解決：使用頻閃控制器提供高峰值亮度脈衝（過驅動模式）")


def write_ch18(p):
    p.ch_title("色彩瑕疵檢測")
    p.txt("許多瑕疵通過顏色差異表現：印刷偏色、塑膠變色、食品腐敗變色、紡織品染色不均、金屬氧化鏽蝕等。灰度影像無法捕捉這些資訊。本章介紹如何利用色彩空間分析、色差計算和多通道處理來檢測色彩相關的瑕疵。")

    # 18.1
    p.sec_title("色彩空間選擇")
    p.txt("不同的色彩空間將顏色資訊以不同方式組織。選擇合適的色彩空間對色彩瑕疵檢測至關重要：\n"
          "- RGB：相機原生格式，但三通道高度耦合，不適合直接分析\n"
          "- HSV：將色調（H）、飽和度（S）和亮度（V）分離，最直覺的色彩分割方式\n"
          "- Lab（CIE L*a*b*）：感知均勻的色彩空間，色差計算最準確\n"
          "- YCrCb：將亮度（Y）和色度（Cr/Cb）分離，對光照變化更魯棒")

    p.formula("RGB 到 Lab 轉換",
              "步驟 1: RGB -> XYZ (線性轉換)\n"
              "  [X]   [0.4124 0.3576 0.1805] [R]\n"
              "  [Y] = [0.2126 0.7152 0.0722] [G]\n"
              "  [Z]   [0.0193 0.1192 0.9505] [B]\n\n"
              "步驟 2: XYZ -> Lab (非線性)\n"
              "  L* = 116*f(Y/Yn) - 16\n"
              "  a* = 500*[f(X/Xn) - f(Y/Yn)]\n"
              "  b* = 200*[f(Y/Yn) - f(Z/Zn)]\n\n"
              "  f(t) = t^(1/3)         若 t > 0.008856\n"
              "  f(t) = 7.787*t + 16/116 否則",
              "Lab 的優勢：L* 代表亮度，a* 代表紅-綠軸，b* 代表黃-藍軸。兩個顏色的歐式距離（Delta E）與人眼感知的色差成正比。")

    p.formula("色差 Delta E",
              "CIE76 (最簡單):\n"
              "  dE76 = sqrt((L1-L2)^2 + (a1-a2)^2\n"
              "              + (b1-b2)^2)\n\n"
              "感知閾值:\n"
              "  dE < 1.0  : 人眼幾乎無法區分\n"
              "  dE ~ 2.0  : 近距離仔細觀察可區分\n"
              "  dE ~ 3.5  : 明顯可見的色差\n"
              "  dE > 5.0  : 不同顏色\n\n"
              "CIE2000 (更精確，考慮感知非均勻性):\n"
              "  dE00 = sqrt((dL'/(kL*SL))^2\n"
              "             + (dC'/(kC*SC))^2\n"
              "             + (dH'/(kH*SH))^2\n"
              "             + RT*(dC'/(kC*SC))\n"
              "                 *(dH'/(kH*SH)))",
              "CIE2000 更精確但計算更複雜。對於工業色差檢測，CIE76 通常足夠。嚴格的印刷品質控制建議使用 CIE2000。")

    p.table(
        ["色彩空間", "優點", "缺點", "適用場景"],
        [
            ["RGB", "相機原生格式", "通道耦合", "原始資料存儲"],
            ["HSV", "直覺的色彩分割", "H 值不連續", "快速色彩篩選"],
            ["Lab", "感知均勻色差", "計算較慢", "精確色差量測"],
            ["YCrCb", "亮度色度分離", "不常用", "光照補償"],
        ]
    )

    # 18.2
    p.sec_title("色差計算與閾值")
    p.txt("色差（Delta E）是量化兩個顏色之間視覺差異的標準方法。在瑕疵檢測中，計算每個像素與參考顏色的 Delta E，超過閾值的像素標記為色彩異常。")

    p.code("import cv2\nimport numpy as np\n\n"
           "def compute_delta_e(img, ref_color_lab,\n"
           "                     threshold=5.0):\n"
           '    """計算影像每個像素與參考色的 Delta E\n'
           "    img:           BGR 影像\n"
           "    ref_color_lab: 參考色 Lab 值 [L, a, b]\n"
           '    threshold:     色差閾值"""\n\n'
           "    # BGR -> Lab\n"
           "    lab = cv2.cvtColor(\n"
           "        img, cv2.COLOR_BGR2Lab)\n"
           "    lab = lab.astype(np.float32)\n\n"
           "    # OpenCV Lab 範圍:\n"
           "    # L: 0~255 (需映射到 0~100)\n"
           "    # a: 0~255 (需映射到 -128~127)\n"
           "    # b: 0~255 (需映射到 -128~127)\n"
           "    lab[:,:,0] = lab[:,:,0] * 100.0 / 255.0\n"
           "    lab[:,:,1] = lab[:,:,1] - 128.0\n"
           "    lab[:,:,2] = lab[:,:,2] - 128.0\n\n"
           "    # 計算 Delta E (CIE76)\n"
           "    ref = np.array(ref_color_lab,\n"
           "                   dtype=np.float32)\n"
           "    dL = lab[:,:,0] - ref[0]\n"
           "    da = lab[:,:,1] - ref[1]\n"
           "    db = lab[:,:,2] - ref[2]\n"
           "    delta_e = np.sqrt(\n"
           "        dL**2 + da**2 + db**2)\n\n"
           "    # 閾值分割\n"
           "    mask = (delta_e > threshold).astype(\n"
           "        np.uint8) * 255\n\n"
           "    return delta_e, mask")

    # 18.3
    p.sec_title("色彩區域分割")
    p.txt("HSV 色彩空間最適合進行色彩區域分割。H 通道直接表示色調（色輪上的角度），S 通道表示飽和度（顏色純度），V 通道表示亮度。通過設定 H/S/V 的上下限，可以精確分割出特定顏色的區域。")

    p.code("def color_segmentation(img,\n"
           "                        target='red'):\n"
           '    """基於 HSV 的色彩區域分割\n'
           "    img:    BGR 影像\n"
           '    target: 目標顏色名稱"""\n\n'
           "    hsv = cv2.cvtColor(\n"
           "        img, cv2.COLOR_BGR2HSV)\n\n"
           "    # 常用顏色的 HSV 範圍\n"
           "    # H: 0~180 (OpenCV), S: 0~255, V: 0~255\n"
           "    ranges = {\n"
           "        'red':    [(0,100,100),\n"
           "                   (10,255,255),\n"
           "                   (160,100,100),\n"
           "                   (180,255,255)],\n"
           "        'green':  [(35,80,80),\n"
           "                   (85,255,255)],\n"
           "        'blue':   [(95,100,100),\n"
           "                   (130,255,255)],\n"
           "        'yellow': [(20,100,100),\n"
           "                   (35,255,255)],\n"
           "    }\n\n"
           "    if target == 'red':  # 紅色跨 0 度\n"
           "        lo1, hi1, lo2, hi2 = [\n"
           "            np.array(x)\n"
           "            for x in ranges['red']]\n"
           "        mask1 = cv2.inRange(\n"
           "            hsv, lo1, hi1)\n"
           "        mask2 = cv2.inRange(\n"
           "            hsv, lo2, hi2)\n"
           "        mask = mask1 | mask2\n"
           "    else:\n"
           "        lo = np.array(ranges[target][0])\n"
           "        hi = np.array(ranges[target][1])\n"
           "        mask = cv2.inRange(\n"
           "            hsv, lo, hi)\n\n"
           "    # 形態學清理\n"
           "    k = cv2.getStructuringElement(\n"
           "        cv2.MORPH_ELLIPSE, (5,5))\n"
           "    mask = cv2.morphologyEx(\n"
           "        mask, cv2.MORPH_OPEN, k)\n"
           "    mask = cv2.morphologyEx(\n"
           "        mask, cv2.MORPH_CLOSE, k)\n\n"
           "    # 提取彩色區域\n"
           "    result = cv2.bitwise_and(\n"
           "        img, img, mask=mask)\n\n"
           "    return mask, result")

    # 18.4
    p.sec_title("色彩一致性檢測")
    p.txt("在批次生產中，不同批次之間的顏色偏差是常見問題。色彩一致性檢測比較目前產品的顏色統計量與歷史基準，判斷是否有系統性的色偏。")

    p.code("def color_consistency_check(\n"
           "        img, roi, reference_stats,\n"
           "        tolerance=3.0):\n"
           '    """色彩一致性檢測\n'
           "    img:        BGR 影像\n"
           "    roi:        ROI 區域 (x,y,w,h)\n"
           "    reference_stats: 參考統計量\n"
           "      {'L_mean','a_mean','b_mean',\n"
           "       'L_std','a_std','b_std'}\n"
           '    tolerance:  容忍倍數 (std)"""\n\n'
           "    x,y,w,h = roi\n"
           "    region = img[y:y+h, x:x+w]\n\n"
           "    # 轉 Lab\n"
           "    lab = cv2.cvtColor(\n"
           "        region, cv2.COLOR_BGR2Lab)\n"
           "    lab = lab.astype(np.float32)\n"
           "    lab[:,:,0] = lab[:,:,0]*100.0/255.0\n"
           "    lab[:,:,1] -= 128.0\n"
           "    lab[:,:,2] -= 128.0\n\n"
           "    # 當前批次統計\n"
           "    L_m = np.mean(lab[:,:,0])\n"
           "    a_m = np.mean(lab[:,:,1])\n"
           "    b_m = np.mean(lab[:,:,2])\n\n"
           "    # 與參考比較\n"
           "    ref = reference_stats\n"
           "    dL = abs(L_m - ref['L_mean'])\n"
           "    da = abs(a_m - ref['a_mean'])\n"
           "    db = abs(b_m - ref['b_mean'])\n\n"
           "    # Delta E（均值色差）\n"
           "    de = np.sqrt(dL**2 + da**2 + db**2)\n\n"
           "    # 各通道偏離度\n"
           "    dev_L = dL / max(ref['L_std'], 0.1)\n"
           "    dev_a = da / max(ref['a_std'], 0.1)\n"
           "    dev_b = db / max(ref['b_std'], 0.1)\n\n"
           "    ok = (dev_L < tolerance and\n"
           "          dev_a < tolerance and\n"
           "          dev_b < tolerance)\n\n"
           "    return {\n"
           "        'delta_e': de,\n"
           "        'dev_L': dev_L,\n"
           "        'dev_a': dev_a,\n"
           "        'dev_b': dev_b,\n"
           "        'pass': ok,\n"
           "        'current_Lab': (L_m, a_m, b_m)\n"
           "    }")

    # 18.5
    p.sec_title("多通道融合瑕疵檢測")
    p.txt("某些瑕疵在單一色彩通道中不明顯，但在多通道組合中可被揭露。多通道融合策略：\n"
          "1. 分別在各通道中檢測異常\n"
          "2. 使用邏輯運算（OR/AND）合併結果\n"
          "3. 或計算通道間的統計偏離度\n\n"
          "這種方法特別適合瑕疵可能表現為顏色變化、亮度變化或兩者兼有的情況。")

    p.code("def multi_channel_detect(img, ref,\n"
           "                          thresh_lab=5.0,\n"
           "                          thresh_gray=30):\n"
           '    """多通道融合瑕疵檢測\n'
           "    結合灰度差異和色差兩個維度\n"
           "    img: 測試影像 (BGR)\n"
           '    ref: 參考影像 (BGR)"""\n\n'
           "    # 通道 1: 灰度差異\n"
           "    g_test = cv2.cvtColor(\n"
           "        img, cv2.COLOR_BGR2GRAY)\n"
           "    g_ref = cv2.cvtColor(\n"
           "        ref, cv2.COLOR_BGR2GRAY)\n"
           "    diff_gray = cv2.absdiff(\n"
           "        g_test, g_ref)\n"
           "    _, mask_gray = cv2.threshold(\n"
           "        diff_gray, thresh_gray, 255,\n"
           "        cv2.THRESH_BINARY)\n\n"
           "    # 通道 2: Lab 色差\n"
           "    lab_t = cv2.cvtColor(\n"
           "        img, cv2.COLOR_BGR2Lab)\n"
           "    lab_r = cv2.cvtColor(\n"
           "        ref, cv2.COLOR_BGR2Lab)\n"
           "    diff_lab = np.sqrt(np.sum(\n"
           "        (lab_t.astype(np.float32)\n"
           "         - lab_r.astype(np.float32))**2,\n"
           "        axis=2))\n"
           "    mask_color = (\n"
           "        diff_lab > thresh_lab\n"
           "    ).astype(np.uint8) * 255\n\n"
           "    # 通道 3: HSV 色調差異\n"
           "    hsv_t = cv2.cvtColor(\n"
           "        img, cv2.COLOR_BGR2HSV)\n"
           "    hsv_r = cv2.cvtColor(\n"
           "        ref, cv2.COLOR_BGR2HSV)\n"
           "    diff_h = np.abs(\n"
           "        hsv_t[:,:,0].astype(np.int16)\n"
           "        - hsv_r[:,:,0].astype(np.int16))\n"
           "    diff_h = np.minimum(\n"
           "        diff_h, 180 - diff_h)\n"
           "    mask_hue = (\n"
           "        diff_h > 15\n"
           "    ).astype(np.uint8) * 255\n\n"
           "    # 融合：任一通道檢測到即標記\n"
           "    combined = (mask_gray | mask_color\n"
           "                | mask_hue)\n\n"
           "    # 形態學清理\n"
           "    k = cv2.getStructuringElement(\n"
           "        cv2.MORPH_ELLIPSE, (5,5))\n"
           "    combined = cv2.morphologyEx(\n"
           "        combined, cv2.MORPH_OPEN, k)\n"
           "    combined = cv2.morphologyEx(\n"
           "        combined, cv2.MORPH_CLOSE, k)\n\n"
           "    return {\n"
           "        'mask_gray': mask_gray,\n"
           "        'mask_color': mask_color,\n"
           "        'mask_hue': mask_hue,\n"
           "        'combined': combined\n"
           "    }")

    p.case("食品包裝印刷色差檢測",
           "場景：零食包裝袋的印刷品質檢測。品牌 LOGO 顏色必須一致（Pantone 色號規格）。\n\n"
           "品質標準：\n"
           "- LOGO 區域平均 Delta E < 3.0（CIE76）\n"
           "- 單一像素最大 Delta E < 8.0\n"
           "- 色差面積佔比 < 5%\n\n"
           "檢測流程：\n"
           "1. ROI 提取：使用模板匹配定位 LOGO 區域\n"
           "2. 轉換至 Lab 色彩空間\n"
           "3. 計算每個像素的 Delta E（與標準色比較）\n"
           "4. 統計分析：\n"
           "   - 平均 Delta E = 2.1（OK，< 3.0）\n"
           "   - 最大 Delta E = 6.2（OK，< 8.0）\n"
           "   - 超標面積 = 2.3%（OK，< 5%）\n"
           "5. 結果：PASS\n\n"
           "實施效果：\n"
           "- 每分鐘檢測 120 個包裝\n"
           "- 色差檢出率 > 99%\n"
           "- 客訴率降低 85%")

    p.case("紡織品染色均勻性",
           "場景：布料染色後的均勻性檢測。要求整匹布的顏色偏差 < 1.5 Delta E。\n\n"
           "挑戰：\n"
           "- 布料表面有紋理，像素級比較會產生大量雜訊\n"
           "- 染色不均通常是漸變的（非銳利邊界）\n"
           "- 需要區域級（而非像素級）的色差評估\n\n"
           "方法：\n"
           "1. 將布料影像分割為 N x M 個小區塊（如 50x50 像素）\n"
           "2. 計算每個區塊的平均 Lab 值\n"
           "3. 計算所有區塊之間的最大 Delta E\n"
           "4. 繪製色差熱力圖定位不均勻區域\n"
           "5. 判定：max(Delta E 所有區塊對) < 1.5\n\n"
           "結果：成功檢測出邊緣到中心的色差梯度問題，幫助調整染缸浴比。")

    p.tip("色彩校正（白平衡）：\n"
          "1. 在每次檢測開始前，拍攝標準白色參考板\n"
          "2. 計算白平衡校正係數：gain_R, gain_G, gain_B\n"
          "3. 對所有檢測影像套用校正\n"
          "4. 這確保了不同時間點的色彩一致性\n"
          "5. 使用 X-Rite ColorChecker 等標準色卡可進行更精確的多點色彩校正\n"
          "6. 定期（每班次或每小時）重新校正白平衡")

    p.warn("光源色溫對色彩檢測的影響：\n"
           "1. 不同色溫的光源會使同一物體呈現不同顏色。5500K 日光色最接近自然色\n"
           "2. LED 光源的色溫可能隨時間和溫度漂移。使用恆溫驅動和定期校正\n"
           "3. 混合光源（如 LED + 日光）會導致色彩不一致。必須隔離環境光\n"
           "4. 不同批次的 LED 可能有色溫差異。購買時要求相同 bin 號\n"
           "5. 相機感測器的色彩回應也會隨溫度變化。精密色彩檢測需要溫控環境")


def write_ch19(p):
    p.ch_title("進階輪廓分析")
    p.txt("前面章節介紹了基本的輪廓提取和特徵計算。本章深入探討進階輪廓分析技術，包括輪廓層次結構解析、形狀描述與匹配、最小外接形狀擬合、輪廓距離計算以及基於多特徵的缺陷分類引擎。這些技術使得瑕疵檢測系統能夠精確描述和分類各種複雜形狀的缺陷。")

    # 19.1
    p.sec_title("輪廓層次結構")
    p.txt("OpenCV 的 findContours 可以返回輪廓的層次結構（hierarchy），描述輪廓之間的父子關係和嵌套關係。不同的檢索模式會產生不同的層次結構：\n\n"
          "- RETR_EXTERNAL：只返回最外層輪廓（無層次）\n"
          "- RETR_LIST：返回所有輪廓，無父子關係（扁平列表）\n"
          "- RETR_CCOMP：返回兩層結構（外輪廓和內孔洞）\n"
          "- RETR_TREE：返回完整的層次樹結構\n\n"
          "hierarchy 的每個條目格式：[next, prev, child, parent]\n"
          "- next：同層次的下一個輪廓索引（-1 表示無）\n"
          "- prev：同層次的上一個輪廓索引\n"
          "- child：第一個子輪廓索引\n"
          "- parent：父輪廓索引")

    p.code("import cv2\nimport numpy as np\n\n"
           "def analyze_hierarchy(binary):\n"
           '    """解析輪廓層次結構\n'
           '    識別外輪廓、孔洞和嵌套結構"""\n\n'
           "    contours, hierarchy = cv2.findContours(\n"
           "        binary, cv2.RETR_TREE,\n"
           "        cv2.CHAIN_APPROX_SIMPLE)\n\n"
           "    if hierarchy is None:\n"
           "        return [], [], []\n\n"
           "    hier = hierarchy[0]  # shape: (N, 4)\n\n"
           "    outer = []    # 最外層輪廓\n"
           "    holes = []    # 孔洞（有父輪廓的）\n"
           "    nested = []   # 嵌套物件（孔洞中的物件）\n\n"
           "    for i, h in enumerate(hier):\n"
           "        nxt, prev, child, parent = h\n\n"
           "        if parent == -1:\n"
           "            # 頂層輪廓\n"
           "            area = cv2.contourArea(\n"
           "                contours[i])\n"
           "            n_holes = 0\n"
           "            c = child\n"
           "            while c != -1:\n"
           "                n_holes += 1\n"
           "                c = hier[c][0]  # next\n"
           "            outer.append({\n"
           "                'idx': i,\n"
           "                'contour': contours[i],\n"
           "                'area': area,\n"
           "                'holes': n_holes\n"
           "            })\n"
           "        elif hier[parent][3] == -1:\n"
           "            # 父輪廓是頂層 -> 這是孔洞\n"
           "            holes.append({\n"
           "                'idx': i,\n"
           "                'contour': contours[i],\n"
           "                'area': cv2.contourArea(\n"
           "                    contours[i]),\n"
           "                'parent': parent\n"
           "            })\n"
           "        else:\n"
           "            # 更深層嵌套\n"
           "            nested.append({\n"
           "                'idx': i,\n"
           "                'contour': contours[i],\n"
           "                'depth': get_depth(\n"
           "                    hier, i)\n"
           "            })\n\n"
           "    return outer, holes, nested\n\n"
           "def get_depth(hier, idx):\n"
           '    """計算輪廓在層次樹中的深度"""\n'
           "    depth = 0\n"
           "    current = idx\n"
           "    while hier[current][3] != -1:\n"
           "        depth += 1\n"
           "        current = hier[current][3]\n"
           "    return depth")

    # 19.2
    p.sec_title("輪廓近似與形狀描述")
    p.txt("輪廓可以用多種方式描述和簡化：\n"
          "- 多邊形近似（approxPolyDP）：用更少的頂點近似輪廓\n"
          "- 凸包（convexHull）：最小凸多邊形\n"
          "- 凸包缺陷（convexityDefects）：輪廓與凸包的偏差\n"
          "- Hu 矩：旋轉、縮放和平移不變的形狀描述子\n"
          "- matchShapes：基於 Hu 矩的形狀比較")

    p.formula("Hu 矩不變量",
              "基於中心矩 mu_pq 的歸一化中心矩:\n"
              "  eta_pq = mu_pq / mu_00^((p+q)/2 + 1)\n\n"
              "7 個 Hu 矩不變量 (h1~h7):\n"
              "  h1 = eta_20 + eta_02\n"
              "  h2 = (eta_20-eta_02)^2\n"
              "       + 4*eta_11^2\n"
              "  h3 = (eta_30-3*eta_12)^2\n"
              "       + (3*eta_21-eta_03)^2\n"
              "  ...\n"
              "  h7 = (3*eta_21-eta_03)\n"
              "       *(eta_30+eta_12)*[...]\n"
              "       - (eta_30-3*eta_12)\n"
              "       *(eta_21+eta_03)*[...]",
              "Hu 矩對平移、旋轉和縮放不變。h1~h6 對鏡像也不變，h7 的符號在鏡像時翻轉。在形狀匹配中非常有用。")

    p.code("def shape_analysis(contour, ref_contour):\n"
           '    """全面的形狀分析與匹配\n'
           "    contour:     待分析輪廓\n"
           '    ref_contour: 參考形狀輪廓"""\n\n'
           "    # 1. 多邊形近似\n"
           "    epsilon = 0.02 * cv2.arcLength(\n"
           "        contour, True)\n"
           "    approx = cv2.approxPolyDP(\n"
           "        contour, epsilon, True)\n"
           "    n_vertices = len(approx)\n\n"
           "    # 2. 凸包和凸包缺陷\n"
           "    hull = cv2.convexHull(contour)\n"
           "    hull_idx = cv2.convexHull(\n"
           "        contour, returnPoints=False)\n"
           "    if len(hull_idx) > 3 and \\\n"
           "       len(contour) > 3:\n"
           "        defects = cv2.convexityDefects(\n"
           "            contour, hull_idx)\n"
           "    else:\n"
           "        defects = None\n\n"
           "    # 分析凸包缺陷\n"
           "    max_defect_depth = 0\n"
           "    n_defects = 0\n"
           "    if defects is not None:\n"
           "        for d in defects:\n"
           "            s, e, f, depth = d[0]\n"
           "            # depth 單位是像素*256\n"
           "            real_depth = depth / 256.0\n"
           "            if real_depth > 5:\n"
           "                n_defects += 1\n"
           "            max_defect_depth = max(\n"
           "                max_defect_depth,\n"
           "                real_depth)\n\n"
           "    # 3. Hu 矩\n"
           "    moments = cv2.moments(contour)\n"
           "    hu = cv2.HuMoments(moments)\n"
           "    # 對數尺度（便於比較）\n"
           "    hu_log = -np.sign(hu) * np.log10(\n"
           "        np.abs(hu) + 1e-30)\n\n"
           "    # 4. 形狀匹配（與參考比較）\n"
           "    # 方法 1: I1 (最常用)\n"
           "    match_1 = cv2.matchShapes(\n"
           "        contour, ref_contour,\n"
           "        cv2.CONTOURS_MATCH_I1, 0)\n"
           "    # 方法 2: I2\n"
           "    match_2 = cv2.matchShapes(\n"
           "        contour, ref_contour,\n"
           "        cv2.CONTOURS_MATCH_I2, 0)\n"
           "    # 方法 3: I3\n"
           "    match_3 = cv2.matchShapes(\n"
           "        contour, ref_contour,\n"
           "        cv2.CONTOURS_MATCH_I3, 0)\n\n"
           "    return {\n"
           "        'n_vertices': n_vertices,\n"
           "        'n_convex_defects': n_defects,\n"
           "        'max_defect_depth':\n"
           "            max_defect_depth,\n"
           "        'hu_moments': hu_log.ravel(),\n"
           "        'match_I1': match_1,\n"
           "        'match_I2': match_2,\n"
           "        'match_I3': match_3,\n"
           "    }")

    # 19.3
    p.sec_title("最小外接形狀")
    p.txt("將輪廓擬合到標準幾何形狀（矩形、圓、橢圓），可以量化形狀特徵並與規格比較。OpenCV 提供多種擬合方法：")

    p.code("def fit_shapes(contour):\n"
           '    """擬合各種最小外接形狀"""\n'
           "    results = {}\n\n"
           "    # 1. 最小外接旋轉矩形\n"
           "    rect = cv2.minAreaRect(contour)\n"
           "    center, (w, h), angle = rect\n"
           "    results['rect'] = {\n"
           "        'center': center,\n"
           "        'width': max(w, h),\n"
           "        'height': min(w, h),\n"
           "        'angle': angle,\n"
           "        'aspect_ratio': max(w,h) /\n"
           "            (min(w,h) + 1e-6)\n"
           "    }\n\n"
           "    # 2. 最小外接圓\n"
           "    (cx,cy), radius = \\\n"
           "        cv2.minEnclosingCircle(contour)\n"
           "    area = cv2.contourArea(contour)\n"
           "    circle_area = np.pi * radius**2\n"
           "    results['circle'] = {\n"
           "        'center': (cx, cy),\n"
           "        'radius': radius,\n"
           "        'fill_ratio':\n"
           "            area / (circle_area + 1e-6)\n"
           "    }\n\n"
           "    # 3. 擬合橢圓（需 >= 5 點）\n"
           "    if len(contour) >= 5:\n"
           "        ellipse = cv2.fitEllipse(contour)\n"
           "        e_center, (e_ma, e_mi), e_angle = \\\n"
           "            ellipse\n"
           "        results['ellipse'] = {\n"
           "            'center': e_center,\n"
           "            'major_axis': max(e_ma, e_mi),\n"
           "            'minor_axis': min(e_ma, e_mi),\n"
           "            'angle': e_angle,\n"
           "            'eccentricity': np.sqrt(\n"
           "                1 - (min(e_ma,e_mi)/\n"
           "                     max(e_ma,e_mi))**2)\n"
           "        }\n\n"
           "    # 4. 擬合直線\n"
           "    if len(contour) >= 2:\n"
           "        line = cv2.fitLine(\n"
           "            contour, cv2.DIST_L2,\n"
           "            0, 0.01, 0.01)\n"
           "        vx, vy, x0, y0 = line.ravel()\n"
           "        results['line'] = {\n"
           "            'direction': (vx, vy),\n"
           "            'point': (x0, y0),\n"
           "            'angle_deg': np.degrees(\n"
           "                np.arctan2(vy, vx))\n"
           "        }\n\n"
           "    return results")

    # 19.4
    p.sec_title("輪廓距離與相似度")
    p.txt("比較兩個輪廓的相似程度，在模板比對和缺陷分類中非常重要。常用的輪廓距離度量包括 Hausdorff 距離和基於 Hu 矩的匹配距離。")

    p.formula("Hausdorff 距離",
              "h(A,B) = max{ min{d(a,b) : b in B}\n"
              "              : a in A }\n\n"
              "H(A,B) = max{ h(A,B), h(B,A) }\n\n"
              "其中 d(a,b) 為兩點間的歐式距離",
              "Hausdorff 距離衡量兩個點集之間的最大不匹配程度。值越小表示兩個輪廓越相似。對異常值敏感，實務中常使用百分位數版本（如第 95 百分位）。")

    p.code("from scipy.spatial.distance import \\\n"
           "    directed_hausdorff\n\n"
           "def contour_similarity(cnt1, cnt2):\n"
           '    """計算兩個輪廓的多種相似度指標"""\n\n'
           "    # 1. Hausdorff 距離\n"
           "    pts1 = cnt1.reshape(-1, 2).astype(\n"
           "        np.float64)\n"
           "    pts2 = cnt2.reshape(-1, 2).astype(\n"
           "        np.float64)\n"
           "    h_fwd = directed_hausdorff(\n"
           "        pts1, pts2)[0]\n"
           "    h_bwd = directed_hausdorff(\n"
           "        pts2, pts1)[0]\n"
           "    hausdorff = max(h_fwd, h_bwd)\n\n"
           "    # 2. 平均最近點距離\n"
           "    from scipy.spatial import cKDTree\n"
           "    tree2 = cKDTree(pts2)\n"
           "    dists1, _ = tree2.query(pts1)\n"
           "    tree1 = cKDTree(pts1)\n"
           "    dists2, _ = tree1.query(pts2)\n"
           "    avg_dist = (\n"
           "        np.mean(dists1) +\n"
           "        np.mean(dists2)) / 2\n\n"
           "    # 3. matchShapes (Hu 矩)\n"
           "    match_hu = cv2.matchShapes(\n"
           "        cnt1, cnt2,\n"
           "        cv2.CONTOURS_MATCH_I1, 0)\n\n"
           "    # 4. 面積比\n"
           "    a1 = cv2.contourArea(cnt1)\n"
           "    a2 = cv2.contourArea(cnt2)\n"
           "    area_ratio = min(a1,a2) / (\n"
           "        max(a1,a2) + 1e-6)\n\n"
           "    # 5. 圓度比較\n"
           "    p1 = cv2.arcLength(cnt1, True)\n"
           "    p2 = cv2.arcLength(cnt2, True)\n"
           "    c1 = 4*np.pi*a1/(p1**2+1e-6)\n"
           "    c2 = 4*np.pi*a2/(p2**2+1e-6)\n"
           "    circ_diff = abs(c1 - c2)\n\n"
           "    return {\n"
           "        'hausdorff': hausdorff,\n"
           "        'avg_dist': avg_dist,\n"
           "        'hu_match': match_hu,\n"
           "        'area_ratio': area_ratio,\n"
           "        'circ_diff': circ_diff\n"
           "    }")

    # 19.5
    p.sec_title("缺陷輪廓分類策略")
    p.txt("在工業檢測中，不同類型的瑕疵有不同的輪廓特徵。通過計算多個輪廓特徵並建立規則引擎，可以自動將檢測到的缺陷分類為不同的瑕疵類型。")

    p.table(
        ["瑕疵類型", "面積", "圓度", "長寬比", "實心度"],
        [
            ["刮痕", "中等", "< 0.2", "> 5.0", "> 0.6"],
            ["凹坑", "小~中", "> 0.7", "~ 1.0", "> 0.85"],
            ["缺角", "大", "< 0.5", "1~3", "< 0.7"],
            ["氣泡", "小", "> 0.8", "~ 1.0", "> 0.9"],
            ["裂紋", "小~中", "< 0.15", "> 8.0", "> 0.5"],
            ["污漬", "中~大", "0.3~0.7", "1~3", "> 0.8"],
            ["毛邊", "小", "< 0.3", "> 3.0", "< 0.6"],
        ]
    )

    p.code("def classify_defect_contour(contour,\n"
           "                             gray_img):\n"
           '    """基於多特徵的缺陷輪廓分類器\n'
           "    contour: 單個瑕疵輪廓\n"
           '    gray_img: 灰度影像（計算亮度特徵）"""\n\n'
           "    # --- 幾何特徵 ---\n"
           "    area = cv2.contourArea(contour)\n"
           "    peri = cv2.arcLength(contour, True)\n"
           "    if peri < 1e-6:\n"
           "        return 'noise', 0.0, {}\n\n"
           "    circ = 4*np.pi*area / (peri**2)\n\n"
           "    # 最小外接旋轉矩形\n"
           "    rect = cv2.minAreaRect(contour)\n"
           "    w, h = rect[1]\n"
           "    if w < 1e-6 or h < 1e-6:\n"
           "        return 'noise', 0.0, {}\n"
           "    aspect = max(w,h) / min(w,h)\n\n"
           "    # 凸包\n"
           "    hull = cv2.convexHull(contour)\n"
           "    hull_area = cv2.contourArea(hull)\n"
           "    solidity = area / (\n"
           "        hull_area + 1e-6)\n\n"
           "    # 凸包缺陷數\n"
           "    hull_idx = cv2.convexHull(\n"
           "        contour, returnPoints=False)\n"
           "    n_defects = 0\n"
           "    if len(hull_idx) > 3 and \\\n"
           "       len(contour) > 3:\n"
           "        defs = cv2.convexityDefects(\n"
           "            contour, hull_idx)\n"
           "        if defs is not None:\n"
           "            for d in defs:\n"
           "                if d[0][3]/256.0 > 3:\n"
           "                    n_defects += 1\n\n"
           "    # 外接矩形填充率\n"
           "    bx,by,bw,bh = cv2.boundingRect(\n"
           "        contour)\n"
           "    extent = area / (bw * bh + 1e-6)\n\n"
           "    # --- 亮度特徵 ---\n"
           "    mask = np.zeros_like(gray_img)\n"
           "    cv2.drawContours(\n"
           "        mask, [contour], -1, 255, -1)\n"
           "    mean_val = cv2.mean(\n"
           "        gray_img, mask=mask)[0]\n"
           "    # 周圍亮度\n"
           "    k = cv2.getStructuringElement(\n"
           "        cv2.MORPH_ELLIPSE, (15,15))\n"
           "    surround = cv2.dilate(mask, k)\n"
           "    surround = surround - mask\n"
           "    bg_val = cv2.mean(\n"
           "        gray_img, mask=surround)[0]\n"
           "    contrast = abs(\n"
           "        mean_val - bg_val) / 255.0\n\n"
           "    # --- 特徵向量 ---\n"
           "    features = {\n"
           "        'area': area,\n"
           "        'circularity': circ,\n"
           "        'aspect_ratio': aspect,\n"
           "        'solidity': solidity,\n"
           "        'extent': extent,\n"
           "        'n_defects': n_defects,\n"
           "        'contrast': contrast,\n"
           "        'is_dark': mean_val < bg_val\n"
           "    }\n\n"
           "    # --- 規則引擎分類 ---\n"
           "    if area < 30:\n"
           "        label = 'noise'\n"
           "        conf = 0.95\n"
           "    elif circ < 0.15 and aspect > 8:\n"
           "        label = 'crack'\n"
           "        conf = 0.85\n"
           "    elif circ < 0.25 and aspect > 4:\n"
           "        label = 'scratch'\n"
           "        conf = 0.80\n"
           "    elif circ > 0.8 and aspect < 1.5:\n"
           "        if area < 200:\n"
           "            label = 'bubble'\n"
           "            conf = 0.85\n"
           "        else:\n"
           "            label = 'pit'\n"
           "            conf = 0.80\n"
           "    elif solidity < 0.65:\n"
           "        if n_defects > 2:\n"
           "            label = 'chipping'\n"
           "            conf = 0.75\n"
           "        else:\n"
           "            label = 'burr'\n"
           "            conf = 0.70\n"
           "    elif circ > 0.3 and aspect < 3:\n"
           "        label = 'stain'\n"
           "        conf = 0.70\n"
           "    else:\n"
           "        label = 'unknown'\n"
           "        conf = 0.50\n\n"
           "    return label, conf, features\n\n"
           "# 使用範例\n"
           "def inspect_all_defects(binary, gray):\n"
           '    """檢測並分類所有缺陷"""\n'
           "    contours, _ = cv2.findContours(\n"
           "        binary, cv2.RETR_EXTERNAL,\n"
           "        cv2.CHAIN_APPROX_SIMPLE)\n\n"
           "    results = []\n"
           "    for cnt in contours:\n"
           "        label, conf, feat = \\\n"
           "            classify_defect_contour(\n"
           "                cnt, gray)\n"
           "        if label != 'noise':\n"
           "            results.append({\n"
           "                'contour': cnt,\n"
           "                'label': label,\n"
           "                'confidence': conf,\n"
           "                'features': feat\n"
           "            })\n\n"
           "    # 按類型統計\n"
           "    from collections import Counter\n"
           "    types = Counter(\n"
           "        r['label'] for r in results)\n"
           "    print(f'缺陷統計: {dict(types)}')\n\n"
           "    return results")

    p.case("沖壓件邊緣缺陷分類",
           "場景：金屬沖壓件的邊緣品質檢測。常見缺陷包括毛邊、缺角、裂紋和卷邊。\n\n"
           "檢測流程：\n"
           "1. 背光照明拍攝零件輪廓影像\n"
           "2. Otsu 閾值分割出零件輪廓\n"
           "3. 與標準輪廓（CAD 產生）進行 Hausdorff 距離比較\n"
           "4. 距離 > 3 像素的區域標記為缺陷候選\n"
           "5. 提取每個缺陷區域的輪廓\n"
           "6. 多特徵分類：\n"
           "   - 毛邊：向外凸出、長寬比 > 3、面積小\n"
           "   - 缺角：向內缺損、面積大、圓度低\n"
           "   - 裂紋：從邊緣延伸、極細長、長寬比 > 8\n"
           "   - 卷邊：邊緣附近的雙重輪廓、灰度漸變\n\n"
           "判定標準：\n"
           "- 毛邊長度 < 0.3mm：OK\n"
           "- 毛邊長度 0.3~0.5mm：Minor\n"
           "- 毛邊長度 > 0.5mm：Major\n"
           "- 缺角面積 > 0.5mm2：Scrap\n"
           "- 任何裂紋：Scrap\n\n"
           "結果：分類準確率 94%，檢出率 98%。主要混淆：短毛邊 vs 小缺角。")

    p.tip("輪廓分析的最佳實踐：\n"
          "1. 始終先用 RETR_EXTERNAL 排除內部雜訊，再用 RETR_TREE 分析結構\n"
          "2. approxPolyDP 的 epsilon 值影響近似精度：0.01*周長 保留細節，0.05*周長 簡化形狀\n"
          "3. matchShapes 返回值越小越相似（0 = 完全匹配）。閾值通常設 0.1~0.3\n"
          "4. Hu 矩對非常細長的形狀效果不佳，此時應改用長寬比+方向特徵\n"
          "5. 建立標準缺陷樣本庫，用 matchShapes 與樣本庫比對可提升分類準確度\n"
          "6. 對於複雜形狀，組合多個特徵（幾何+亮度+紋理）比依賴單一特徵更穩健\n"
          "7. 使用 mm 單位的特徵（而非像素）使規則在不同解析度下通用")
