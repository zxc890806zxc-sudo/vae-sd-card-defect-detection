import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# ===============================
# 1) 輸出與命名設定
# ===============================
SAVE_FOLDER = "test"         # 臨時擷取的單張影像、以及 12 區塊裁切輸出的位置
SAVE_img_name = "t"          # 擷取影像的前綴檔名
os.makedirs(SAVE_FOLDER, exist_ok=True)
save_index = 0               # 每次按 'c' 擷取時的流水號 (呼叫端可自行管理遞增)

# ===============================
# 2) VAE 基本參數
# ===============================
IMG_SIZE = (256, 350)        # VAE 輸入影像的統一大小 (h, w)
LATENT_DIM = 160             # 潛在空間維度：壓縮後的特徵向量長度

# ===============================
# 3) VAE 結構：Encoder / Decoder / VAE
# ===============================
class VAE_Encoder(layers.Layer):
    """
    將輸入影像 -> 萃取多層卷積特徵 -> 展平 -> 產生 (z_mean, z_log_var)
    用於重參數化技巧 (reparameterization) 以得到隨機 latent z。
    """
    def __init__(self, latent_dim):
        super(VAE_Encoder, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")
        self.pool1 = layers.MaxPooling2D((2, 2), padding="same")
        self.conv2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")
        self.pool2 = layers.MaxPooling2D((2, 2), padding="same")
        self.flatten = layers.Flatten()
        self.mean = layers.Dense(latent_dim)     # μ
        self.log_var = layers.Dense(latent_dim)  # log(σ^2)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        z_mean = self.mean(x)
        z_log_var = self.log_var(x)
        return z_mean, z_log_var

def sampling(z_mean, z_log_var):
    """
    Reparameterization: z = μ + σ * ε, 其中 ε ~ N(0, I)
    透過此技巧可對 z 反向傳遞梯度。
    """
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE_Decoder(layers.Layer):
    """
    將 latent 向量 z 解碼回影像大小：
      Dense 展開 → Reshape 成 feature map → ConvTranspose/UpSampling 放大 → Crop 尺寸修正 → Sigmoid 輸出灰階影像
    """
    def __init__(self):
        super(VAE_Decoder, self).__init__()
        self.dense = layers.Dense(64 * 88 * 64, activation="relu")
        self.reshape = layers.Reshape((64, 88, 64))
        self.deconv1 = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same")
        self.up1 = layers.UpSampling2D((2, 2))  
        self.deconv2 = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same")
        self.up2 = layers.UpSampling2D((2, 2))  
        self.crop = layers.Cropping2D(((0, 0), (1, 1)))  # 尺寸微調，避免 off-by-one
        self.output_layer = layers.Conv2DTranspose(1, (3, 3), activation="sigmoid", padding="same")

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.up1(x)
        x = self.deconv2(x)
        x = self.up2(x)
        x = self.crop(x)
        return self.output_layer(x)

class VAE(keras.Model):
    """
    高層封裝：Encoder 取 (μ, logσ²) → sampling → Decoder 輸出重建影像
    """
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(latent_dim)
        self.decoder = VAE_Decoder()

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = sampling(z_mean, z_log_var)
        return self.decoder(z)

# ===============================
# 4) 載入 4 組 VAE 權重 (D1~D4)
#    每組權重可對應不同區塊訓練資料/視角/瑕疵型態
# ===============================
vae_models = []
for i in range(4):
    model = VAE(LATENT_DIM)
    dummy_input = np.zeros((1, 256, 350, 1))  # 先跑一次建立圖 (建好權重對應的計算圖)
    _ = model(dummy_input)
    model.load_weights(f"vae_D{i+1}.weights.h5")
    vae_models.append(model)
    print(f"✅ 第{i+1}個 VAE 權重載入成功！")

# ===============================
# 5) 圖像前處理 + VAE 重建
#    會產生兩種前處理版本，各自重建 (增加穩定性)
# ===============================
def reconstruct_image(image_path, vae_model):
    # 以 Keras 讀圖並縮放到 VAE 輸入大小；color_mode='grayscale'
    img1 = load_img(image_path, target_size=IMG_SIZE, color_mode="grayscale")
    img1 = np.array(img1, dtype=np.uint8)

    # 先做高斯模糊，再對兩張版本都做雙邊濾波 (更保邊緣)
    img2 = cv2.GaussianBlur(img1, (3, 3), 0)
    img1 = cv2.bilateralFilter(img2, 20, 35, 100)
    img2 = cv2.bilateralFilter(img2, 20, 35, 100)

    # 正規化到 [0,1]，並餵進 VAE 重建
    img1 = img_to_array(img1) / 255.0
    img2 = img_to_array(img2) / 255.0
    rec1 = vae_model(np.expand_dims(img1, 0)).numpy().squeeze()
    rec2 = vae_model(np.expand_dims(img2, 0)).numpy().squeeze()
    return img1.squeeze(), rec1, img2.squeeze(), rec2

# ===============================
# 6) 評估工具：誤差與 SSIM
# ===============================
def calculate_error(original, reconstructed):
    """
    回傳像素級平方誤差圖與平均誤差值。
    """
    difference = np.abs((original - reconstructed)**2)
    error_value = np.mean(difference)
    return difference,float(error_value)

def calculate_ssim(original, reconstructed):
    """
    SSIM 結構相似度 (1 越相似)，輸入已正規化於 [0,1]，故 data_range=1.0。
    """
    return float(ssim(original, reconstructed, data_range=1.0))

def first_comparison(orig, recon, recon2):
    """
    對兩個重建版本分別與原圖比較，將差異圖/誤差/SSIM 取平均，降低單一路徑的隨機波動。
    """
    diff1,error1 = calculate_error(orig, recon)
    diff2,error2 = calculate_error(orig, recon2)
    diff  = np.abs((diff1 + diff2)/2)
    error = (error1 + error2)/2
    ssim1 = calculate_ssim(orig, recon)
    ssim2 = calculate_ssim(orig, recon2)
    ssim  = (ssim1 + ssim2) / 2
    return  diff, error, ssim

# ===============================
# 7) 差異圖導引的 ROI 偵測
#    (從差異圖擷取高紅值區域；多邊形 mask 內找輪廓；以條件過濾正常區域)
# ===============================
def findContours(orig, recon, img_path, hot):
    """
    參數:
      orig, recon: 原圖/重建圖 (float, 0~1)
      img_path   : 先前存下的差異圖 (已用 'hot' colormap)，此處直接讀回並用「紅通道」做高差異指示
      hot        : 紅通道的二值化閾值 (越高越嚴格)
    回傳:
      rois_o, rois_r, diffs, errs, ssims_：各 ROI 的對照資料 (第 0 筆是整張圖的基準)
    """
    rois_o, rois_r, diffs, errs, ssims_ = [], [], [], [], []

    # 先保留整張圖的差異 (作為 baseline)
    rois_o.append(orig); rois_r.append(recon)
    diff,error0 = calculate_error(orig, recon)
    errs.append(error0)
    ssim0 = calculate_ssim(orig, recon)
    ssims_.append(ssim0)
    diffs.append(diff)

    # 從差異圖 (熱度圖) 讀回來，取紅通道作為高差異指示
    image = cv2.imread(img_path)       # BGR
    red_channel = image[:, :, 2]

    # 只取 polygon_points.npy 指定的多邊形內部 (避免背景干擾)
    polygon = np.load("polygon_points.npy").astype(np.int32)
    mask = np.zeros_like(red_channel, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    masked_red = cv2.bitwise_and(red_channel, red_channel, mask=mask)

    # 二值化 + 輕微膨脹 → 找外部輪廓
    _, thresh = cv2.threshold(masked_red, hot, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, np.ones((1, 1), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxed_image = image.copy()
    area_threshold = 30
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    index = 0
    for cnt in contours:
        if index > 10:  # 最多取 11 個 ；原邏輯保留
            break
        area = cv2.contourArea(cnt)
        if area < area_threshold:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 7 or h < 7:
            # 若整體已很正常(低誤差/高SSIM)，太小的區塊多半是噪音 → 跳過
            if (error0 <=0.002 and ssim0 >=0.89) :
                 continue
            # 邊界修正避免超出影像
            w = max(w, 7)
            h = max(h, 7)
            if x + w > image.shape[1]:
                w = image.shape[1] - x
            if y + h > image.shape[0]:
                h = image.shape[0] - y

            # 擷取 ROI 並計算 ROI 級的誤差/SSIM
            roi_o = orig[y:y+h, x:x+w]
            roi_r = recon[y:y+h, x:x+w]
            d,e = calculate_error(roi_o, roi_r)
            s = calculate_ssim(roi_o, roi_r)
            # 多條件過濾「看起來正常」的區域
            if (e <= 0.011) or (s >= 0.85) or  (e < 0.03 and s > 0.6 ):
                continue
            else:
                rois_o.append(roi_o); rois_r.append(roi_r); errs.append(e); diffs.append(d); ssims_.append(s)
        else:   
            roi_o = orig[y:y+h, x:x+w]
            roi_r = recon[y:y+h, x:x+w]
            d,e = calculate_error(roi_o, roi_r)
            s = calculate_ssim(roi_o, roi_r)
            if (e <= 0.011)  :
                continue
            else:
                rois_o.append(roi_o); rois_r.append(roi_r); errs.append(e); diffs.append(d); ssims_.append(s)

        # 在差異圖上畫框，覆寫儲存 (可視化檢查)
        cv2.rectangle(boxed_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        index += 1
        cv2.imwrite(img_path, boxed_image)

    return rois_o, rois_r, diffs, errs, ssims_

# ===============================
# 8) 單張影像主流程：重建 → 全圖比較 → ROI 偵測 → 輸出
#    (此函式依賴上層設定的 input_folder/output_folder/filename/col_out)
# ===============================
def Reconstructed(vae):
    # 逐檔路徑 (由外層批次流程提供)
    image_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    # 以相同模型，對兩種前處理版本重建兩次 (增加穩定性)
    original_img, reconstructed_img , original_img2, reconstructed_img3 = reconstruct_image(image_path, vae)
    original_img, reconstructed_img2, original_img2, reconstructed_img4 = reconstruct_image(image_path, vae)

    # 對兩次重建取平均誤差/SSIM，先做全圖級的初判
    difference1, error_score1, ssim_score1 = first_comparison(original_img, reconstructed_img, reconstructed_img2)

    # 將全圖差異輸出成 "hot" 圖供 ROI 偵測用 (之後 findContours 會讀回來)
    plt.imsave(col_out, difference1, cmap='hot')

    # 若全圖看起來「偏正常」(低誤差或高 SSIM)，再往局部 ROI 深挖
    if error_score1<=0.0059  or ssim_score1>=0.79 :
        rois_o, rois_r, diffs, errs, ssims_ = findContours(original_img, reconstructed_img, col_out, 200)
        ssim_score1 = np.mean(ssims_)
        error_score1 = np.mean(errs)

        # 若第一輪 ROI 結果仍顯示「不太正常」，提高熱圖閾值再抓一次 (提高 precision)
        if error_score1>=0.0018  and ssim_score1<=0.89 :
            # 以另一組重建差異 (未平方) 作為輔助，提升弱缺陷的敏感度
            difference5 = np.abs((original_img2 - reconstructed_img3))
            difference6 = np.abs((original_img2 - reconstructed_img4))
            difference7 = np.abs((difference5 + difference6)/2)
            plt.imsave(col_out, difference7, cmap='hot')

            # 第二輪 ROI 偵測 (更嚴格）
            rois_o, rois_r, diffs, errs, ssims_ = findContours(original_img2, reconstructed_img3, col_out, 250)
            ssim_score1 = np.mean(ssims_)
            error_score1 = np.mean(errs)

            # 依 ROI 數量決定輸出排版
            if len(ssims_)==len(errs)==1:
                show_1(rois_o[0], rois_r[0],diffs[0], output_path,ssim_score1,error_score1)
            else:
                show_2(rois_o, rois_r, diffs, errs, ssims_, output_path, ssim_score1, error_score1)
        else:
            # 第一輪已足夠
            if len(ssims_)==len(errs)==1:
                show_1(rois_o[0], rois_r[0],diffs[0], output_path,ssim_score1,error_score1)
            else:
                show_2(rois_o, rois_r, diffs, errs, ssims_, output_path, ssim_score1, error_score1)
    else:
        # 全圖就已顯示「偏異常」→ 直接輸出整張對照與差異
        show_1(original_img, reconstructed_img,difference1, output_path,ssim_score1,error_score1)

    return ssim_score1,error_score1

# ===============================
# 9) 視覺化輸出：單張 / 多 ROI
# ===============================
def show_1(original, reconstructed,difference, output_path,ssim,error):
    """
    單張版：原圖/重建/差異 1×3 排列。
    根據 (SSIM>=0.83 且 Error<=0.02) 標示 Defect-free / Defect。
    """
    if ssim >= 0.83 and error<=0.02:
        txt,cor="Defect-free " ,'green'
    else:
        txt,cor="Defect " ,'red'
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
    plt.suptitle(txt, fontsize=20 , color=cor)

    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed, cmap="gray")
    plt.title("Reconstructed Image")
    plt.axis("off")
    plt.suptitle(txt, fontsize=20 , color=cor)

    plt.subplot(1, 3, 3)
    plt.imshow(difference, cmap="hot")
    plt.title(f"Difference Map\nError: {error:.4f}   ssim:{ssim:.4f}")
    plt.axis("off")
    plt.suptitle(txt, fontsize=20 , color=cor)
    
    plt.savefig(output_path)
    plt.close()

def show_2(rois_o, rois_r, diffs, errs, ssims_, output_path, st,et):
    """
    多 ROI 版：每列顯示一個 ROI 的 原圖/重建/差異。
    頂端標題依 (st>=0.83 且 et<=0.02) 標示 Defect-free / Defect。
    """
    # 強制 axes 為 2D 陣列，避免只有一列時出現 1D 陣列導致索引錯誤
    fig, axes = plt.subplots(len(rois_o), 3, figsize=(12, 2 * len(rois_o)), squeeze=False)
    
    for i, (o, r, d, err, s) in enumerate(zip(rois_o, rois_r, diffs, errs, ssims_)):
        # 顯示原始 ROI
        ax = axes[i, 0]
        ax.imshow(o, cmap="gray", aspect='equal')
        ax.set_title(f"ROI {i+1}: Original")
        ax.axis('off')

        # 顯示重建 ROI
        ax = axes[i, 1]
        ax.imshow(r, cmap="gray", aspect='equal')
        ax.set_title(f"Reconstructed")
        ax.axis('off')

        # 顯示差異圖
        ax = axes[i, 2]
        ax.imshow(d, cmap="hot", aspect='equal')
        ax.set_title(f"Difference  Error={err:.4f}, SSIM={s:.4f}")
        ax.axis('off')

    if st >= 0.83 and et<=0.02:
        plt.suptitle("Defect-Free ", fontsize=20, color='green')
    else:
        plt.suptitle("Defect ", fontsize=20, color='red')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4,top=0.89)
    plt.savefig(output_path)
    plt.close()

# ===============================
# 10) 12 片切割與相機流程：segmentation() + main loop
# ===============================
SSIM_THRESHOLD =0.83
ERROR_THRESHOLD =0.02

def segmentation(image_path,t, ssim_data, error_data, output_path):
    """
    目的：
      - 對單張影像進行 HSV 遮罩 + 形態學處理，找出 12 張 SD 卡輪廓。
      - 對每個輪廓區域進行旋轉校正後裁切，依序丟到 D1~D4 資料夾 (每 3 張為一組)。
      - 若 t==1，會把 ROI 的 NG/PASS 資訊畫到整張圖上並另存 (output_path)，同時顯示視窗。
    參數：
      image_path:   要處理的影像路徑 (擷取到的整張圖)
      t:            0=只切割儲存；1=切割並在原圖標示 NG/PASS 與編號
      ssim_data:    ROI 的 SSIM 清單 (與 error_data 對應)
      error_data:   ROI 的 Error 清單
      output_path:  若 t=1，標記後的整張輸出位置
    """
    image = cv2.imread(image_path)
    img_contours = image.copy()

    # --- 提升前景與卡片邊緣可見度 ---
    img = cv2.GaussianBlur(image, (5, 5), 0)
    h_min,s_min,v_min = 0, 0, 110
    h_max,s_max,v_max = 179, 255, 255
    contrast = 100
    brightness =0
    imgc = img.copy()
    imgc = imgc * (contrast/127 + 1) - contrast + brightness  # 對比/亮度線性調整
    imgc = np.clip(imgc, 0, 255).astype(np.uint8)

    # --- HSV 遮罩：取亮度高的區域 ---
    hsv = cv2.cvtColor(imgc,cv2.COLOR_BGR2HSV)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(hsv,lower,upper)

    # --- 形態學處理：先開再閉，去雜訊填空洞 ---
    kernel = np.ones((7, 7), np.uint8) 
    ope = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 
    kernel2 = np.ones((7, 7), np.uint8)
    clo = cv2.morphologyEx(ope, cv2.MORPH_CLOSE, kernel2) 
    kernel_dilated = np.ones((1,1), np.uint8)
    dilated = cv2.dilate(clo, kernel_dilated, iterations=2)

    if image is None:
        print("Error: Unable to load image!")
        exit()

    # --- 將畫面以 4×3 網格切區，幫助排序輪廓 ---
    height, width = image.shape[:2]
    rows, cols = 4, 3
    cell_height, cell_width = height // rows, width // cols

    # 找輪廓 → 以面積過濾 → 依區塊 (row,col) + 左上座標排序
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas)
    filtered_contours = [contours[i] for i in range(len(contours)) if areas[i] > max_area / 2]

    contours_sorted = []
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        row_idx = y // cell_height
        col_idx = x // cell_width
        region_idx = row_idx * cols + col_idx  # 區塊編號 (0~11)
        contours_sorted.append((region_idx, contour, x, y))

    # 先按區塊，再按座標排序，確保輸出順序穩定
    contours_sorted.sort(key=lambda x: (x[0], x[2], x[3])) 

    for i, (region_idx, contour, x, y) in enumerate(contours_sorted):
        cv2.drawContours(img_contours, contour, -1, (0, 0, 255), 1)
        _, _, w, h = cv2.boundingRect(contour)

        # 依最小外接矩形取得旋轉角，做旋轉校正
        cropped_image = image[y:y+h, x:x+w]
        rect = cv2.minAreaRect(contour)
        angle = rect[2]  
        if angle > 45:
            angle = angle - 90  
        rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        rotated_image = cv2.warpAffine(cropped_image, rotation_matrix, (w, h))

        # 每 3 張分一組 → D1, D2, D3, D4
        group_index = i // 3
        contour_folder = os.path.join(SAVE_FOLDER, f"D{group_index + 1}")
        os.makedirs(contour_folder, exist_ok=True)

        # 輸出裁切結果 (D?/t_<save_index>-<i+1>.jpg)
        image_path = os.path.join(contour_folder, f"t_{save_index}-{i+1}.jpg")
        cv2.imwrite(image_path, rotated_image)

        # 若需要在原圖上標示 NG/PASS 與 ROI 序號
        if t==1:
            is_defective = ssim_data[i] < SSIM_THRESHOLD or error_data[i] > ERROR_THRESHOLD
            label = "NG" if is_defective else "PASS"
            color = (0, 0, 200) if is_defective else (0, 189, 0)
            z = 20 if is_defective else 40

            # 外接框與中心標註
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(image, [box], 0, color, 2)
            center_x = x + w // 2
            center_y = y + h // 2

            # 編號與結果字樣
            cv2.putText(image, f'{i+1}', (x + 8, y + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, label, (center_x - z, center_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # 輸出整張含標記的結果，並顯示視窗
            cv2.imwrite(output_path, image)
            cv2.imshow("Result", image)

# ===============================
# 11) 主程式：攝影機即時框選 + 擷取 ('c') → 切割 → VAE 批量推論 → 回寫標記 → 顯示
#     操作鍵：
#       移動框：W/S/A/D；調高：I/K；調寬：J/L；擷取：C；離開：Q
# ===============================
if __name__ == "__main__":
    
    webcam = cv2.VideoCapture(1, cv2.CAP_DSHOW) # 使用的相機索引值 (依實機情況調整)
    
    if not webcam.isOpened():
        print("can't open the camera!!!") 
    cv2.namedWindow("video", 0) 
    cv2.resizeWindow("video",1920,1080)

    # 嘗試設置原始擷取解析度 (實際是否生效取決於相機/驅動)
    webcam.set(3, 4000)  
    webcam.set(4, 3000)  

    # 互動式框選 (在即時畫面中可用 WSAD/IKJL 調整)
    CAPTURE_WIDTH = 772
    CAPTURE_HEIGHT = 816
    CROP_X = 1625  # 擷取區域左上角 X
    CROP_Y = 1088  # 擷取區域左上角 Y

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to grab frame")
            break

        # 畫面資訊
        height, width = frame.shape[:2]
        info_text = f"Resolution: {width}x{height}"
        cv2.putText(frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 在畫面上畫出擷取框與標籤
        color  = (0, 162, 255)
        label  = f"sd card {CAPTURE_HEIGHT}x{CAPTURE_WIDTH}"
        cv2.rectangle(frame, (CROP_X, CROP_Y), 
                      (CROP_X + CAPTURE_WIDTH, CROP_Y + CAPTURE_HEIGHT), color, 2)

        # 標籤底色區塊 (避免文字疊在影像上看不清)
        labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
        _x1 = CROP_X 
        _y1 = CROP_Y 
        _x2 = CROP_X+labelSize[0][0] 
        _y2 = CROP_Y-labelSize[0][1] 
        cv2.rectangle(frame, (_x1,_y1), (_x2,_y2), color, cv2.FILLED)
        cv2.putText(frame, label, (CROP_X, CROP_Y), 
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

        # 依目前框選座標擷取子畫面 (邊框內縮 2 px)
        crop_frame = frame[CROP_Y + 2 :CROP_Y + CAPTURE_HEIGHT - 2, CROP_X + 2 :CROP_X + CAPTURE_WIDTH - 2]
   
        cv2.imshow("video", frame)

        # --- 鍵盤互動：框位移/縮放 ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('w'):
            CROP_Y = max(0, CROP_Y - 1)
        elif key == ord('s'):
            CROP_Y = min(frame.shape[0] - CAPTURE_HEIGHT, CROP_Y + 1)
        elif key == ord('a'):
            CROP_X = max(0, CROP_X - 1)
        elif key == ord('d'):
            CROP_X = min(frame.shape[1] - CAPTURE_WIDTH, CROP_X + 1)
        elif key == ord('i'):
            CAPTURE_HEIGHT = min(frame.shape[0] - CROP_Y, CAPTURE_HEIGHT + 2)
        elif key == ord('k'):
            CAPTURE_HEIGHT = max(10, CAPTURE_HEIGHT - 2)
        elif key == ord('j'):
            CAPTURE_WIDTH = max(10, CAPTURE_WIDTH - 2)
        elif key == ord('l'):
            CAPTURE_WIDTH = min(frame.shape[1] - CROP_X, CAPTURE_WIDTH + 2)

        # --- 擷取 ('c')：存檔 → 12區塊切割 → VAE 批次推論 → 回寫標記 ---
        if key == ord('c'): 
            name, errs, ssims_,path = [], [], [], []
            print(f"({CROP_X},{CROP_Y}) {CAPTURE_HEIGHT}x{CAPTURE_WIDTH}")

            # 1) 存下擷取框的子圖
            save_path = os.path.join(SAVE_FOLDER, f"{SAVE_img_name}_{save_index}.jpg")  
            cv2.imwrite(save_path,crop_frame)
            print(f"已擷取影像: {save_path}")

            # 2) 分割 12 區塊，按 D1~D4 子資料夾輸出 (t=0 僅切割不標記)
            segmentation(save_path,0,ssims_,errs,save_path)

            # 3) 為每個 D? 底下影像跑 VAE 推論，輸出結果到 test/result/
            folder = SAVE_FOLDER
            output_folder = rf"{folder}\result"
            ssim_log = rf"{output_folder}\ssim_log.txt"
            colored_folder = f"{output_folder}\Defect Location"
            os.makedirs(output_folder, exist_ok=True)
            
            with open(ssim_log, "w") as log_file:
                for i in range(4):
                    txt=["D1","D2","D3","D4"]
                    input_folder = rf"{folder}\{txt[i]}"
                    for filename in os.listdir(input_folder):
                        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                            # 每張影像的差異熱圖 (含框) 存放位置；Reconstructed() 會使用 col_out
                            col_out = f"{colored_folder}\{filename}.png"
                            # 單張主流程：重建/比較/ROI/輸出，並回傳這張的平均 SSIM/Error
                            ssim_end,error_end = Reconstructed(vae_models[i])

                            # 記錄到 log 與清單 (後續回寫到整張圖)
                            log_file.write(f"{filename}\t{ssim_end:.4f}\t{error_end:.4f}\n")
                            name.append(filename); ssims_.append(ssim_end); errs.append(error_end); path.append(f"{output_folder}\{filename}")
                            print(f"Processed: {filename}, SSIM: {ssim_end:.4f}, Error: {error_end:.4f}, path: {output_folder}\{filename}")

            # 4) 回到整張擷取畫面做 NG/PASS 標記 (t=1 會在原圖上用 ssim/err 清單畫標)
            save_path2 = os.path.join(SAVE_FOLDER, f"{SAVE_img_name}_{save_index}_ans.jpg")
            segmentation(save_path,1,ssims_,errs,save_path2)
            print(f"✅ 所有圖片處理完成！")

        # --- 離開 ---
        elif key == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

