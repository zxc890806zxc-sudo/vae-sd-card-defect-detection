import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# ======================
# 基本全域參數設定
# ======================
IMG_SIZE = (256, 350)   # 所有輸入影像會統一縮放到的大小 (height=256, width=350)
LATENT_DIM = 160        # VAE 潛在向量的維度數量，控制壓縮後的特徵容量


# ======================
# VAE 編碼器 (Encoder)
# 負責將輸入影像壓縮成 latent vector (z_mean, z_log_var)
# ======================
class VAE_Encoder(layers.Layer):
    def __init__(self, latent_dim):
        super(VAE_Encoder, self).__init__()
        # 第一層卷積：擷取低階特徵 (邊緣、紋理等)
        self.conv1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")
        # 最大池化：降低解析度，保留主要特徵
        self.pool1 = layers.MaxPooling2D((2, 2), padding="same")

        # 第二層卷積：擷取更高階的特徵
        self.conv2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")
        # 再次下採樣
        self.pool2 = layers.MaxPooling2D((2, 2), padding="same")

        # 平坦化，準備進入全連接層
        self.flatten = layers.Flatten()

        # 輸出 latent vector 的平均值 (μ) 與對數方差 (log(σ²))
        # μ 和 log_var 會用來進行重參數化
        self.mean = layers.Dense(latent_dim)
        self.log_var = layers.Dense(latent_dim)

    def call(self, inputs):
        # 卷積與池化提取特徵
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        # 展平成一維
        x = self.flatten(x)

        # 得到 μ 和 log(σ²)
        z_mean = self.mean(x)
        z_log_var = self.log_var(x)
        return z_mean, z_log_var


# ======================
# 重參數化技巧 (Reparameterization Trick)
# 從 μ 和 log(σ²) 生成隨機的 z (latent vector)
# ======================
def sampling(z_mean, z_log_var):
    # 產生與 z_mean 相同 shape 的標準常態分佈隨機數
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
    # z = μ + σ * ε
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ======================
# VAE 解碼器 (Decoder)
# 負責將 latent vector 還原成影像
# ======================
class VAE_Decoder(layers.Layer):
    def __init__(self):
        super(VAE_Decoder, self).__init__()
        # 將 latent vector 展開成特徵圖 (64x88x64)
        self.dense = layers.Dense(64 * 88 * 64, activation="relu")
        self.reshape = layers.Reshape((64, 88, 64))

        # 反卷積 + 上採樣逐步恢復影像尺寸
        self.deconv1 = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same")
        self.up1 = layers.UpSampling2D((2, 2))  # 高寬各放大 2 倍

        self.deconv2 = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same")
        self.up2 = layers.UpSampling2D((2, 2))

        # 修剪邊界以符合輸出尺寸
        self.crop = layers.Cropping2D(((0, 0), (1, 1)))

        # 最後輸出一張灰階影像 (sigmoid 確保像素值在 0~1)
        self.output_layer = layers.Conv2DTranspose(1, (3, 3), activation="sigmoid", padding="same")

    def call(self, inputs):
        # 展開 latent vector
        x = self.dense(inputs)
        x = self.reshape(x)

        # 逐步放大回原始尺寸
        x = self.deconv1(x)
        x = self.up1(x)
        x = self.deconv2(x)
        x = self.up2(x)

        # 修正尺寸差異
        x = self.crop(x)

        # 生成重建影像
        return self.output_layer(x)


# ======================
# VAE 主架構
# ======================
class VAE(keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(latent_dim)  # 編碼器
        self.decoder = VAE_Decoder()            # 解碼器

    def call(self, inputs):
        # 編碼 → 重參數化 → 解碼
        z_mean, z_log_var = self.encoder(inputs)
        z = sampling(z_mean, z_log_var)
        return self.decoder(z)


# ======================
# 載入已訓練好的 4 個 VAE 權重檔
# 每個模型負責對應不同資料區域 (D1~D4)
# ======================
vae_models = []
for i in range(4):
    model = VAE(LATENT_DIM)
    _ = model(np.zeros((1, 256, 350, 1)))  # 先跑一次假資料建立模型結構
    model.load_weights(f"vae_D{i+1}.weights.h5")
    vae_models.append(model)
    print(f"✅ 第{i+1}個 VAE 權重載入成功！")


# ======================
# reconstruct_image()
# 功能：將輸入影像進行預處理後，用 VAE 重建
# 輸出：原圖1、重建圖1、原圖2、重建圖2 (不同預處理版本)
# ======================
def reconstruct_image(image_path, vae_model):
    # 讀取影像並轉灰階，縮放到統一大小
    img1 = load_img(image_path, target_size=IMG_SIZE, color_mode="grayscale")
    img1 = np.array(img1, dtype=np.uint8)

    # 套用高斯模糊
    img2 = cv2.GaussianBlur(img1, (3, 3), 0)

    # 對兩張版本進行雙邊濾波 (保留邊緣)
    img1 = cv2.bilateralFilter(img2, 20, 35, 100)
    img2 = cv2.bilateralFilter(img2, 20, 35, 100)

    # 轉為 float 並正規化到 [0,1]
    img1 = img_to_array(img1) / 255.0
    img2 = img_to_array(img2) / 255.0

    # 分別送入 VAE 進行重建
    rec1 = vae_model(np.expand_dims(img1, 0)).numpy().squeeze()
    rec2 = vae_model(np.expand_dims(img2, 0)).numpy().squeeze()

    return img1.squeeze(), rec1, img2.squeeze(), rec2

# ----------------- 影像重建後的評估工具 ----------------- #
def calculate_error(original, reconstructed):
    """
    計算像素級平方誤差 (per-pixel squared error) 以及其平均值。
    回傳:
        difference: 每個像素 (o - r)^2 的矩陣 (保留做差異圖用)
        error_value: 所有像素誤差的平均 (單一浮點數，做判斷用)
    """
    difference = np.abs((original - reconstructed)**2)
    error_value = np.mean(difference)
    return difference, float(error_value)

def calculate_ssim(original, reconstructed):
    """
    計算 SSIM (結構相似度)，範圍通常是 [-1, 1]，越接近 1 越相似。
    data_range=1.0 因為我們的畫素值已正規化到 [0, 1]。
    """
    return float(ssim(original, reconstructed, data_range=1.0))

def first_comparison(orig, recon, recon2):
    """
    對同一張影像的兩個重建結果進行「平均比較」：
      1) 兩個 (orig vs recon)、(orig vs recon2) 的平方誤差圖取平均，得到更穩定的 difference。
      2) 兩個平均誤差值取平均。
      3) 兩個 SSIM 取平均。
    用意：降低單一路徑預處理造成的偶然波動。
    """
    diff1, error1 = calculate_error(orig, recon)
    diff2, error2 = calculate_error(orig, recon2)
    diff  = np.abs((diff1 + diff2) / 2)
    error = (error1 + error2) / 2
    ssim1 = calculate_ssim(orig, recon)
    ssim2 = calculate_ssim(orig, recon2)
    ssim_avg = (ssim1 + ssim2) / 2
    return diff, error, ssim_avg


# ----------------- 缺陷區域偵測：從差異圖導引 ROI ----------------- #
def findContours(orig, recon, img_path, hot):
    """
    依據差異圖 (已事先以 plt.imsave 存到 img_path，並使用 'hot' colormap) 來定位可能的缺陷區域。
    流程：
      1) 讀取差異圖檔 (彩色)，取紅色通道 (hot 色表中，紅色對應高差異)。
      2) 使用 polygon_points.npy 定義的多邊形區域作為有效檢測範圍 (mask)。
      3) 對紅通道套用閾值 (hot) → 具體化差異區塊 → 膨脹 → 找輪廓。
      4) 每個輪廓取 bounding box，擷取對應 ROI (orig 與 recon)，計算 ROI 的誤差與 SSIM；
         再用條件過濾掉「誤差很低/SSIM很高」的明顯正常區域。
      5) 最多保留 10 個候選區域，並在差異圖上畫框覆寫回 img_path。
    回傳:
      rois_o, rois_r, diffs, errs, ssims_ (各為 list；第 0 筆為整張圖的 (orig, recon, diff, error0, ssim0))
    """
    # 初始化列表，先放入「整張圖」的評估 (作為基準)
    rois_o, rois_r, diffs, errs, ssims_ = [], [], [], [], []
    rois_o.append(orig)
    rois_r.append(recon)

    diff, error0 = calculate_error(orig, recon)
    diffs.append(diff)
    errs.append(error0)
    ssim0 = calculate_ssim(orig, recon)
    ssims_.append(ssim0)

    # 讀入差異圖 (colormap=hot)，取紅色通道作為高差異指示
    image = cv2.imread(img_path)       # BGR
    red_channel = image[:, :, 2]       # 紅色通道

    # 只在指定 polygon 範圍內找差異：避免背景或無關區域干擾
    polygon = np.load("polygon_points.npy").astype(np.int32)  # shape=(N,2)
    mask = np.zeros_like(red_channel, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    masked_red = cv2.bitwise_and(red_channel, red_channel, mask=mask)

    # 閾值化 (hot)：紅通道超過某強度視為可疑；再做輕微膨脹，讓輪廓更完整
    _, thresh = cv2.threshold(masked_red, hot, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, np.ones((1, 1), np.uint8), iterations=1)

    # 找外部輪廓，依面積由大到小排序；畫出多邊形邊界與候選框 (綠框)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxed_image = image.copy()
    cv2.polylines(boxed_image, [polygon], isClosed=True, color=(255, 0, 0), thickness=1)

    area_threshold = 30
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    index = 0
    for cnt in contours:
        if index >= 10:          # 只保留最多 10 個 ROI，避免過多小碎片
            break

        area = cv2.contourArea(cnt)
        if area < area_threshold:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # 針對小尺寸區域：若整體誤差夠低且 SSIM 夠高，則很可能是噪音，直接略過
        if w < 7 or h < 7:
            if (error0 <= 0.002 and ssim0 >= 0.89):
                continue
            # 邊界修正，確保 ROI 不超出影像
            w = max(w, 7)
            h = max(h, 7)
            if x + w > image.shape[1]:
                w = image.shape[1] - x
            if y + h > image.shape[0]:
                h = image.shape[0] - y

            roi_o = orig[y:y+h, x:x+w]
            roi_r = recon[y:y+h, x:x+w]
            d, e = calculate_error(roi_o, roi_r)
            s = calculate_ssim(roi_o, roi_r)

            # 條件意義：
            #   - e <= 0.011：誤差很小 → 傾向正常
            #   - s >= 0.85：相似度很高 → 傾向正常
            #   - (e < 0.03 and s > 0.6)：中度相似、誤差不大 → 仍傾向正常
            if (e <= 0.011) or (s >= 0.85) or (e < 0.03 and s > 0.6):
                continue
            else:
                rois_o.append(roi_o)
                rois_r.append(roi_r)
                diffs.append(d)
                errs.append(e)
                ssims_.append(s)
        else:
            # 一般尺寸區域
            roi_o = orig[y:y+h, x:x+w]
            roi_r = recon[y:y+h, x:x+w]
            d, e = calculate_error(roi_o, roi_r)
            s = calculate_ssim(roi_o, roi_r)
            if (e <= 0.011):     # 誤差很小視為正常，直接跳過
                continue
            else:
                rois_o.append(roi_o)
                rois_r.append(roi_r)
                diffs.append(d)
                errs.append(e)
                ssims_.append(s)

        # 視覺化：在差異圖上畫出候選框
        cv2.rectangle(boxed_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        index += 1
        cv2.imwrite(img_path, boxed_image)  # 直接覆寫差異圖：包含藍色 polygon 與綠框

    return rois_o, rois_r, diffs, errs, ssims_


# ----------------- 單張影像處理主流程 ----------------- #
def Reconstructed(vae):
    """
    對單張輸入影像：
      1) 用同一個 VAE 兩次重建 (對兩種預處理)，先做「全圖級」的初步比較。
      2) 若初步顯示正常 (err 低 / ssim 高)，再用差異圖引導 ROI 偵測，集中於可疑區域。
      3) 視情況調整強度阈值 (200 → 250) 再做一次 ROI 偵測以提高召回。
      4) 視最後 ROI 結果決定以單圖或多 ROI 版面輸出；同時回寫平均 SSIM / Error。
    注意：
      - 本函式依賴外部的變數 input_folder、output_folder、filename、col_out (由批次流程設定)。
    """
    # 檔案 I/O 路徑
    image_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    # 兩次重建 (用相同 VAE，但前處理輸入略不同) → 得兩個重建版本
    original_img, reconstructed_img , original_img2, reconstructed_img3 = reconstruct_image(image_path, vae)
    original_img, reconstructed_img2, original_img2, reconstructed_img4 = reconstruct_image(image_path, vae)

    # 以兩個重建版本與原圖做「平均差異/平均 SSIM」→ 得到較穩定的全圖級差異
    difference1, error_score1, ssim_score1 = first_comparison(original_img, reconstructed_img, reconstructed_img2)

    # 先把「全圖差異」輸出成熱度圖（之後會被畫框覆蓋）
    plt.imsave(col_out, difference1, cmap='hot')

    # 如果整體看起來正常 (誤差很低或 SSIM 高)，就進一步找局部可疑區域
    if error_score1 <= 0.0059 or ssim_score1 >= 0.79:
        # 第一次 ROI 偵測：較低的 hot 閾值，抓出可疑區域
        rois_o, rois_r, diffs, errs, ssims_ = findContours(original_img, reconstructed_img, col_out, 200)
        ssim_score1 = np.mean(ssims_)
        error_score1 = np.mean(errs)

        # 若平均誤差仍偏高、平均 SSIM 偏低，再嘗試提高熱圖閾值 (250) 做第二輪
        if error_score1 >= 0.0018 and ssim_score1 <= 0.89:
            # 以第二組重建生成更強的差異圖 (未平方)，提高對弱缺陷的敏感度
            difference5 = np.abs((original_img2 - reconstructed_img3))
            difference6 = np.abs((original_img2 - reconstructed_img4))
            difference7 = np.abs((difference5 + difference6) / 2)
            plt.imsave(col_out, difference7, cmap='hot')

            # 第二次 ROI 偵測（更嚴格的 hot 閾值）
            rois_o, rois_r, diffs, errs, ssims_ = findContours(original_img2, reconstructed_img3, col_out, 250)
            ssim_score1 = np.mean(ssims_)
            error_score1 = np.mean(errs)

            # 視 ROI 數量決定輸出格式（單 ROI / 多 ROI）
            if len(ssims_) == len(errs) == 1:
                show_images_1(rois_o[0], rois_r[0], diffs[0], output_path, ssim_score1, error_score1)
            else:
                display_results_matplotlib(rois_o, rois_r, diffs, errs, ssims_, output_path, ssim_score1, error_score1)
        else:
            # 第一次偵測結果就足夠
            if len(ssims_) == len(errs) == 1:
                show_images_1(rois_o[0], rois_r[0], diffs[0], output_path, ssim_score1, error_score1)
            else:
                display_results_matplotlib(rois_o, rois_r, diffs, errs, ssims_, output_path, ssim_score1, error_score1)
    else:
        # 整體看起來就有問題 → 直接輸出整張原圖/重建圖/差異圖
        show_images_1(original_img, reconstructed_img, difference1, output_path, ssim_score1, error_score1)

    return ssim_score1, error_score1


# ----------------- 單張圖輸出（整張） ----------------- #
def show_images_1(original, reconstructed, difference, output_path, ssim, error):
    """
    將整張原圖、重建圖、差異圖以 1×3 排列並輸出。
    依照 (SSIM>=0.83 且 Error<=0.02) 決定標題為 Defect-free 或 Defect。
    """
    if ssim >= 0.83 and error <= 0.02:
        txt, cor = "Defect-free", 'green'
    else:
        txt, cor = "Defect", 'red'

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
    plt.suptitle(txt, fontsize=20, color=cor)  # 全圖標頭

    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed, cmap="gray")
    plt.title("Reconstructed Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(difference, cmap="hot")
    plt.title(f"Difference Map\nError: {error:.4f}   ssim:{ssim:.4f}")
    plt.axis("off")

    plt.savefig(output_path)
    plt.close()


# ----------------- 多 ROI 輸出（矩陣版） ----------------- #
def display_results_matplotlib(rois_o, rois_r, diffs, errs, ssims_, output_path, st, et):
    """
    將每個 ROI 的 原圖/重建/差異 以列顯示（每列一個 ROI）。
    頂端標題依 (st>=0.83 且 et<=0.02) 標示 Defect-free 或 Defect。
    注意：st, et 是整體平均 (或上一層傳入的代表值)。
    """
    fig, axes = plt.subplots(len(rois_o), 3, figsize=(12, 2 * len(rois_o)), squeeze=False)

    for i, (o, r, d, err, s) in enumerate(zip(rois_o, rois_r, diffs, errs, ssims_)):
        ax = axes[i, 0]
        ax.imshow(o, cmap="gray", aspect='equal')
        ax.set_title(f"ROI {i+1}: Original")
        ax.axis('off')

        ax = axes[i, 1]
        ax.imshow(r, cmap="gray", aspect='equal')
        ax.set_title("Reconstructed")
        ax.axis('off')

        ax = axes[i, 2]
        ax.imshow(d, cmap="hot", aspect='equal')
        ax.set_title(f"Difference  Error={err:.4f}, SSIM={s:.4f}")
        ax.axis('off')

    if st >= 0.83 and et <= 0.02:
        plt.suptitle("Defect-Free ", fontsize=20, color='green')
    else:
        plt.suptitle("Defect ", fontsize=20, color='red')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, top=0.89)
    plt.savefig(output_path)
    plt.close()


# ----------------- 批次處理整個資料夾 ----------------- #
for j in range(2):
    txt_1 = ["defect", "defect-free"]          # 兩個主資料夾
    folder = f"test-image\\{txt_1[j]}"
    for i in range(4):
        txt = ["D1", "D2", "D3", "D4"]         # 各子區域
        # I/O 資料夾規劃
        input_folder  = f"{folder}\\{txt[i]}"
        output_folder = f"result\\{txt_1[j]}\\{txt[i]}"
        ssim_log      = f"{output_folder}\\ssim_log.txt"
        colored_folder = f"{output_folder}\\Defect Location"
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(colored_folder, exist_ok=True)

        # 每個子資料夾各自記錄 SSIM / Error
        with open(ssim_log, "w") as log_file:
            for filename in os.listdir(input_folder):
                if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                    # 差異圖（含框）輸出的路徑；Reconstructed() 會覆蓋它
                    col_out = f"{colored_folder}\\{filename}.png"

                    # 執行單張流程；注意：此函式內部使用 input_folder / output_folder / filename / col_out
                    ssim_end, error_end = Reconstructed(vae_models[i])

                    # 紀錄並回報
                    log_file.write(f"{filename}\t{ssim_end:.4f}\t{error_end:.4f}\n")
                    print(f"Processed: {filename}, SSIM: {ssim_end:.4f}, Error: {error_end:.4f}")

print("✅ 所有圖片處理完成！")


