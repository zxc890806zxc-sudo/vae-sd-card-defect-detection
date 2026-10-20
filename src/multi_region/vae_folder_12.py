import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# 設定影像大小與參數
IMG_SIZE = (256, 350)
LATENT_DIM = 160  # 潛在空間維度

# ----------------- VAE 架構 ----------------- #
class VAE_Encoder(layers.Layer):
    def __init__(self, latent_dim):
        super(VAE_Encoder, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")
        self.pool1 = layers.MaxPooling2D((2, 2), padding="same")
        self.conv2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")
        self.pool2 = layers.MaxPooling2D((2, 2), padding="same")
        self.flatten = layers.Flatten()
        self.mean = layers.Dense(latent_dim)
        self.log_var = layers.Dense(latent_dim)

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
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE_Decoder(layers.Layer):
    def __init__(self):
        super(VAE_Decoder, self).__init__()
        self.dense = layers.Dense(64 * 88 * 64, activation="relu")
        self.reshape = layers.Reshape((64, 88, 64))
        self.deconv1 = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same")
        self.up1 = layers.UpSampling2D((2, 2))  
        self.deconv2 = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same")
        self.up2 = layers.UpSampling2D((2, 2))  
        self.crop = layers.Cropping2D(((0, 0), (1, 1)))
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
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(latent_dim)
        self.decoder = VAE_Decoder()

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = sampling(z_mean, z_log_var)
        return self.decoder(z)

vae_models = []
for i in range(4):
    model = VAE(LATENT_DIM)
    dummy_input = np.zeros((1, 256, 350, 1))
    _ = model(dummy_input)
    model.load_weights(f"vae_D{i+1}.weights.h5")
    vae_models.append(model)
    print(f"✅ 第{i+1}個 VAE 權重載入成功！")

def reconstruct_image(image_path, vae_model):
    img1 = load_img(image_path, target_size=IMG_SIZE, color_mode="grayscale")
    img1 = np.array(img1, dtype=np.uint8)
    img2 = cv2.GaussianBlur(img1, (3, 3), 0)
    img1 = cv2.bilateralFilter(img2, 20, 35, 100)
    img2 = cv2.bilateralFilter(img2, 20, 35, 100)
    img1 = img_to_array(img1) / 255.0
    img2 = img_to_array(img2) / 255.0
    rec1 = vae_model(np.expand_dims(img1, 0)).numpy().squeeze()
    rec2 = vae_model(np.expand_dims(img2, 0)).numpy().squeeze()
    return img1.squeeze(), rec1, img2.squeeze(), rec2

# ----------------- 影像重建 ----------------- #
def calculate_error(original, reconstructed):
    difference = np.abs((original - reconstructed)**2)
    error_value = np.mean(difference)
    return difference,float(error_value)

def calculate_ssim(original, reconstructed):
    return float(ssim(original, reconstructed, data_range=1.0))

def first_comparison(orig, recon, recon2):
    diff1,error1 =calculate_error(orig, recon)
    diff2,error2 =calculate_error(orig, recon2)
    diff = np.abs((diff1 + diff2)/2)
    error =(error1+error2)/2
    ssim1=calculate_ssim(orig, recon)
    ssim2=calculate_ssim(orig, recon2)
    ssim = (ssim1 + ssim2 ) / 2
    return  diff, error, ssim

def findContours(orig, recon, img_path, hot):
    rois_o, rois_r, diffs, errs, ssims_ = [], [], [], [], []
    rois_o.append(orig); rois_r.append(recon)
    diff,error0 =calculate_error(orig, recon)
    errs.append(error0)
    ssim0 = calculate_ssim(orig, recon)
    ssims_.append(ssim0)
    diffs.append(diff)

    image = cv2.imread(img_path)
    red_channel = image[:, :, 2]
    polygon = np.load("polygon_points.npy").astype(np.int32)
    mask = np.zeros_like(red_channel, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    masked_red = cv2.bitwise_and(red_channel, red_channel, mask=mask)
    _, thresh = cv2.threshold(masked_red, hot, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, np.ones((1, 1), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxed_image = image.copy()
    cv2.polylines(boxed_image, [polygon], isClosed=True, color=(255, 0, 0), thickness=1)
    area_threshold = 30
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    index = 0
    for cnt in contours:
        if index >= 10:
            break
        area = cv2.contourArea(cnt)
        if area < area_threshold:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 7 or h < 7:
            if (error0 <=0.002 and ssim0 >=0.89) :
                 continue
            w = max(w, 7)
            h = max(h, 7)
            if x + w > image.shape[1]:
                w = image.shape[1] - x
            if y + h > image.shape[0]:
                h = image.shape[0] - y
            roi_o = orig[y:y+h, x:x+w]
            roi_r = recon[y:y+h, x:x+w]
            d,e = calculate_error(roi_o, roi_r)
            s = calculate_ssim(roi_o, roi_r)
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

        cv2.rectangle(boxed_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        index += 1
        cv2.imwrite(img_path, boxed_image)

    return rois_o, rois_r, diffs, errs, ssims_

def Reconstructed(vae):
    image_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    original_img, reconstructed_img , original_img2, reconstructed_img3= reconstruct_image(image_path, vae)
    original_img, reconstructed_img2, original_img2, reconstructed_img4 = reconstruct_image(image_path, vae)
    difference1,error_score1,ssim_score1 = first_comparison(original_img, reconstructed_img, reconstructed_img2)
    plt.imsave(col_out, difference1, cmap='hot')
    if error_score1<=0.0059  or ssim_score1>=0.79 :
        rois_o, rois_r, diffs, errs, ssims_ = findContours(original_img, reconstructed_img,col_out, 200)
        ssim_score1 = np.mean(ssims_)
        error_score1 = np.mean(errs)
        if error_score1>=0.0018  and ssim_score1<=0.89 :
            difference5 = np.abs((original_img2 - reconstructed_img3))
            difference6 = np.abs((original_img2 - reconstructed_img4))
            difference7 = np.abs((difference5 + difference6)/2)
            plt.imsave(col_out, difference7, cmap='hot')
            rois_o, rois_r, diffs, errs, ssims_ = findContours(original_img2, reconstructed_img3,col_out, 250)
            ssim_score1 = np.mean(ssims_)
            error_score1 = np.mean(errs)
            if len(ssims_)==len(errs)==1:
                show_images_1(rois_o[0], rois_r[0],diffs[0], output_path,ssim_score1,error_score1)
            else:
                display_results_matplotlib(rois_o, rois_r, diffs, errs, ssims_, output_path,ssim_score1,error_score1)
        else:
            if len(ssims_)==len(errs)==1:
                show_images_1(rois_o[0], rois_r[0],diffs[0], output_path,ssim_score1,error_score1)
            else:
                display_results_matplotlib(rois_o, rois_r, diffs, errs, ssims_, output_path,ssim_score1,error_score1)
    else:
        show_images_1(original_img, reconstructed_img,difference1, output_path,ssim_score1,error_score1)
    
    return ssim_score1,error_score1

# ----------------- 解果顯示 ----------------- #
def show_images_1(original, reconstructed,difference, output_path,ssim,error):
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

def display_results_matplotlib(rois_o, rois_r, diffs, errs, ssims_, output_path, st,et):
    fig, axes = plt.subplots(len(rois_o), 3, figsize=(12, 2 * len(rois_o)), squeeze=False)
    
    for i, (o, r, d, err, s) in enumerate(zip(rois_o, rois_r, diffs, errs, ssims_)):
        ax = axes[i, 0]
        ax.imshow(o, cmap="gray", aspect='equal')
        ax.set_title(f"ROI {i+1}: Original")
        ax.axis('off')

        ax = axes[i, 1]
        ax.imshow(r, cmap="gray", aspect='equal')
        ax.set_title(f"Reconstructed")
        ax.axis('off')

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

# -----------執行整個資料夾的影像修復----------- #
for j in range(2):
    txt_1=["defect","defect-free"]
    folder =f"test-image\{txt_1[j]}"
    for i in range(4):
        txt=["D1","D2","D3","D4"]
        # 檔案路徑
        input_folder = f"{folder}\{txt[i]}"
        output_folder = f"result\{txt_1[j]}\{txt[i]}"
        ssim_log = f"{output_folder}\ssim_log.txt"
        colored_folder = f"{output_folder}\Defect Location"
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(colored_folder, exist_ok=True)

        with open(ssim_log, "w") as log_file:
            for filename in os.listdir(input_folder):
                if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                    col_out = f"{colored_folder}\{filename}.png"
                    ssim_end,error_end = Reconstructed(vae_models[i])
                    log_file.write(f"{filename}\t{ssim_end:.4f}\t{error_end:.4f}\n")
                    print(f"Processed: {filename}, SSIM: {ssim_end:.4f}, Error: {error_end:.4f}")

print("✅ 所有圖片處理完成！")

