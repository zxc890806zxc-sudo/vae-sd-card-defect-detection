import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

SAVE_FOLDER = "test"
SAVE_img_name = "t"
os.makedirs(SAVE_FOLDER, exist_ok=True)
save_index = 0  

# 設定影像大小
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
    area_threshold = 30
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    index = 0
    for cnt in contours:
        if index > 10:
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
                show_1(rois_o[0], rois_r[0],diffs[0], output_path,ssim_score1,error_score1)
            else:
                show_2(rois_o, rois_r, diffs, errs, ssims_, output_path, ssim_score1, error_score1)
        else:
            if len(ssims_)==len(errs)==1:
                show_1(rois_o[0], rois_r[0],diffs[0], output_path,ssim_score1,error_score1)
            else:
                show_2(rois_o, rois_r, diffs, errs, ssims_, output_path, ssim_score1, error_score1)
    else:
        show_1(original_img, reconstructed_img,difference1, output_path,ssim_score1,error_score1)
    return ssim_score1,error_score1

# ----------------- 解果顯示 ----------------- #
def show_1(original, reconstructed,difference, output_path,ssim,error):
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

# ----------------- 12片切割、啟用相機 ----------------- #
SSIM_THRESHOLD =0.83
ERROR_THRESHOLD =0.02
def segmentation(image_path,t, ssim_data, error_data, output_path):
    image = cv2.imread(image_path)
    img_contours = image.copy()
    img=cv2.GaussianBlur(image, (5, 5), 0)
    h_min,s_min,v_min = 0, 0, 110
    h_max,s_max,v_max = 179, 255, 255
    contrast = 100
    brightness =0
    imgc = img.copy()
    imgc = imgc * (contrast/127 + 1) - contrast + brightness 
    imgc = np.clip(imgc, 0, 255)
    imgc = np.uint8(imgc)
    hsv = cv2.cvtColor(imgc,cv2.COLOR_BGR2HSV)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(hsv,lower,upper)
    kernel = np.ones((7, 7), np.uint8) 
    ope = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 
    kernel2 = np.ones((7, 7), np.uint8)
    clo = cv2.morphologyEx(ope, cv2.MORPH_CLOSE, kernel2) 
    kernel_dilated = np.ones((1,1), np.uint8)
    dilated = cv2.dilate(clo, kernel_dilated, iterations=2)

    if image is None:
        print("Error: Unable to load image!")
        exit()

    height, width = image.shape[:2]
    rows, cols = 4, 3
    cell_height, cell_width = height // rows, width // cols
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas)
    filtered_contours = [contours[i] for i in range(len(contours)) if areas[i] > max_area / 2]
    contours_sorted = []
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        row_idx = y // cell_height
        col_idx = x // cell_width
        region_idx = row_idx * cols + col_idx
        contours_sorted.append((region_idx, contour, x, y))
    contours_sorted.sort(key=lambda x: (x[0], x[2], x[3])) 
    for i, (region_idx, contour, x, y) in enumerate(contours_sorted):
        cv2.drawContours(img_contours, contour, -1, (0, 0, 255), 1)
        _, _, w, h = cv2.boundingRect(contour)
        cropped_image = image[y:y+h, x:x+w]
        rect = cv2.minAreaRect(contour)
        angle = rect[2]  
        if angle > 45:
            angle = angle - 90  
        #print(f"輪廓 {i+1} 旋轉角度: {angle}")
        rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        rotated_image = cv2.warpAffine(cropped_image, rotation_matrix, (w, h))
        group_index = i // 3
        contour_folder = os.path.join(SAVE_FOLDER, f"D{group_index + 1}")
        os.makedirs(contour_folder, exist_ok=True)
        image_path = os.path.join(contour_folder, f"t_{save_index}-{i+1}.jpg")
        cv2.imwrite(image_path, rotated_image)
        if t==1:
            is_defective = ssim_data[i] < SSIM_THRESHOLD or error_data[i] > ERROR_THRESHOLD
            label = "NG" if is_defective else "PASS"
            color = (0, 0, 200) if is_defective else (0, 189, 0)
            z = 20 if is_defective else 40
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            cv2.drawContours(image, [box], 0, color, 2)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.putText(image, f'{i+1}', (x + 8, y + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, label, (center_x - z, center_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.imwrite(output_path, image)
            cv2.imshow("Result", image)
                

if __name__ == "__main__":
    
    webcam = cv2.VideoCapture(1, cv2.CAP_DSHOW) #使用的相機索引值
    
    if not webcam.isOpened():
        print("can't open the camera!!!") 
    cv2.namedWindow("video", 0) 
    cv2.resizeWindow("video",1920,1080)
    webcam.set(3, 4000)  
    webcam.set(4, 3000)  
    CAPTURE_WIDTH = 772
    CAPTURE_HEIGHT = 816
    CROP_X = 1625  # 擷取區域的左上角 X 座標
    CROP_Y = 1088  # 擷取區域的左上角 Y 座標

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to grab frame")
            break
        height, width = frame.shape[:2]
        info_text = f"Resolution: {width}x{height}"
        cv2.putText(frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        color  = (0, 162, 255)
        label  = f"sd card {CAPTURE_HEIGHT}x{CAPTURE_WIDTH}"
        cv2.rectangle(frame, (CROP_X, CROP_Y), 
                      (CROP_X + CAPTURE_WIDTH, CROP_Y + CAPTURE_HEIGHT), color, 2)
        labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
        _x1 = CROP_X 
        _y1 = CROP_Y 
        _x2 = CROP_X+labelSize[0][0] 
        _y2 = CROP_Y-labelSize[0][1] 
        cv2.rectangle(frame, (_x1,_y1), (_x2,_y2), color, cv2.FILLED)
        cv2.putText(frame, label, (CROP_X, CROP_Y), 
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        crop_frame = frame[CROP_Y + 2 :CROP_Y + CAPTURE_HEIGHT - 2, CROP_X + 2 :CROP_X + CAPTURE_WIDTH - 2]
   
        cv2.imshow("video", frame)
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

            
        if key == ord('c'): 
            name, errs, ssims_,path = [], [], [], []
            print(f"({CROP_X},{CROP_Y}) {CAPTURE_HEIGHT}x{CAPTURE_WIDTH}")
            save_path = os.path.join(SAVE_FOLDER, f"{SAVE_img_name}_{save_index}.jpg")  
            cv2.imwrite(save_path,crop_frame)
            print(f"已擷取影像: {save_path}")
            segmentation(save_path,0,ssims_,errs,save_path)

            # 檔案路徑
            folder =SAVE_FOLDER
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
                            col_out = f"{colored_folder}\{filename}.png"
                            ssim_end,error_end = Reconstructed(vae_models[i])
                            log_file.write(f"{filename}\t{ssim_end:.4f}\t{error_end:.4f}\n")
                            name.append(filename);ssims_.append(ssim_end);errs.append(error_end);path.append(f"{output_folder}\{filename}")
                            print(f"Processed: {filename}, SSIM: {ssim_end:.4f}, Error: {error_end:.4f}, path: {output_folder}\{filename}")

            save_path2 = os.path.join(SAVE_FOLDER, f"{SAVE_img_name}_{save_index}_ans.jpg")
            segmentation(save_path,1,ssims_,errs,save_path2)
            print(f"✅ 所有圖片處理完成！")
        elif key == ord('q'):
            break
    webcam.release()
    cv2.destroyAllWindows()

