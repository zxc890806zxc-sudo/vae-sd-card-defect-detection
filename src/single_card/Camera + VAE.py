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
# ------------------------------
# 參數設定
# ------------------------------
IMG_SIZE = (256, 350)
LATENT_DIM = 160 
WEIGHTS_FP = "vae_model.weights.h5"
# ------------------------------
# 定義 VAE 結構
# ------------------------------
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
        x = self.crop(x)  # 裁剪成 (256,350)
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

vae = VAE(LATENT_DIM)
dummy_input = np.zeros((1, *IMG_SIZE, 1))  
_ = vae(dummy_input)  
vae.load_weights( WEIGHTS_FP )  
print("✅ VAE 權重載入成功！")

# ------------------------------
# 影像重建
# ------------------------------
def reconstruct_image(image_input, vae_model):
    if isinstance(image_input, str):
        img = load_img(image_input, target_size=(128, 175), color_mode="grayscale")
        img = np.array(img, dtype=np.uint8)
        img =cv2.resize(img,(0,0),fx=2,fy=2)

        img2 = load_img(image_input, target_size=IMG_SIZE, color_mode="grayscale")
        img2 = np.array(img2, dtype=np.uint8)
        img2 = cv2.GaussianBlur(img2, (3, 3), 0)
        img2 = cv2.bilateralFilter(img2, 10, 15, 100)
        img2 = img_to_array(img2) / 255.0
        rec = vae_model(np.expand_dims(img2, 0)).numpy().squeeze()
        return img, img2.squeeze(), rec

def calculate_error(original, reconstructed):
    difference = np.abs(original - reconstructed)
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

def df_findContours(img, orig, recon):
    imgc = img.copy()
    imgc = img_to_array(imgc) / 255.0
    imgc = imgc.squeeze()
    edges = cv2.Canny(img, 50, 240)
    clo   = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
    dil   = cv2.dilate(clo, np.ones((5,5),np.uint8), iterations=2)
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape

    exclude = [
        (0, 0, int(w*0.02), int(h*0.02)),              
        (int(w*0.98), 0, int(w), int(h*0.02)),    
        (0, int(h*0.98), int(w*0.02), int(h)),   
        (int(w*0.7), int(h*0.9), int(w), int(h)), 
        (int(w*0.4), int(h*0.95), int(w*0.06), int(h))  
    ]
    def overlaps(r1, r2):
        x1,y1,w1,h1 = r1; x2,y2,w2,h2 = r2
        return not (x1+w1 < x2 or x1 > x2+w2 or y1+h1 < y2 or y1 > y2+h2)

    rois_o, rois_r, diffs, errs, ssims_ = [], [], [], [], []
    rois_o.append(orig); rois_r.append(recon)
    diff,error =calculate_error(orig, recon)
    errs.append(error)
    ssims_.append(calculate_ssim(orig, recon))
    diffs.append(diff)
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        box     = (x,y,w,h)
        if any(overlaps(box, ex) for ex in exclude):
            continue
        w = max(w, 7)
        h = max(h, 7)
        if x + w > imgc.shape[1]:
            w = imgc.shape[1] - x
        if y + h > imgc.shape[0]:
            h = imgc.shape[0] - y

        roi_o = imgc[y:y+h, x:x+w]
        roi_r = recon[y:y+h, x:x+w]
        rois_o.append(roi_o); rois_r.append(roi_r)
        d,e = calculate_error(roi_o, roi_r)
        errs.append(e)
        ssims_.append(calculate_ssim(roi_o, roi_r))
        diffs.append(d)
        cv2.rectangle(img,(x, y),(x + w, y + h),(255, 0, 0),1,lineType=cv2.LINE_AA)
    return rois_o, rois_r, diffs, errs, ssims_

def Reconstructed():
    img,original_img, reconstructed_img = reconstruct_image(test_image_path, vae)
    img,original_img, reconstructed_img2 = reconstruct_image(test_image_path, vae)
    difference1,error_score1,ssim_score1 = first_comparison(original_img, reconstructed_img, reconstructed_img2)
    if error_score1<=0.04  or ssim_score1>=0.83 :
        rois_o, rois_r, diffs, errs, ssims_ = df_findContours(img,original_img, reconstructed_img)
        display_results_matplotlib(rois_o, rois_r, diffs, errs, ssims_, out_image_path)
        ssim_score1 = np.mean(ssims_)
        error_score1 = np.mean(errs)
    else:
        show_images_1(original_img, reconstructed_img,difference1, out_image_path,ssim_score1,error_score1)    
    return ssim_score1,error_score1

# ------------------------------
# 解果顯示
# ------------------------------
def show_images_1(original, reconstructed,difference, output_path,ssim,error):
    
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
    plt.show()

def display_results_matplotlib(rois_o, rois_r, diffs, errs, ssims_, output_path):
    fig, axes = plt.subplots(len(rois_o), 3, figsize=(12, 2 * len(rois_o)))
    df = 0
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
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)  
    plt.savefig(output_path)
    plt.show()

# ------------------------------
# 啟用相機
# ------------------------------
if __name__ == "__main__":
    webcam = cv2.VideoCapture(0) #使用的相機索引值
    if not webcam.isOpened():
        print("can't open the camera!!!")
    
    cv2.namedWindow("video", 0) 
    cv2.resizeWindow("video",1920,1080 )

    webcam.set(3, 3840)  
    webcam.set(4, 2160)  

    CAPTURE_WIDTH = 780
    CAPTURE_HEIGHT = 1063
    CROP_X = 1655  # 擷取區域的左上角 X 座標
    CROP_Y = 827   # 擷取區域的左上角 Y 座標

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
            print(f"({CROP_X},{CROP_Y}) {CAPTURE_HEIGHT}x{CAPTURE_WIDTH}")
            save_path = os.path.join(SAVE_FOLDER, f"{SAVE_img_name}_{save_index}.jpg")  
            cv2.imwrite(save_path, cv2.rotate(crop_frame, cv2.ROTATE_90_CLOCKWISE))
            print(f"已擷取影像: {save_path}")   
            test_image_path = os.path.join("test", f"{SAVE_img_name}_{save_index}.jpg")
            out_image_path = os.path.join("test", f"{SAVE_img_name}_{save_index}_2.jpg")
            #save_index += 1
            ssim_end,error_end = Reconstructed()
        
        elif key == ord('q'):
            break
        
    webcam.release()
    cv2.destroyAllWindows()

