import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# =========================
# 全域設定
# =========================
IMG_SIZE = (256, 350)      # 輸入影像尺寸 (h, w)；訓練時統一縮放
LATENT_DIM = 160           # VAE 潛在空間維度 (壓縮後特徵向量長度)
DATA_PATH = "model-image"  # 訓練資料根目錄，下含 D1~D4 子資料夾
EPOCHS = 500               # 每個分區(District)的訓練週期數
BATCH_SIZE = 12            # batch 大小
LEARNING_RATE = 0.0001     # Adam 學習率


# =========================
# VAE 結構：Encoder
# 將影像抽取特徵並壓縮成 (z_mean, z_log_var)
# =========================
class VAE_Encoder(layers.Layer):
    def __init__(self, latent_dim):
        super().__init__()
        # 基礎卷積堆疊 + 下採樣，逐步提取更高層次特徵
        self.conv1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")
        self.pool1 = layers.MaxPooling2D((2, 2), padding="same")
        self.conv2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")
        self.pool2 = layers.MaxPooling2D((2, 2), padding="same")
        self.flatten = layers.Flatten()
        # 生成潛在空間的平均與對數方差
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


# =========================
# Reparameterization trick
# z = μ + σ * ε (ε ~ N(0, I))，讓隨機抽樣可反傳梯度
# =========================
def sampling(z_mean, z_log_var):
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# =========================
# VAE 結構：Decoder
# 將 latent z 還原成影像大小
# =========================
class VAE_Decoder(layers.Layer):
    def __init__(self):
        super().__init__()
        # 先將 z 展開成 feature map，之後透過 ConvTranspose + Upsample 放大
        self.dense = layers.Dense(64 * 88 * 64, activation="relu")
        self.reshape = layers.Reshape((64, 88, 64))
        self.deconv1 = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same")
        self.up1 = layers.UpSampling2D((2, 2))
        self.deconv2 = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same")
        self.up2 = layers.UpSampling2D((2, 2))
        # 因轉置卷積/上採樣尺寸偶爾 off-by-one，裁掉 1px 對齊輸出
        self.crop = layers.Cropping2D(((0, 0), (1, 1)))
        # 單通道輸出 (灰階)，Sigmoid 將像素壓在 [0,1]
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


# =========================
# 高層 VAE：Encoder + sampling + Decoder
# =========================
class VAE(keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = VAE_Encoder(latent_dim)
        self.decoder = VAE_Decoder()

    def call(self, inputs):
        # 前向傳遞回傳「重建影像」，便於外層直接拿 outputs 做 reconstruction loss
        z_mean, z_log_var = self.encoder(inputs)
        z = sampling(z_mean, z_log_var)
        return self.decoder(z)


# =========================
# 損失函數：重建誤差 + KL 散度
# 注意：
# - binary_crossentropy 預期 inputs/outputs 為 [0,1]
# - KL 權重 (0.05) 控制壓縮程度與生成品質的折衷
# =========================
def vae_loss(inputs, outputs, z_mean, z_log_var):
    # (N, H, W, 1) → element-wise BCE，再平均
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    # KL = -0.5 * Σ (1 + logσ² - μ² - σ²)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return reconstruction_loss + 0.05 * kl_loss


# =========================
# 載入單一分區資料夾內的訓練圖片
# 流程：讀檔→縮放→(高斯/雙邊/中值)濾波→正規化→堆叠
# 備註：
# - medianBlur(kernel=1) 等於不太會改變影像 (核須為奇數>=1)，保留你的原設定不變
# - 若要提高穩定性，可考慮固定讀檔排序與隨機種子
# =========================
def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        # 以 Keras 讀檔並縮放；灰階
        img = load_img(img_path, target_size=IMG_SIZE, color_mode="grayscale")
        img = np.array(img, dtype=np.uint8)

        # 輕量前處理：抑制雜訊但盡量保留邊緣
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.bilateralFilter(img, 15, 25, 100)
        img = cv2.medianBlur(img, 1)   # kernel=1 幾乎無影響；保留你的設定

        # 正規化到 [0,1] 並轉為 (H, W, 1) float32
        img = img_to_array(img) / 255.0
        images.append(img)
    return np.array(images)


# =========================
# 逐一訓練 4 組模型 (D1~D4)
# 每一組對應一個資料分布/視角/區域，彼此獨立訓練與存權重
# =========================
for district_id in range(4):
    print(f"\n🚀 開始訓練 District{district_id+1} 模型")

    # 讀取對應分區的資料
    district_folder = os.path.join(DATA_PATH, f"D{district_id+1}")
    X_train = load_images(district_folder)
    print(f"✅ 載入 {len(X_train)} 張訓練圖片")

    # 建立 VAE 與最佳化器
    vae = VAE(LATENT_DIM)
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # ===== 手刻訓練迴圈 =====
    # 逐 epoch → 逐 batch 前向計算 → 損失 → 反傳 → 更新
    for epoch in range(EPOCHS):
        # 簡單切片 batching；如需 shuffle 可在這裡先 np.random.permutation
        for i in range(0, len(X_train), BATCH_SIZE):
            x_batch = X_train[i:i+BATCH_SIZE]
            with tf.GradientTape() as tape:
                outputs = vae(x_batch)                 # 直接呼叫 VAE：取得重建影像
                z_mean, z_log_var = vae.encoder(x_batch)  # 再取 encoder 的 μ/σ 用於 KL
                loss = vae_loss(x_batch, outputs, z_mean, z_log_var)
            gradients = tape.gradient(loss, vae.trainable_variables)
            optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        print(f"District{district_id+1} | Epoch {epoch+1} / {EPOCHS} | Loss: {loss.numpy():.4f}")

    # 儲存該分區的權重檔，供推論腳本載入使用
    weight_path = f"vae_D{district_id+1}.weights.h5"
    vae.save_weights(weight_path)
    print(f"✅ 模型 District{district_id+1} 已訓練完成，權重儲存為 {weight_path}")

