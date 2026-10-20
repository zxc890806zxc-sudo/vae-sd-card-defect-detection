import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 設定影像大小與參數
IMG_SIZE = (256, 350)
LATENT_DIM = 160
DATA_PATH = "model-image"
EPOCHS = 500
BATCH_SIZE = 12
LEARNING_RATE = 0.0001

# ----------------- VAE 架構 ----------------- #
class VAE_Encoder(layers.Layer):
    def __init__(self, latent_dim):
        super().__init__()
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
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE_Decoder(layers.Layer):
    def __init__(self):
        super().__init__()
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
        super().__init__()
        self.encoder = VAE_Encoder(latent_dim)
        self.decoder = VAE_Decoder()

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = sampling(z_mean, z_log_var)
        return self.decoder(z)

def vae_loss(inputs, outputs, z_mean, z_log_var):
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return reconstruction_loss + 0.05 * kl_loss

# ----------------- 讀取訓練圖片 ----------------- #
def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = load_img(img_path, target_size=IMG_SIZE, color_mode="grayscale")
        img = np.array(img, dtype=np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img =cv2.bilateralFilter(img, 15, 25, 100)
        img = cv2.medianBlur(img, 1)#不是和
        img = img_to_array(img) / 255.0
        images.append(img)
    return np.array(images)

# ----------------- 開始訓練 4 組模型 ----------------- #
for district_id in range(4):
    print(f"\n🚀 開始訓練 District{district_id+1} 模型")

    district_folder = os.path.join(DATA_PATH, f"D{district_id+1}")
    X_train = load_images(district_folder)
    print(f"✅ 載入 {len(X_train)} 張訓練圖片")

    vae = VAE(LATENT_DIM)
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    for epoch in range(EPOCHS):
        for i in range(0, len(X_train), BATCH_SIZE):
            x_batch = X_train[i:i+BATCH_SIZE]
            with tf.GradientTape() as tape:
                outputs = vae(x_batch)
                z_mean, z_log_var = vae.encoder(x_batch)
                loss = vae_loss(x_batch, outputs, z_mean, z_log_var)
            gradients = tape.gradient(loss, vae.trainable_variables)
            optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        print(f"District{district_id+1} | Epoch {epoch+1} / {EPOCHS} | Loss: {loss.numpy():.4f}")

    weight_path = f"vae_D{district_id+1}.weights.h5"
    vae.save_weights(weight_path)
    print(f"✅ 模型 District{district_id+1} 已訓練完成，權重儲存為 {weight_path}")

