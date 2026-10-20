import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
#conda activate gpu
# 設定影像大小
IMG_SIZE = (256, 350)
DATA_PATH = "model-image" 

# 訓練模型參數
EPOCHS = 500  
BATCH_SIZE = 12
LATENT_DIM = 160 

def load_images():
    images = []
    folder_path = os.path.join(DATA_PATH)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = load_img(img_path, target_size=IMG_SIZE, color_mode="grayscale")
        img = np.array(img, dtype=np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.bilateralFilter(img, 10, 15, 100)
        img = img_to_array(img) / 255.0  
        images.append(img)
    return np.array(images)

X_train = load_images()
print(f"載入 {len(X_train)} 張 defect-free 影像")

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
        # 這裡調整 dense 層輸出尺寸為 64*88*64，對應下採樣後的尺寸 (64,88) 
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
        reconstructed = self.decoder(z)
        return reconstructed

vae = VAE(LATENT_DIM)

def vae_loss(inputs, outputs, z_mean, z_log_var):
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss = tf.reduce_mean(reconstruction_loss)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return reconstruction_loss + 0.05 * kl_loss  # 調整 KL loss 權重

optimizer = keras.optimizers.Adam(learning_rate=0.0001)

for epoch in range(EPOCHS):
    for i in range(0, len(X_train), BATCH_SIZE):
        x_batch = X_train[i:i+BATCH_SIZE]
        with tf.GradientTape() as tape:
            outputs = vae(x_batch)
            z_mean, z_log_var = vae.encoder(x_batch)
            loss = vae_loss(x_batch, outputs, z_mean, z_log_var)
        gradients = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")

# **儲存 VAE 權重**
vae.save_weights("vae_model.weights.h5")
print("✅ VAE 訓練完成，權重已儲存！")

