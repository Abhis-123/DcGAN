import os
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers as L
from tqdm import tqdm
import numpy as np
from PIL import Image


class Model:
    def __init__(self, latent_dimension=128):
        self.discriminator = K.Sequential(
            [
                K.Input(shape=(64, 64, 3), name="discriminator_layer_input"),
                L.Conv2D(64, kernel_size=4, strides=2, padding="same", name="discriminator_layer_conv1"),
                L.LeakyReLU(alpha=0.2),
                L.Conv2D(128, kernel_size=4, strides=2, padding="same", name="discriminator_layer_conv2"),
                L.LeakyReLU(0.2),
                L.Conv2D(128, kernel_size=4, strides=2, padding="same", name="discriminator_layer_conv3"),
                L.LeakyReLU(0.2),
                L.Flatten(),
                L.Dense(128, activation="sigmoid", name="discriminator_layer_"),
                L.Dropout(0.2),
                L.Dense(1, activation="sigmoid"),
            ],
            name="discriminator",
        )
        self.generator = K.Sequential([
            K.Input(shape=(latent_dimension,)),
            L.Dense(8 * 8 * latent_dimension),
            L.Reshape((8, 8, latent_dimension)),
            L.Conv2DTranspose(latent_dimension, kernel_size=4, strides=2, padding="same"),
            L.LeakyReLU(0.2),
            L.Conv2DTranspose(latent_dimension * 2, kernel_size=4, strides=2, padding="same"),
            L.LeakyReLU(0.2),
            L.Conv2DTranspose(latent_dimension * 3, kernel_size=4, strides=2, padding="same"),
            L.LeakyReLU(0.2),
            L.Conv2DTranspose(3, kernel_size=5, padding="same"),

        ])
        self.latent_dimension = latent_dimension
        self.generator_optimiser = K.optimizers.Adam(1e-4)
        self.discriminator_optimiser = K.optimizers.Adam(1e-4)
        self.loss = K.losses.BinaryCrossentropy()

    def summary(self):
        print("generator layers summary ..........")
        print(self.generator.summary())
        print('discriminator summary ..............')
        print(self.discriminator.summary())

    def train(self, dataset, epochs=2):
        d_loss=[]
        g_loss=[]
        for epoch in range(epochs):
            gen_loss = []
            disc_loss = []
            for idx, (real) in enumerate(tqdm(dataset)):
                batch_size = real.shape[0]
                random_latent_vector = tf.random.normal(shape=(batch_size, self.latent_dimension))
                fake = self.generator(random_latent_vector)
                with tf.GradientTape() as disc_tape:
                    d_loss_real = self.loss(tf.ones((batch_size, 1)), self.discriminator(real))
                    d_loss_fake = self.loss(tf.zeros((batch_size, 1)), self.discriminator(fake))
                    loss = d_loss_real + d_loss_fake
                    disc_loss.append(loss.numpy())

                grads = disc_tape.gradient(loss, self.discriminator.trainable_weights)
                var = self.discriminator_optimiser
                var.apply_gradients(
                    zip(grads, self.discriminator.trainable_weights)
                )

                with tf.GradientTape() as gen_tape:
                    fake = self.generator(random_latent_vector)
                    output = self.discriminator(fake)
                    loss_gen = self.loss(tf.ones((batch_size, 1)), output)
                    gen_loss.append(loss_gen.numpy())
                grads_gen = gen_tape.gradient(loss_gen, self.generator.trainable_weights)
                self.generator_optimiser.apply_gradients(zip(grads_gen, self.generator.trainable_weights))
            epoch_gen_loss=sum(gen_loss)
            epoch_disc_loss=sum(disc_loss)
            d_loss.append(epoch_disc_loss)
            g_loss.append(epoch_gen_loss)
            print(f"Epoch {epoch} generator loss {epoch_gen_loss} discriminator loss {epoch_disc_loss}")
        
        loss={
            "discriminator_loss":d_loss,
            "generator_loss":g_loss,
        }
        return loss

    def save(self, path, folder_name="model"):
        if not os.path.exists(path + "/"+str(folder_name)):
            os.mkdir(path + "/"+str(folder_name))
        self.generator.save(f"{path}/{folder_name}/generator.h5")
        self.discriminator.save(f"{path}/{folder_name}/discriminator.h5")

    def load_weights(self, path):
        if not os.path.exists(path):
            print("weights folder not found..")
            return False
        self.generator.load_weights(f"{path}/generator.h5")
        self.discriminator.load_weights(f"{path}/discriminator.h5")
        return True

    def generate_images(self, count=1, save_dir="."):
        if not os.path.exists(save_dir+"/generated"):
            os.mkdir(save_dir + '/generated')
        save_dir = save_dir + "/generated"
        for i in range(count):
            random_latent_vector = tf.random.normal(shape=(1, self.latent_dimension))
            fake = self.generator(random_latent_vector)

            img = tf.keras.preprocessing.image.array_to_img(fake[0])
            name = f"generated_image_{i}.png"
            img.save(save_dir + "/" + name)

    def generate_image(self):
        random_latent_vector = tf.random.normal(shape=(1, self.latent_dimension))
        fake = self.generator(random_latent_vector)
        return fake[0]*255


if __name__ == "__main__":
    files = os.listdir("../data/train")
    dataset = []
    for file in tqdm(files):
        try:
            # print(file)
            img = Image.open("../data/train" + '/' + file)
            img = img.convert('RGB')
            img = img.resize((64, 64))
            img = np.asarray(img) / 255
            dataset.append(img)
        except:
            print("something went wrong")

    dataset = np.array(dataset)

    print(dataset.shape)
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.batch(32)
    model = Model()
    model.summary()
    model.train(dataset, epochs=20)
