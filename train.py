import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from dcgan.model import Model
from tqdm import tqdm
from PIL import Image
import numpy as np
import tensorflow as tf
import os
TRAIN_IMAGE_DIR="./data/train"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
if __name__ == "__main__":
    train_path = TRAIN_IMAGE_DIR
    files = os.listdir(train_path)
    dataset = []
    for file in tqdm(files):
        try:
            # print(file)
            img = Image.open(train_path + '/' + file)
            img = img.convert('RGB')
            img = img.resize((64, 64))
            img = np.asarray(img) / 255
            dataset.append(img)
        except:
            print("something went wrong")

    dataset = np.array(dataset)
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.batch(32)
    model = Model()
    model.train(dataset, epochs=10)
    model.save("./weights")
    model.load_weights("./weights")
    model.generate_images(save_dir="./data")
