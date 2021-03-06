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
    for file in files:
        try:
            # print(file)
            img = Image.open(train_path + '/' + file)
            img = img.convert('RGB')
            img = img.resize((64, 64))
            img = np.asarray(img) / 255
            dataset.append(img)
        except:
            print("something went wrong")
    print(f"loaded all images ...")
    dataset = np.array(dataset)
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.batch(32)
    model = Model()
    #loss=model.train(dataset, epochs=1)
    image = model.generate_image()
    # print(f"loss : {loss}")
    #model.save("./weights",folder_name="model2")
    model.load_weights("./weights/model2")
    # model.generate_images(save_dir="./data")

    
