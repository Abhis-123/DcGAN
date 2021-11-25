# DCGAN 
This is my [Tensorflow](https://www.tensorflow.org/) implementation of **Deep Convolutional Generative Adversarial Networks in Tensorflow** proposed in the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434). The main contribution comes from the tricks which stablize the training of Generative Adversarial Networks. The proposed architecture is as followed.
<img src="figure/dcgan.png" height="300"/>

This project is an implementation of famous DCGAN algorithm. I have created a package named dc gan which contains the actual implementation of the algorithm.  
# Prerequisites

- Python 2.7 or Python 3.3+
- [Tensorflow 1.0.0](https://github.com/tensorflow/tensorflow/tree/r1.0)
- [SciPy](http://www.scipy.org/install.html)
- [NumPy](http://www.numpy.org/)

# Project Structure 
## Data 
'data' folder contains the data for training the generator and discriminator.If you want to train the generator and discriminator on your dataset there is only one step to do this just replace the images in data/train folder with your images in data

## weights 
weights folder contains the saved weights for the generator and discriminator

## Model
model class contains the code for DCGAN Model 

functions : 
    
    summary(): This function prints the the layer wise information about generator and discriminator
    
    train(): This function train the generator and discriminator 
        parameters : 
            dataset : a tensor of shape (None,None, 64,64,3)  which means dataset should be supplied in forms of batches
            epochs  : number of epochs to train the generator and discriminator
    
    save: to save the weights of the generator and discriminator after training the model
        parameters: 
            path : base path to save the folder of weights of the generator and discriminator
            name : name for weights folder 

    load_weights: to load the weights of the saved generator and discriminator
            parameters :
                path : path to load the weights of the generator and discriminator , this path should target the folder containing weights

    generate_images(): This function generate multiple images and save them to directory
        parameters : 
            count : the number of image to be generated 
            save_dir : the directory to save the generated images

    generate_image() :To generate a image 
        return :
            a 3D images tensor


