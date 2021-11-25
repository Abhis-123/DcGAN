# DCGAN 
This is my [Tensorflow](https://www.tensorflow.org/) implementation of **Deep Convolutional Generative Adversarial Networks in Tensorflow** proposed in the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434). The main contribution comes from the tricks which stablize the training of Generative Adversarial Networks. The proposed architecture is as followed.
<img src="pictures\dcgan.png" height="300"/>

This project is an implementation of famous DCGAN algorithm. I have created a package named dc gan which contains the actual implementation of the algorithm.  
# Prerequisites

- Python 3.3+
- [Tensorflow 2.3.1](https://github.com/tensorflow/tensorflow/tree/r1.0)
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


# Results During Training
## MNIST

* Generated samples (200th epochs)

<img src="pictures\results\mnist/samples.png" height="250"/>

* First 50 epochs

<img src="pictures\results\mnist/training.gif" height="250"/>

## CFAR

* Generated samples (500th epochs)

<img src="pictures\results\cfar10/samples.png" height="250"/>

* First 40 epochs

<img src="pictures\results\cfar10/training.gif" height="250"/>

## Training tricks

* To avoid the fast convergence of the discriminator network
    * The generator network is updated more frequently.
    * Higher learning rate is applied to the training of the generator.
* One-sided label smoothing is applied to the positive labels.
* Gradient clipping trick is applied to stablize training


## Related works
* [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) by Radford
* [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)

* Training Tips i borrowed from [shaohua0116 training](https://github.com/shaohua0116/DCGAN-Tensorflow/)

