# DCGAN 
This project is an implementation of famous DCGAN algorithm. I have created a package named dc gan which contains the actual implementation of the algorithm.  

## Data 
This folder contains the data for training the generator and discriminator.If you want to train the generator and discriminator on your dataset there is only one step to do this just replace the images in data/train folder with your images in data

## weights 
weights folder contains the saved weights for the generator and discriminator

# Model
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


