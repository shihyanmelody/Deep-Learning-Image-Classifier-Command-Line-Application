# Image-Classifier-Command-Line-Application

## This project contains two parts, if you are using this code, I suggest you to run in a gpu server:
### Part one: a Jupyter notebooks training an image classifier with VGG16 model to recognize different species of flowers.
### Part two: a command line application for the image classifier. Using this application, you can train a new network on a dataset and save the model as a checkpoint, then you can also used the trained network to predict the class for an inpout image. Detailed description of the app is as followed: 
#### Train a new network on a data set with train.py:
1. Basic usage: input 'python train.py data_directory' in the command line, Prints out training loss, validation loss, and validation accuracy as the network trains
2. Others commands: 
- Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
- Choose architecture: python train.py data_dir --arch "vgg13"
- Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
- Use GPU for training: python train.py data_dir --gpu
#### Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the name (in this example it's flowers with the cat_to_name.json, but feel free to adapt it to your own) and class probability.
1.Basic usage: python predict.py /path/to/image checkpoint

2.Other commands: 
- Return top K most likely classes: python predict.py input checkpoint --top_k 3
- Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
- Use GPU for inference: python predict.py input checkpoint --gpu

*this project is part of Udacity Data Scientist Degree and 
