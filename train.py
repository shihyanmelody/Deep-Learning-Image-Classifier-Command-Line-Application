import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image
import numpy as np
import os.path
from si_cnn import train_load, classifier_build, validation, training, save_model, load_checkpoint, process_image, imshow, predict
import argparse
from workspace_utils import active_session

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', action='store', type = str,
                        help='data directory is required')
    parser.add_argument('--save_dir', action='store', dest='save_directory', default='',
                        help='model save directory')
    parser.add_argument('--arch', action='store', dest='architecture', default='vgg16',
                        help='model architecture, can be either vgg16 or vgg13, default is vgg16')
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', default=0.001,
                        type=float, help='model learning rate, default is 0.001')
    parser.add_argument('--hidden_units', action='store', dest='hidden_units', default=2048,
                        type=int, help='hidden units, default is 2048')
    parser.add_argument('--epochs', action='store', dest='epochs', default='3',
                        type=int, help='training epochs, default is 3')
    parser.add_argument('--gpu', action='store_true', default = False, dest='device',
                        help='machine to train the model, defaul is cpu')    
    results = parser.parse_args()    
    print('parse result is ', results)
#     data_dir = 'flowers'
    data_dir = results.data_directory
    save_dir = results.save_directory
    architecture = results.architecture
    learning_rate_input = results.learning_rate
    hidden_units_input = results.hidden_units
    epochs_input = results.epochs
    device = 'cpu'
    if results.device:
        device = 'gpu'
    print('data_dir is ', data_dir, 'save_dir is', save_dir, 'architecture is ', architecture, 'learning_rate is ', learning_rate_input, 'hidden_units is ', hidden_units_input, 'epochs is ', epochs_input, 'device is ', device)
    dataloader = {}
    dataloader = train_load(data_dir)
    try:
        model = load_checkpoint(save_dir+'checkpoint.pth')
    except:
        model = classifier_build(model_type = architecture, hidden_units = hidden_units_input)
#     with active_session():
#         train_model = training(model, dataloader['trainloaders'], dataloader['devloaders'], learning_rate = learning_rate_input, epochs = epochs_input,  device = "gpu")
#         save_model(train_model, save_dir, dataloader['train_data'], hidden_units = hidden_units_input)
    print('------Test the model with test dataset----------')
    test_criterion = nn.NLLLoss()
    test_loss, accuracy = validation(model, dataloader['testloaders'], test_criterion)
    print("test_loss: {:.4f}".format(test_loss),
          "accuracy:{:.4f}".format(accuracy))    
    
if __name__ == "__main__":
    main()