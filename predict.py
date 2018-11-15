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
    parser.add_argument('test_image_path', action='store', type = str,
                        help='test image path is required')
    parser.add_argument('checkpoint_path', action='store', type = str,
                        help='checkpoint path is required')    
    parser.add_argument('--top_k', action='store', dest='top_k', default=5,
                        type=int, help='top_k, default is 5')
    parser.add_argument('--category_names', action='store', dest='category_names', default='cat_to_name.json',
                        type=str, help='category_names, default is cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', default = False, dest='device',
                        help='machine to train the model, defaul is cpu')    
    results = parser.parse_args()
    image_path = results.test_image_path
    checkpoint_path = results.checkpoint_path
    top_k = results.top_k
    category_names = results.category_names
    device_input = 'cpu'
    if results.device:
        device_input = 'gpu'    
    probs, classes, names = predict(image_path, checkpoint_path, topk=top_k, device = device_input)
#     print(names[0])
    print('top ', top_k, 'results are:')
    print('------------------------------')
    for i in range(top_k):
          print('name:', names[i],'   probability:', probs[i], '   class', classes[i])
    print('most probable flower:', names[0])
#     processed_image = process_image(image_path)
#     out_image = imshow(processed_image)
#     plt.figure()
#     # plt.subplot(1,2,1)
#     plt.barh(names, probs)
#     plt.show()

if __name__ == "__main__":
    main()