
# Imports here
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image
import numpy as np
import os.path
import json

def train_load(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([
    transforms.RandomRotation(38),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dev_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms) 
    dev_data = datasets.ImageFolder(valid_dir, transform = dev_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    devloaders = torch.utils.data.DataLoader(dev_data, batch_size = 32)
    testloaders = torch.utils.data.DataLoader(test_data, batch_size = 32)
    dataloader = {}
    dataloader['train_data'] = train_data
    dataloader['dev_data'] = dev_data
    dataloader['test_data'] = test_data
    dataloader['trainloaders'] = trainloaders
    dataloader['devloaders'] = devloaders
    dataloader['testloaders'] = testloaders
    print('successfully load data')
    return dataloader

def classifier_build(model_type = 'vgg16', hidden_units = 2048):   # model_type can be vgg16 or vgg13
    if model_type == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif model_type =='vgg13':
        model = models.vgg13(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    print('successfully build up the classifier as followed:')
    print(model)
    return model

def validation(model, testloader, criterion, device = "cpu"):
    test_loss = 0
    accuracy = 0
    batch_count = 0
    print('validation function is called')
    if device == "gpu":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for images, labels in testloader:
        batch_count += 1
        images, labels = images.to(device), labels.to(device)
        model.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim = 1)[1])
#         print('ps is:', ps, 'ps.max(dim=1) is', ps.max(dim = 1), 'labels.data is', labels.data)
        accuracy += equality.type(torch.FloatTensor).mean()
#         print('equality is', equality, 'accuracy is', accuracy)
    accuracy = accuracy/batch_count        
    return test_loss, accuracy

def training(model, trainloaders, devloaders, learning_rate = 0.001, epochs = 3, device = "cpu"):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    print_every = 50
    steps = 0
    if device == "gpu":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('start training using', device)    
    for e in range(epochs):
        running_loss = 0
        for inputs, labels in iter(trainloaders):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            model.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                test_loss, accuracy = validation(model, devloaders, criterion)
                print("Epoch: {}/{}... ".format(steps, e+1),
                      "Loss: {:.4f}".format(running_loss/print_every), 
                     "test_loss: {:.4f}".format(test_loss),
                      "accuracy:{:.4f}".format(accuracy))
                running_loss = 0
                model.train()
    print('training finish')                
    return model

def save_model(model, save_dir, train_data, hidden_units = 2048):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'classifier':nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(25088, hidden_units)),
                    ('relu', nn.ReLU()),
                    ('fc2', nn.Linear(hidden_units, 102)),
                    ('output', nn.LogSoftmax(dim=1))
                ])),
                  'state_dict': model.state_dict(),
                  'class_to_index':model.class_to_idx
                 }
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))
    print('the model is saved')

def load_checkpoint(filepath, model_type ="vgg16"):
    checkpoint = torch.load(filepath)
    if model_type == "vgg16":
        model = models.vgg16(pretrained = True)
    elif model_type == "vgg13":
        model = models.vgg13(pretrained = True)        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict']) 
    print('successfully load the model')
    return model
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    max_min_ratio = max(im.size)/min(im.size)
#     print(max_min_ratio)
    size = 256*max_min_ratio, 256*max_min_ratio
    im.thumbnail(size)
#     print(im.size)
    center_position = int(im.size[0]/2-112), int(im.size[1]/2-112), int(im.size[0]/2+112), int(im.size[1]/2+112)
#     print(center_position)
#     im = im.resize((224,224))
    im = im.crop(center_position)
#     print(im.size)
    np_image = np.array(im)
    np_image = np_image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image-mean)/std
    np_image = np_image.transpose((2,0,1))
    return np_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model_path, topk=5, device = "cpu"):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    checkpoint = torch.load(model_path)
    model = load_checkpoint(model_path)
    index_to_class = {v: k for k, v in checkpoint['class_to_index'].items()}
    if device == "gpu":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    im = process_image(image_path)
#     print('step1')
    im_tensor = torch.from_numpy(im).to(device).unsqueeze_(0)
#     print(im_tensor)
    output = model.forward(im_tensor.float())
    ps = torch.exp(output)
    probs = list(ps.topk(topk)[0].cpu().detach().numpy()[0])
    indices = ps.topk(topk)[1].cpu().detach().numpy()[0]
    classes = []
    names= []
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
#     print(probs)
        for i in range(topk):
            idx = indices[i]
            class_ = index_to_class[idx]
    #         print(index_to_class[idx])
            classes.append(class_)
            names.append(cat_to_name[class_])
#         print(classes)
#     print('')
    return probs, classes, names



