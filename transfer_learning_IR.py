#!/usr/bin/env python
# coding: utf-8

# In[121]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import sys
import argparse
import copy
import os
# install tabulate
import tabulate

plt.style.use('seaborn-bright')
#for reproducible training.
torch.manual_seed(123)
np.random.seed(33)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


# Transformation for the image, both for training and testing
img_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# get the number of epochs.
parser = argparse.ArgumentParser()

parser.add_argument("-e", "--Epochs", help = "Set number of epochs")

args = parser.parse_args()

if args.Epochs:
    Epochs = int(args.Epochs)
    print("Number of epochs : ",Epochs)
else:
    sys.exit("Provide the number of epochs !! (-e=...)")

# In[122]:


# Food Dataset
image_ds = datasets.ImageFolder("FoodDS",img_transform)
image_ds1 = datasets.ImageFolder("FoodDS1",img_transform)
targets = image_ds.targets


# In[123]:

dataset_sizes = {}
dataset_sizes['train'] = len(image_ds)
dataset_sizes['val'] = len(image_ds1)


# In[124]:

dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(image_ds, batch_size=128, num_workers=8,pin_memory=True,shuffle=True)
dataloaders['val'] = torch.utils.data.DataLoader(image_ds1, batch_size=128,num_workers=8,pin_memory=True,shuffle=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class_names = image_ds.classes

# In[125]:


def reset_final_layer(model,class_names,device):
    """Resets final fully connected layer of the model"""
    num_ftrs = model.fc.in_features
    # Output size -> 3 (ants,bees,wasps)
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)



# In[127]:


def imshow(inp, label,count):
    """Image from Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imsave("Misclassified_IR/"+ label + "/" +str(count)+".jpg",inp)

def save_plots(train_loss,test_loss,train_acc,val_acc,num_epochs,model_name):
    """Save the loss and accuracy plots"""
    # Loss plots
    plt.plot(np.arange(num_epochs),train_loss,'-b',label="training")
    plt.plot(np.arange(num_epochs),test_loss,'-r',label="validation")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('IR_' + model_name+"_loss.png",dpi=300,bbox_inches='tight')
    plt.clf()
    
    # Accuracy plot
    plt.plot(np.arange(num_epochs),train_acc,'-b',label="training")
    plt.plot(np.arange(num_epochs),val_acc,'-r',label="validation")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('IR_' + model_name+"_acc.png",dpi=300,bbox_inches='tight')
    plt.clf()


def train_model(model, criterion, optimizer, scheduler,max_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_losses = [];train_acc = []
    val_losses = [];val_acc = []

    for epoch in range(max_epochs):
        print('Epoch {}/{}'.format(epoch+1, max_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward pass
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
            	scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if(phase=="train"):
                train_losses.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_acc.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Accuarcy: {:4f}'.format(best_acc))
    
    # Save the loss plots
    save_plots(train_losses,val_losses,train_acc,val_acc,max_epochs,model.__class__.__name__)
    print("Loss plots saved !!")
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# ## Finetuning the CNN

# ### Below is a generic code compatible with resnet, googlenet, densenet & VGG

# In[128]:


# To use any other model, change the name as e.g. for resnet18 : models.resnet18(pretrained=True)

model = models.resnet152(pretrained = True)

# transfer learning -> fixed feature extractor
# for param in model.parameters():
# 	param.requires_grad = False

for name, child in model.named_children():
	if name in ['layer3','layer4']:
		for param in child.parameters():
			param.requires_grad = True
	else:
		for param in child.parameters():
			param.requires_grad = False

reset_final_layer(model,class_names,device)

criterion = nn.CrossEntropyLoss()


optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.01, momentum = 0.9)

scheduler = lr_scheduler.CyclicLR(optimizer,mode='exp_range',base_lr=0.00007,max_lr = 0.01,step_size_up = 20,gamma = 0.991)


# Runs for provided epochs
best_model = train_model(model, criterion, optimizer,scheduler,max_epochs=Epochs)
torch.save(best_model,"IR_" + "model_"+ model.__class__.__name__ + ".pt")
print("best model saved!!")
print()


def visualize_misclassified(model, num_images=6):
    # was_training = model.training
    plt.clf()
    model.eval()
    images_so_far = 0
    miss = {}
    for i in class_names:
        miss[i] = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                if labels[j].item() != preds[j].item():
                    p = class_names[preds[j].item()]
                    miss[p] += 1
                    images_so_far += 1
                    imshow(inputs.cpu().data[j],p,miss[p])

                    if images_so_far == num_images:
                        return