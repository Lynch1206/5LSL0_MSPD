# %% imports
# pytorch
import torch
from torchvision import transforms,datasets
from torch.utils.data import Dataset,DataLoader
import os
# pyplot
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import LinearAxis, Range1d
import warnings

warnings.filterwarnings("ignore")
print('Warnings are turned off')

# %% Noisy MNIST dataset
class Noisy_MNIST(Dataset):
    # initialization of the dataset
    def __init__(self, split,data_loc,noise=0.5):
        # save the input parameters
        self.split    = split 
        self.data_loc = data_loc
        self.noise    = noise
        
        if self.split == 'train':
            Train = True
        else:
            Train = False
            
        # get the original MNIST dataset   
        Clean_MNIST = datasets.MNIST(self.data_loc, train=Train, download=True)
        
        # reshuffle the test set to have digits 0-9 at the start
        if self.split == 'train':
            data = Clean_MNIST.data.unsqueeze(1)
        else:
            data = Clean_MNIST.data.unsqueeze(1)
            idx = torch.load('test_idx.tar')
            data[:,:] = data[idx,:]
            
        
        # reshape and normalize
        resizer = transforms.Resize(32)
        resized_data = resizer(data)*1.0
        normalized_data = 2 *(resized_data/255) - 1
        
        # create the data
        self.Clean_Images = normalized_data
        self.Noisy_Images = normalized_data + torch.randn(normalized_data.size())*self.noise
        self.Labels       = Clean_MNIST.targets
    
    # return the number of examples in this dataset
    def __len__(self):
        return self.Labels.size(0)
    
    # create a a method that retrieves a single item form the dataset
    def __getitem__(self, idx):
        clean_image = self.Clean_Images[idx,:,:,:]
        noisy_image = self.Noisy_Images[idx,:,:,:]
        label =  self.Labels[idx]
        
        return clean_image,noisy_image,label
    
# %% dataloader for the Noisy MNIST dataset
def create_dataloaders(data_loc, batch_size):
    Noisy_MNIST_train = Noisy_MNIST("train", data_loc)
    Noisy_MNIST_test  = Noisy_MNIST("test" , data_loc)
    
    Noisy_MNIST_train_loader =  DataLoader(Noisy_MNIST_train, batch_size=batch_size, shuffle=True,  drop_last=False)
    Noisy_MNIST_test_loader  =  DataLoader(Noisy_MNIST_test , batch_size=batch_size, shuffle=False, drop_last=False)
    
    return Noisy_MNIST_train_loader, Noisy_MNIST_test_loader

# %% test if the dataloaders work
fileDirectory = os.path.dirname(os.path.abspath(__file__))            # curent file directory 
MODEL__DIR = os.path.join(fileDirectory , "trained_model_")          # Directory for storing model
if __name__ == "__main__":
    # define parameters
    data_loc = fileDirectory #change the datalocation to something that works for you
    batch_size = 64
    
    # get dataloader
    train_loader, test_loader = create_dataloaders(data_loc, batch_size)
    
    # get some examples
    examples = enumerate(test_loader)
    _, (x_clean_example, x_noisy_example, labels_example) = next(examples)
    # use these example images througout the assignment as the first 10 correspond to the digits 0-9
    # show the examples in a plot
    plt.figure(figsize=(12,3))
    for i in range(10):
        plt.subplot(2,10,i+1)
        plt.imshow(x_clean_example[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(2,10,i+11)
        plt.imshow(x_noisy_example[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.savefig("data_examples.png",dpi=300,bbox_inches='tight')
    plt.show()

# Question b
# Network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32, 32*32) 
        self.fc2 = nn.Linear(32*32, 32*32)
        self.fc3 = nn.Linear(32*32, 32*32)
    def forward(self, x):
        x = x.view(-1, 32*32)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
# Question c
model = Net()
model = model.cuda() # Transfer to GPU
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Question d
#  training-validation sets split
train_set, val_set = torch.utils.data.random_split(train_loader.dataset, [50000, 10000])

# Predict test set with untrained network
# Test the model
model.eval()
with torch.no_grad():
    plt.figure(figsize=(12,3))              # show the examples in a plot
    x_noisy_example = x_noisy_example.cuda() # Transfer to GPU
    output = model(x_noisy_example).resize(64,1,32,32).cpu() # resize our output as the same size of input and transfer back to cpu
    for i in range(10):
        plt.subplot(2,10,i+1)
        plt.imshow(x_clean_example[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2,10,i+11)
        plt.imshow(output[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.savefig("untrained_test_data_examples.png",dpi=300,bbox_inches='tight')
    plt.show()

# Question e  
# Train the model
loss_list_train = []
loss_list_validation = []
num_epochs = 50
for epoch in range(num_epochs):
    for (x_clean_train, x_noisy_train, labels_train),(x_clean_val, x_noisy_val, labels_val) in zip(train_set,val_set):
        # Transfer to GPU
        x_clean_train = x_clean_train.cuda()
        x_noisy_train = x_noisy_train.cuda()
        labels_train = labels_train.cuda()
        x_clean_val = x_clean_val.cuda()
        x_noisy_val = x_noisy_val.cuda()
        labels_val = labels_val.cuda()
        # Run the forward pass
        outputs_train = model(x_noisy_train)
        outputs_validation = model(x_noisy_val)
        loss_train = criterion(outputs_train, x_clean_train.resize(1024))
        loss_validation = criterion(outputs_validation, x_clean_val.resize(1024))
        # loss_train = criterion(outputs_train, x_clean_train.resize(labels_train.size(0),1024))
        # loss_validation = criterion(outputs_validation, x_clean_val.resize(labels_val.size(0),1024))
        loss_list_train.append(loss_train.item())
        loss_list_validation.append(loss_validation.item())

        # Backprop and perform SGD optimisation on training set
        optimizer.zero_grad()
        loss_train.backward()
        # Perform the SGD optimizer
        optimizer.step()
    # Save the model 
    torch.save(model, MODEL__DIR + 'epoch'+ str(epoch+1)+ 'FNN_net_model.ckpt')

# Plot the loss 
p = figure(y_axis_label='Train Loss ', width=850, y_range=(0, 1), title='Train and validation losses')
p.extra_y_ranges = {'Validation Loss': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Validation Loss', axis_label='Validation Loss'), 'right')
p.line(np.arange(len(loss_list_train)), loss_list_train, legend_label="Train Loss")
p.line(np.arange(len(loss_list_validation)), loss_list_validation, legend_label="Validation Loss",y_range_name='Validation Loss', color='red')
p.legend.location = "top_right"
show(p)

# Question f
# Load the model 
load_model = torch.load(MODEL__DIR +'epoch'+ str(num_epochs)+ 'FNN_net_model.ckpt')
# Make prediction on the test set
load_model.eval()
with torch.no_grad():
    plt.figure(figsize=(12,3))              # show the examples in a plot
    output = load_model(x_noisy_example).resize(64,1,32,32).cpu() # resize our output as the same size of input and transfer back to cpu
    for i in range(10):
        plt.subplot(2,10,i+1)
        plt.imshow(x_clean_example[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2,10,i+11)
        plt.imshow(output[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.savefig("trained_test_data_examples.png",dpi=300,bbox_inches='tight')
    plt.show()
