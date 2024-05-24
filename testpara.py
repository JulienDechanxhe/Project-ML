import matplotlib.pyplot as plt
import scipy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
import keras.utils
import tensorflow as tf
import time
import random

class LeNet5(nn.Module):
    def __init__(self,mydropout):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # C1
            nn.ReLU(),
            nn.Dropout(mydropout),
            nn.AvgPool2d(2),  # S2
            nn.Conv2d(6, 16, 5),  # C3
            nn.ReLU(),
            nn.Dropout(mydropout),
            nn.AvgPool2d(2),  # C4
        )
        self.dense = nn.Sequential(
            nn.Linear(5 * 5 * 16, 120),  # F5
            nn.ReLU(),
            nn.Dropout(mydropout),
            nn.Linear(120, 84),  # F6
            nn.ReLU(),
            nn.Dropout(mydropout),
            nn.Linear(84, 26)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        y = self.dense(x)
        return y


def train_epoch(model, optimizer, num_epochs, batch_size, x_train_tensor, y_train_tensor, x_train):
#    num_epochs = 10
#    batch_size = 100
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
       # Shuffle the data at the beginning of an epoch
       indices = np.arange(len(x_train_tensor))
       np.random.shuffle(indices)
       total_correct_predictions = 0
       total_samples = 0

       for i in range(0, len(x_train), batch_size):
           # Select batch by batch
           batch_indices = indices[i:i + batch_size]
           inputs = x_train_tensor[batch_indices].permute(0, 3, 1, 2)

           labels = y_train_tensor[batch_indices]

           # Forward propagation
           outputs = model(inputs)

           # Compute the loss
           loss = criterion(outputs, labels)

           with torch.no_grad():
               model.eval()
               _, predicted_labels = torch.max(outputs, 1)
               _, target_labels = torch.max(labels, 1)
               batch_correct_predictions = (predicted_labels == target_labels).sum().item()

               # Accumulate the total number of correct predictions and total samples
               total_correct_predictions += batch_correct_predictions
               total_samples += labels.size(0)
               model.train()

           # Back-propagation and optimisation
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

        # Calculate accuracy at the end of the epoch
       epoch_accuracy = total_correct_predictions / total_samples
#       print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {epoch_accuracy:.4f}')


def acctest(model, x_test_tensor, y_test_tensor):
    criterion = nn.CrossEntropyLoss()
    inputs = x_test_tensor.permute(0, 3, 1, 2)
    labels = y_test_tensor
    total_correct_predictions = 0
    total_samples = 0
    # Forward propagation
    model.eval()
    outputs = model(inputs)

    # Compute the loss
    loss_test = criterion(outputs, labels)
    # Set model to evaluation mode and evaluate it
    with torch.no_grad():
       model.eval()
       _, predicted_labels = torch.max(outputs, 1)
       _, target_labels = torch.max(labels, 1)
       batch_correct_predictions = (predicted_labels == target_labels).sum().item()

   # Accumulate the total number of correct predictions and total samples
    total_correct_predictions += batch_correct_predictions
    total_samples += labels.size(0)
    accuracy_test = total_correct_predictions / total_samples
#    print("accuracy_test", accuracy_test)
    return accuracy_test

def objective(config):

    PATH = "C:/Users/user/OneDrive/Bureau/Machine learning/emnist-letters.mat"
    EMNIST = scipy.io.loadmat(PATH)
    x_train = EMNIST["dataset"][0][0][0][0][0][0].astype("float64")
    y_train = EMNIST["dataset"][0][0][0][0][0][1]

    x_test = EMNIST['dataset'][0][0][1][0][0][0].astype("float64")
    y_test = EMNIST['dataset'][0][0][1][0][0][1]

    # Filter out lowercase letters (class labels 1 to 26)
    lowercase_indices_train = np.where((y_train >= 1) & (y_train <= 26))[0]
    lowercase_indices_test = np.where((y_test >= 1) & (y_test <= 26))[0]

    x_train = x_train[lowercase_indices_train]
    y_train = y_train[lowercase_indices_train]

    x_test = x_test[lowercase_indices_test]
    y_test = y_test[lowercase_indices_test]

    # Scaling data
    x_train = (x_train - np.mean(x_train)) / np.std(x_train)
    x_test = (x_test - np.mean(x_train)) / np.std(x_train)

    nb_classes = 26  # Number of classes

    y_train = keras.utils.to_categorical(y_train-1, nb_classes)
    y_test = keras.utils.to_categorical(y_test-1, nb_classes)

    x_train_scaled = x_train.reshape(-1, 28, 28, 1)
    x_test_scaled = x_test.reshape(-1, 28, 28, 1)

    # Padding to have 32x32 images has in the paper about LeNet5
    x_train_padded = np.array(tf.pad(tensor=x_train_scaled, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]]))
    x_test_padded = np.array(tf.pad(tensor=x_test_scaled, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]]))

    # Pytorch needs a special format
    x_train_tensor = torch.tensor(x_train_padded, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    x_test_tensor = torch.tensor(x_test_padded, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    model = LeNet5(config["mydropout"])  # Create a PyTorch conv net
    optimizer = SGD(model.parameters(), lr=config["lr"])
#    start = time.time()
    train_epoch(model, optimizer, config["num_epochs"], config["batch_size"], x_train_tensor, y_train_tensor, x_train)  # Train the model
    acc = acctest(model, x_test_tensor, y_test_tensor)  # Compute test accuracy
#    end = time.time()
#    print('Elapsed time:', end - start)
    return acc




#Below the code to make several tests with different parameters

myseed=random.randint(1,10000)
random.seed(myseed)
#If we want to retry on the same seed
#print(myseed)

myconfig={'mydropout': 0.2, 'lr': 0.08446027404935158, 'batch_size': 256, 'num_epochs': 14}

nbrtest=10

for i in range(1,nbrtest):
#If we want to vary a parameters on the whole domain (!nbrtest has to be adjusted then)
#    mondropout=0.02*i
#    monlr=0.01+i*0.005
#    monbatchsize=15*i
#    monnumepochs=i

#If we want to have random parameters
#    mondropout=random.uniform(0.01,0.6)
#    monlr=random.uniform(0.0001,0.1)
#    monbatchsize=random.choice([32,64,128,256])
#    monnumepochs=random.randint(5,20)

#Uncomment the parameters we wanted to vary
#    myconfig["mydropout"]=mondropout
#    myconfig["lr"]=monlr
#    myconfig["batch_size"]=monbatchsize
#    myconfig["num_epochs"]=monnumepochs

    print("config actuelle", myconfig)

    monacc=objective(myconfig)



#N.B.
#myconfig={'mydropout': 0.2, 'lr': 0.08446027404935158, 'batch_size': 256, 'num_epochs': 14}
#meanacc=0.895500480769231
