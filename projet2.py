# Importing libraries
import torch
from torch.optim import SGD
import matplotlib.pyplot as plt
import scipy
import numpy as np
import keras.utils
import tensorflow as tf
import time
from LeNet5_NEW import LeNet5

# Retrieving data
PATH = "C:/Users/user/OneDrive/Bureau/Machine learning/emnist-byclass.mat"
EMNIST = scipy.io.loadmat(PATH)
x_train = EMNIST["dataset"][0][0][0][0][0][0].astype("float64")
y_train = EMNIST["dataset"][0][0][0][0][0][1]

x_test = EMNIST['dataset'][0][0][1][0][0][0].astype("float64")
y_test = EMNIST['dataset'][0][0][1][0][0][1]


def monlabel(x):
    label='0'
    if x<10:
        label=str(x)
    elif x<36:
        label=chr(x+87)
    elif x<63:
        label=chr(x+61).upper()

    return label

# Filter out lowercase letters (class labels 1 to 26)
#lowercase_indices_train = np.where((y_train >= 1) & (y_train <= 62))[0]
#lowercase_indices_test = np.where((y_test >= 1) & (y_test <= 62))[0]

#x_train = x_train[lowercase_indices_train]
#y_train = y_train[lowercase_indices_train]

#x_test = x_test[lowercase_indices_test]
#y_test = y_test[lowercase_indices_test]

# Scaling data
#x_train = (x_train - np.mean(x_train)) / np.std(x_train)
#x_test = (x_test - np.mean(x_train)) / np.std(x_train)

nb_classes = 62  # Number of classes

y_train = keras.utils.to_categorical(y_train, nb_classes)
y_test = keras.utils.to_categorical(y_test, nb_classes)

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

start = time.time() # Measuring computation time

num_epochs = 10
batch_size = 200

# Model of CNN
model = LeNet5(nb_classes)
# Loss
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = SGD(model.parameters(), lr=0.05)

# Training
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

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {epoch_accuracy:.4f}')

# Evaluation of the model

# Evaluation loop
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

print(f'TEST : Loss: {loss_test.item():.4f}, Accuracy: {accuracy_test:.4f}')

# Affichage de quelques résultats
#for i in range(20):
#    print('Essai', i, ':\n Predicted =', predicted_labels[i], '\n Correction =', target_labels[i],'\n')


end = time.time()
print('Elapsed time:', end - start)

# Affichage de quelques résultats

# Affichage des images, des vrais labels et des prédictions
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    # Label associé
#    true_label = y_test[i][0].astype(int)  # Convertir le label en lettre minuscule
    true_label = monlabel(np.argmax(y_test[i].astype(int)))
    #ax.set_title(f'True: {true_label}')

    # Prédiction faite par le modèle
    predicted_label = monlabel(np.argmax(outputs.detach().numpy()[i])) # Convertir la prédiction en lettre minuscule

    # Affichage de l'image
    ax.imshow(x_test[i].reshape(28, 28), cmap='gray' if true_label == predicted_label else 'Oranges')
    ax.axis('off')
    title_text = f'True: {true_label}, Predicted: {predicted_label}'
    ax.set_title(title_text, color='green' if true_label == predicted_label else 'red')

plt.tight_layout()
plt.show()
