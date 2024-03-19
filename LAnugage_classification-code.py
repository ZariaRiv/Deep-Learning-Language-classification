import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import torchaudio
from copy import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#################################################
# Data Loading
#################################################
sampling_rate = 8000
#languages = ["de", "en", "es", "fr", "nl", "pt"] # German, English, Spanish, French, Dutch, Portuguese
#language_dict = {languages[i]: i for i in range(len(languages))}

# load the training data
X_train, y_train = np.load("dataset/inputs_train_fp16.npy"), np.load(
    "dataset/targets_train_int8.npy"
)

# load the testing data
X_test, y_test = np.load("dataset/inputs_test_fp16.npy"), np.load(
    "dataset/targets_test_int8.npy"
)

X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

class CustomDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]


def calculate_min_max(x):
    min_val = np.min(x)
    max_val = np.max(x)
    return min_val, max_val

min_train, max_train = calculate_min_max(X_train)

#then use the class
train_dataset = CustomDataset(X_train, torch.tensor(y_train, dtype=torch.long))
test_dataset = CustomDataset(X_test, torch.tensor(y_test, dtype=torch.long))

#################################################


#################################################
# Model Definition
#################################################
class CombinedModel(nn.Module):
    def __init__(self, n_cnn_l, n_rnn_l, min_val, max_val):
      super(CombinedModel, self).__init__()
      self.min = torch.tensor(min_val)
      self.max = torch.tensor(max_val)

      # MFCC transform
      self.mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=8000,
        n_mfcc=20,
        melkwargs={'n_fft': 2048, 'n_mels': 128, 'hop_length': 512, 'win_length': 400}
      )

      # CNN layers
      layers = [nn.Conv2d(1, 32, kernel_size=(3, 3)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2))]

      for _ in range(n_cnn_l - 1):
        layers.extend([nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
                      nn.ReLU(),
                      nn.MaxPool2d(kernel_size=(2, 2))])

      self.conv_layers = nn.Sequential(*layers)

      # Dummy input to compute the flattened size after the CNN layers
      dummy_input = torch.randn(1, 1, 20, 79)
      dummy_output = self.conv_layers(dummy_input)
      self.rnn_input_size = dummy_output.view(1, -1).size(-1)

      # RNN layers
      # Update input_size based on the output size of the last convolutional layer
      self.rnn = nn.LSTM(input_size=self.rnn_input_size, hidden_size=64, num_layers=n_rnn_l, batch_first=True)

      # Fully connected layers
      self.linear1 = nn.Linear(64, 32)
      self.relu = nn.ReLU()
      self.linear2 = nn.Linear(32, 6)

    def forward(self, x):
        if len(x.shape) < 3: #to configure it to one sample in the input
          x = x.unsqueeze(0)
        #preprocessing
          #normalize:
        x = (x - self.min) / (self.max - self.min + 1e-7)
            #produce spectrograms:
        x = self.mfcc_transform(x) 

         # CNN layers
        # x = self.conv_layers(x)
        for i, layer in enumerate(self.conv_layers):
          x = layer(x)
          # print(f"conv_2d_{i}: {x.size()}")

        # Reshape the tensor for RNN input
        x = x.view(x.size(0), -1, self.rnn_input_size)
        # print(f"reshaped: {x.size()}")

        # RNN layers
        out, _ = self.rnn(x)
        x = out
        # print(f"rnn: {x.size()}")

        # Extract the last output of the RNN
        x = x[:, -1, :]
        # print(f"rnn output: {x.size()}")

        # Fully connected layers
        x = self.linear1(x)
        # print(f"linear_1: {x.size()}")
        x = self.relu(x)
        # print(f"relu: {x.size()}")
        x = self.linear2(x)
        # print(f"linear_2: {x.size()}")

        return x


model = CombinedModel(4, 1, min_train, max_train) #one can customize N layers
#################################################


################################################
# Training function
################################################
def train_model(model, batch_size, lr, **kwargs):
  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  model.to(device)

  criterion = nn.CrossEntropyLoss().to(device)
  optimizer = optim.Adam(model.parameters(), lr = lr, **kwargs)

  train_loss_l = []
  val_loss_l = []

  train_acc_l = []
  val_acc_l = []

  for epoch in range(40):
    print("Entering epoch {} / 40".format(epoch+1))
    #train
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0
    i = 0
    for inputs, targets in train_loader:
        i+=1
        inputs, targets = inputs.unsqueeze(1).to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.squeeze(), targets)
        loss.backward()
        optimizer.step()
        preds = torch.sigmoid(logits)
        _, preds = torch.max(logits, 1)
        train_loss += loss.item()
        train_correct += torch.sum(preds == targets.data)
        train_total += targets.size(0)
        if i%10==0 and i!=0:
            print("TRAIN step",i,", loss:",train_loss/10,", accuracy:",(train_correct/train_total).item())
            train_loss_l.append(train_loss/10)
            train_acc_l.append((train_correct/train_total).item())
            train_loss = 0
            train_total = 0
            train_correct = 0

    #validate
    val_correct = 0
    val_total = 0
    val_loss = 0
    i = 0
    model.eval()
    for inputs, targets in test_loader:
        i+=1
        inputs, targets = inputs.unsqueeze(1).to(device), targets.to(device)
        logits = model(inputs)
        _, preds = torch.max(logits, 1)
        val_loss += loss.item()
        val_correct += torch.sum(preds == targets.data)
        val_total += targets.size(0)
        if i%10==0 and i!=0:
            print("VALIDATION step",i,", loss:",val_loss/10,", accuracy:",(val_correct/val_total).item())
            val_loss_l.append(val_loss/10)
            val_acc_l.append((val_correct/val_total).item())
            val_loss = 0
            val_total = 0
            val_correct = 0

  return model, test_loader, train_loss_l, val_loss_l, train_acc_l, val_acc_l
################################################


################################################
# Train/Test the model
################################################
# The model performed best when trained with these hyperparameters
trained_model, test_loader, train_loss_l, val_loss_l, train_acc_l, val_acc_l = train_model(model, batch_size=20, lr=0.0001)

# Compute the normalized x-axis values for the training and validation sets
train_iters = len(train_loss_l)
val_iters = len(val_loss_l)
train_x = np.linspace(0, 1, train_iters)
val_x = np.linspace(0, 1, val_iters)

# Plot the training and validation losses
plt.plot(train_x, train_loss_l, label='Train Loss')
plt.plot(val_x, val_loss_l, label='Validation Loss')

# Add labels and legend
plt.title("Loss over steps")
plt.xlabel('Normalized Iterations')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.pdf')
plt.clf()

train_acc_np = [float(acc) for acc in train_acc_l]
val_acc_np = [float(acc) for acc in val_acc_l]

# Plot the training and validation losses
plt.plot(train_x, train_acc_l, label='Train Acc')
plt.plot(val_x, val_acc_l, label='Validation Acc')

# Add labels and legend
plt.title("Accuracy over steps")
plt.xlabel('Normalized Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy.pdf')
plt.clf()
