import matplotlib.pyplot as plt
import time
start_time = time.time()

import sys
import os

sys.path.append('.')

from math import gcd
from functools import reduce

import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from kan import *

torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data_dir = os.path.join('.', 'data')

# KAN model properties
input_size = 5000
hidden_layers = 3
output_size = 5000
epochs = 100
learning_rate = 0.001
train_samples = 5000

model = MultKAN(
    ckpt_path="./model",
    width=[input_size, hidden_layers, output_size], 
    grid=5, 
    k=3, 
    seed=42, 
    device=device, 
    noise_scale=1
)

# Load train and test dataset
df_train = pd.read_pickle(os.path.join(data_dir, "train_data.pkl"))
df_test = pd.read_pickle(os.path.join(data_dir, "test_data.pkl"))

# Graph the train dataset
"""
fig, ax = plt.subplots(1, 1, sharex='col', figsize=(6, 2))
fig.suptitle('Training Dataset')
ax.plot(df_train.iloc[1,:], color='black', linewidth=1)
ax.grid(True)

plt.xticks(rotation=20)
fig.tight_layout()'
"""

train_tmp = []
test_tmp = []
for col in df_train.columns: 
    train_tmp.append(df_train[col])
    test_tmp.append(df_test[col])
dataset = {}
dataset["train_input"] = torch.transpose(torch.Tensor(train_tmp), 0, 1)
dataset["train_label"] = torch.transpose(torch.Tensor(train_tmp), 0, 1)
dataset["test_input"] = torch.transpose(torch.Tensor(test_tmp), 0, 1)
dataset["test_label"] = torch.transpose(torch.Tensor(test_tmp), 0, 1)

if ( torch.cuda.is_available() ):
  dataset["train_input"] = dataset["train_input"].cuda()
  dataset["train_label"] = dataset["train_label"].cuda()
  dataset["test_input"] = dataset["test_input"].cuda()
  dataset["test_label"] = dataset["test_label"].cuda()

model(dataset['train_input'])
# model.plot() # Disabled it takes too much time

# train the model
model.fit(dataset, opt="LBFGS", steps=epochs);

model.fit(dataset, opt="LBFGS", steps=epochs);
# model.plot() # Disabled it takes too much time
finish_time = time.time()
time_in_secs = finish_time - start_time
print(f"Elapsed Time: {time_in_secs} seconds")
