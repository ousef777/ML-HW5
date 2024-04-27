import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pandas import DataFrame
import time
start_time = time.time()

from shakespeareLoader import train_loader, test_loader


X_train, y_train = next(iter(train_loader))
X_val, y_val = next(iter(test_loader))

# Check if GPU is available and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach().to(device)

# Defining the Transformer model
class CharTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nhead):
        super(CharTransformer, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)  # Softmax layer over the feature dimension

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.pos_encoder(embedded)
        transformer_output = self.transformer_encoder(embedded)
        output = self.fc(transformer_output)
        return self.softmax(output)  # Apply softmax to the linear layer output

# Hyperparameters
hidden_size = 128
num_layers = 3
nhead = 2
learning_rate = 0.005
epochs = 1

# Model, loss, and optimizer
model = CharTransformer(len(X_train), hidden_size, len(X_train), num_layers, nhead).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(epochs):
    model.train()
    for X_train, y_train in train_loader:
        X_train, y_train = X_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        output = model(X_train)
        #print(output.transpose(1, 2).size())
        #print(y_train.size())
        loss = criterion(output.transpose(1, 2)[:,:,-1], y_train)  # Reshape output to match the CrossEntropyLoss expectations
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        for X_val, y_val in test_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            val_output = model(X_val)
            val_loss = criterion(val_output.transpose(1, 2)[:,:,-1], y_val)
            _, predicted = torch.max(val_output.transpose(1, 2), 2)
            if (predicted.size() == torch.Size([128, 128])):
                val_accuracy = (predicted == y_val).float().mean()

    if (epoch+1) % 1 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_accuracy.item()}')

print("--- %s seconds ---" % (time.time() - start_time))