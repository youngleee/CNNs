# We use Binary Cross- Entropy Loss - nn.BCEWithLogitsLoss() as our loss function.
# This combines a Sigmoid activation and the Binary Cross-Entropy loss in one single class, which is numerically more stable.

# Loss function:

import torch
import torch.nn as nn
import torch.optim as optim

criterion = nn.BCEWithLogitsLoss()

# Adam-Optimizer initialisieren
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# def train(model, optimizer, criterion, inputs, labels):
#     optimizer.zero_grad()
#     outputs = model(inputs)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()
#     return loss.item()