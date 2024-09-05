# We use Binary Cross- Entropy Loss - nn.BCEWithLogitsLoss() as our loss function.
# This combines a Sigmoid activation and the Binary Cross-Entropy loss in one single class, which is numerically more stable.

# Loss function:

import torch
import torch.nn as nn
import torch.optim as optim

criterion = nn.BCEWithLogitsLoss()

