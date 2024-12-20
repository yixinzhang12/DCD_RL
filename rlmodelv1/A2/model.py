import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        # TODO YOUR CODE HERE FOR INITIALIZING THE MODEL
        # Guidelines for network size: start with 2 hidden layers and maximum 32 neurons per layer
        # feel free to explore different sizes
        self.fc1 = nn.Linear(state_size, 32) 
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        # # TODO YOUR CODE HERE FOR THE FORWARD PASS
        # raise NotImplementedError()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def select_action(self, state):
        self.eval()
        x = self.forward(state)
        self.train()
        return x.max(1)[1].view(1, 1).to(torch.long)
