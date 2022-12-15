import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        #self.V = nn.Linear(fc2_units, 1)
        #self.A = nn.Linear(fc2_units, action_size)

    
    def forward(self, state):
        ###Build a network that maps state -> action values.
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


    #def forward(self, state):

    #    x = F.relu(self.fc1(state))
    #    x = F.relu(self.fc2(x))

    #    V = self.V(x)
    #    A = self.A(x)
    #    V = V.expand_as(A)

    #    q = V + A - A.mean(dim = 1, keepdim = True).expand_as(A)

    #    return q



    #    return V + (A - A.mean(dim = 1, keepdim = True))

        
