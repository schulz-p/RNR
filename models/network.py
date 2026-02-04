import torch
from torch import nn

""" Residual block: component of ResNet """
class ResBlock(nn.Module):
    def __init__(self, in_out_channels, hidden_channels):
        super().__init__()

        # Layer initialization
        self.linear1 = nn.Linear(in_out_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, hidden_channels)
        self.linear3 = nn.Linear(hidden_channels, in_out_channels)

        # Weight initialization
        std1 = 1E0
        std2 = 1E-3
        torch.nn.init.normal_(self.linear1.weight, std=std1)
        torch.nn.init.normal_(self.linear1.bias, std=std1)
        torch.nn.init.normal_(self.linear2.weight, std=std1)
        torch.nn.init.normal_(self.linear2.bias, std=std1)
        torch.nn.init.normal_(self.linear3.weight, std=std2)
        torch.nn.init.normal_(self.linear3.bias, std=std2)


    def forward(self, input):

        # Propagate through layers
        velocity = self.linear1(input)
        velocity = torch.sin(velocity)
        velocity = self.linear2(velocity)
        velocity = torch.sin(velocity)
        velocity = self.linear3(velocity)

        # Add skip connection
        output = input + velocity

        output = torch.reshape(output, (input.size(dim=0), input.size(dim=1)))
        velocity = torch.reshape(velocity, (input.size(dim=0), input.size(dim=1)))

        return output, velocity


""" Residual network: consists of stacked ResBlocks """
class ResNet(nn.Module):
    def __init__(self, n_layers, hidden_channels, shared_weights):
        super(ResNet, self).__init__()

        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.shared_weights = shared_weights

        # Load one residual block, which is used in case of shared weights for all layers
        block = ResBlock(3, hidden_channels)

        # Make layers
        self.layers = []
        for i in range(self.n_layers):
            if self.shared_weights:
                # Same ResBlock is used for all time steps
                self.layers.append(block)
            else:
                # Individual ResBlock is used for each time step
                self.layers.append(ResBlock(3, hidden_channels))

        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)


    def forward(self, x, steps="all", get_velocity=False):

        if steps == "all":
            steps = self.n_layers

        velocity = []
        # Propagate through layers
        for j in range(steps):
            x, v = self.layers[j](x)
            velocity.append(v)

        if get_velocity:
            return x, velocity
        return x
