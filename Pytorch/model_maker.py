"""
This is the module where the model is defined. It uses the nn.Module as backbone to create the network structure
"""
# Own modules

#Built in

# Libs

# Pytorch module
import torch.nn as nn
import torch.nn.functional as F

class Forward(nn.Module):
    def __init__(self, flags):
        super(Forward, self).__init__()

        # Linear Layer and Batch_norm Layer definitions here
        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        for ind, fc_num in enumerate(flags.linear[0:-1]):               # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, flags.linear[ind + 1]))
            self.bn_linears.append(nn.BatchNorm1d(flags.linear[ind + 1]))


        # Conv Layer definitions here
        self.convs = nn.ModuleList([])
        in_channel = 1                                                  # Initialize the in_channel number
        for ind, (out_channel, kernel_size, stride) in enumerate(zip(flags.conv_out_channel,
                                                                     flags.conv_kernel_size,
                                                                     flags.conv_stride)):
            self.convs.append(nn.ConvTranspose1d(in_channel, out_channel, kernel_size,
                              stride=stride, padding=kernel_size/2 + 1)) # To make sure L_out double each time


    def forward(self, G):
        """
        The forward function which defines how the network is connected
        :param G: The input geometry (Since this is a forward network)
        :return: S: The 300 dimension spectra
        """
        out = G                                                         # initialize the out
        # For the linear part
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear

        # For the conv part
        for ind, conv in enumerate(self.convs):
            out = conv(out)

        # Final touch, because the input is normalized to [-1,1]
        S = F.tanh(out)
        return S

