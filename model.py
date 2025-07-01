import torch
import torch.nn as nn

class DenoisingCNN(nn.Module):#nn.Model类中有已经实现好的_call_方法，所以调用model时，会自动调用_call_方法
    def __init__(self):
        super(DenoisingCNN, self).__init__()
        
        # Encoder - downsample the input image
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Input: 3 channels (RGB)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Halve the spatial dimensions
        )
        
        # Middle part of the network
        self.middle = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder - upsample to get back to original size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # Upsample
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),  # Output: 3 channels (RGB)
            nn.Sigmoid()  # Scales output to [0, 1] range
        )

    def forward(self, x):
        encoded = self.encoder(x)
        middle = self.middle(encoded)
        decoded = self.decoder(middle)
        return decoded