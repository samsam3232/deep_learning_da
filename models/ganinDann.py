from torch import nn
from models import funcs

class DomainClassifier(nn.Module):

    def __init__(self, device):

        self.discriminator = nn.Sequential(
            funcs.ReverseLayerF(),
            nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Linear(1000, 20),
            nn.ReLU(),
            nn.Linear(20, 1)

        ).to(device)

    def get_discriminator(self):
        return self.discriminator