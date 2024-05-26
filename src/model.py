import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.block = \
            nn.Sequential(\
                # 1
                nn.Linear(100, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2),

                # 2
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),

                # 3
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),

                # 4
                nn.Linear(512, 398 * 4),
                nn.BatchNorm1d(398 * 4),
                nn.Tanh()
                )

    def forward(self, input):
        output = self.block(input)

        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.block = \
            nn.Sequential(\
                # 1
                nn.Linear(398 * 4, 512),
                # nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),

                # 2 
                nn.Linear(512, 256), 
                # nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),

                # 3
                nn.Linear(256, 128),
                # nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2),

                # 4
                nn.Linear(128, 1),
                # nn.BatchNorm1d(1),
                nn.Sigmoid()
                )

    def forward(self, input):
        output = self.block(input)

        return output


        