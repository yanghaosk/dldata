import mindspore.nn as nn
from cfg import cfg

class Generator(nn.Cell):
    def __init__(self, num_filters=cfg.num_filters):
        super(Generator, self).__init__()
        self.bn = num_filters
        self.relu = nn.ReLU()

        self.encoder = nn.SequentialCell(
            nn.Conv2d(3, num_filters,2),
            nn.BatchNorm2d(num_filters),nn.ReLU(),
            nn.Conv2d(num_filters, num_filters*2,2),
            nn.BatchNorm2d(num_filters*2),nn.ReLU(),
            nn.Conv2d(num_filters*2, num_filters*4,2),
            nn.BatchNorm2d(num_filters*4),nn.ReLU(),
            nn.Conv2d(num_filters*4, num_filters*8,2),
            nn.BatchNorm2d(num_filters*8),nn.ReLU(),
            nn.Conv2d(num_filters*8, num_filters*8,2),
            nn.BatchNorm2d(num_filters*8),nn.ReLU(),
            nn.Conv2d(num_filters*8, num_filters*8,2),
            nn.BatchNorm2d(num_filters*8),nn.ReLU(),
            nn.Conv2d(num_filters*8, num_filters*8,2),
            nn.BatchNorm2d(num_filters*8),nn.ReLU(),
            nn.Conv2d(num_filters*8, num_filters*8,2),
            nn.BatchNorm2d(num_filters*8),nn.ReLU(),
        )
        
        self.decoder = nn.SequentialCell(
            nn.Conv2dTranspose(num_filters*8, num_filters*8,2,dilation=1),
            nn.BatchNorm2d(num_filters*8),nn.ReLU(),
            nn.Conv2dTranspose(num_filters*8, num_filters*8*2,2,dilation=1),
            nn.BatchNorm2d(num_filters*8*2),nn.ReLU(),
            nn.Conv2dTranspose(num_filters*8*2, num_filters*8*2,2,dilation=1),
            nn.BatchNorm2d(num_filters*8*2),nn.ReLU(),
            nn.Conv2dTranspose(num_filters*8*2, num_filters*4,2,dilation=1),
            nn.BatchNorm2d(num_filters*4),nn.ReLU(),
            nn.Conv2dTranspose(num_filters*4, num_filters*2,2,dilation=1),
            nn.BatchNorm2d(num_filters*2),nn.ReLU(),
            nn.Conv2dTranspose(num_filters*2, num_filters,2,dilation=1),
            nn.BatchNorm2d(num_filters),nn.ReLU(),
            nn.Conv2dTranspose(num_filters, num_filters,2,dilation=1),
            nn.BatchNorm2d(num_filters),nn.ReLU(),
            nn.Conv2dTranspose(num_filters, 3,2,dilation=1),
            nn.BatchNorm2d(3),nn.Tanh(),
        )

    def construct(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Cell):
    def __init__(self, num_filters=cfg.num_filters):
        super(Discriminator, self).__init__()
        self.discriminator = nn.SequentialCell(
        nn.Conv2d(3, num_filters, kernel_size=4, stride=2, pad_mode='valid'),
        nn.LeakyReLU(alpha=0.2),
        nn.Conv2d(num_filters, num_filters*2, kernel_size=4, stride=2, pad_mode='valid'),
        nn.BatchNorm2d(num_filters*2),
        nn.LeakyReLU(alpha=0.2),
        nn.Conv2d(num_filters*2, num_filters*4, kernel_size=4, stride=2, pad_mode='valid'),
        nn.BatchNorm2d(num_filters*4),
        nn.LeakyReLU(alpha=0.2),
        nn.Conv2d(num_filters*4, num_filters*8, kernel_size=4, stride=2, pad_mode='valid'),
        nn.BatchNorm2d(num_filters*8),
        nn.LeakyReLU(alpha=0.2),
        nn.Conv2d(num_filters*8, num_filters*8, kernel_size=4, stride=2, pad_mode='valid'),
        nn.BatchNorm2d(num_filters*8),
        nn.LeakyReLU(alpha=0.2),
        nn.Conv2d(num_filters*8, num_filters*8, kernel_size=4, stride=2, pad_mode='valid'),
        nn.BatchNorm2d(num_filters*8),
        nn.LeakyReLU(alpha=0.2),
        nn.Conv2d(num_filters*8, 1, kernel_size=4, stride=2, pad_mode='valid'),
        nn.LeakyReLU(alpha=0.2),
        nn.MaxPool2d(kernel_size=2),
        nn.Sigmoid()
        )

    def construct(self, x):
        x = self.discriminator(x)
        return x

class Pix2Pix(nn.Cell):
    """Pix2Pix模型网络"""
    def __init__(self, Discriminator, Generator):
        super(Pix2Pix, self).__init__(auto_prefix=True)
        self.net_discriminator = Discriminator
        self.net_generator = Generator

    def construct(self, x):
        x = self.net_generator(x)
        return x
