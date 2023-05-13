import mindspore.nn as nn
import mindspore.ops.operations as P
from cfg import cfg

class Generator(nn.Cell):
    def __init__(self, num_filters=cfg.num_filters):
        super(Generator, self).__init__()

        self.op = P.Concat(1)

        self.down1 = nn.SequentialCell(
            nn.Conv2d(3, num_filters,2),
            nn.BatchNorm2d(num_filters),nn.ReLU()
        )
        self.down2 = nn.SequentialCell(
            nn.Conv2d(num_filters, num_filters*2,2),
            nn.BatchNorm2d(num_filters*2),nn.ReLU()
        )
        self.down3 = nn.SequentialCell(
            nn.Conv2d(num_filters*2, num_filters*4,2),
            nn.BatchNorm2d(num_filters*4),nn.ReLU()
        )
        self.down4 = nn.SequentialCell(
            nn.Conv2d(num_filters*4, num_filters*8,2),
            nn.BatchNorm2d(num_filters*8),nn.ReLU()
        )
        self.down5 = nn.SequentialCell(
            nn.Conv2d(num_filters*8, num_filters*8,2),
            nn.BatchNorm2d(num_filters*8),nn.ReLU()
        )
        self.down6 = nn.SequentialCell(
            nn.Conv2d(num_filters*8, num_filters*8,2),
            nn.BatchNorm2d(num_filters*8),nn.ReLU()
        )
        self.down7 = nn.SequentialCell(
            nn.Conv2d(num_filters*8, num_filters*8,2),
            nn.BatchNorm2d(num_filters*8),nn.ReLU()
        )
        self.down8 = nn.SequentialCell(
            nn.Conv2d(num_filters*8, num_filters*8,2),
            nn.BatchNorm2d(num_filters*8),nn.ReLU()
        )

        self.up1 = nn.SequentialCell(
            nn.Conv2dTranspose(num_filters*8, num_filters*8,2,dilation=1),
            nn.BatchNorm2d(num_filters*8),nn.ReLU(),
        )
        self.up2 = nn.SequentialCell(
            nn.Conv2dTranspose(num_filters*8*2, num_filters*8,2,dilation=1),
            nn.BatchNorm2d(num_filters*8),nn.ReLU(),
        )
        self.up3 = nn.SequentialCell(
            nn.Conv2dTranspose(num_filters*8*2, num_filters*8,2,dilation=1),
            nn.BatchNorm2d(num_filters*8),nn.ReLU(),
        )
        self.up4 = nn.SequentialCell(
            nn.Conv2dTranspose(num_filters*8*2, num_filters*8,2,dilation=1),
            nn.BatchNorm2d(num_filters*8),nn.ReLU(),
        )
        self.up5 = nn.SequentialCell(
            nn.Conv2dTranspose(num_filters*8*2, num_filters*4,2,dilation=1),
            nn.BatchNorm2d(num_filters*4),nn.ReLU(),
        )
        self.up6 = nn.SequentialCell(
            nn.Conv2dTranspose(num_filters*4*2, num_filters*2,2,dilation=1),
            nn.BatchNorm2d(num_filters*2),nn.ReLU(),
        )
        self.up7 = nn.SequentialCell(
            nn.Conv2dTranspose(num_filters*2*2, num_filters,2,dilation=1),
            nn.BatchNorm2d(num_filters),nn.ReLU(),
        )
        self.up8 = nn.SequentialCell(
            nn.Conv2dTranspose(num_filters*2, 3,2,dilation=1),
            nn.BatchNorm2d(3),nn.Tanh(),
        )

    def construct(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x8 = self.down8(x7)

        x = self.up1(x8)
        x = self.op((x, x7))
        x = self.up2(x)
        x = self.op((x, x6))
        x = self.up3(x)
        x = self.op((x, x5))
        x = self.up4(x)
        x = self.op((x, x4))
        x = self.up5(x)
        x = self.op((x, x3))
        x = self.up6(x)
        x = self.op((x, x2))
        x = self.up7(x)
        x = self.op((x, x1))
        x = self.up8(x)
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
