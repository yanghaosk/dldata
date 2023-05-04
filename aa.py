import os
import datetime
import numpy as np

from easydict import EasyDict as edict
from PIL import Image

import mindspore
from mindspore import ops
import mindspore.nn as nn
from mindspore import dataset as ds
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.common.initializer import Normal
from mindspore import Tensor

cfg = edict({
    'train_size': 10,  # 训练集大小
    'test_size': 10,  # 测试集大小
    'channel': 3,  # 图片通道数
    'image_height': 512,  # 图片高度
    'image_width': 512,  # 图片宽度
    'epoch_size': 20,  # 训练次数
    'epoch_num': 10,
    'learn_rate': 1e-10,
    "num_filters": 3
})

weight_init = Normal(mean=0, sigma=0.02)
gamma_init = Normal(mean=1, sigma=0.02)

def get_train_data():
    train_x = []
    train_y = []
    for i in range(cfg.train_size):
        x_path = './train_A/'+str(i)+'.png'
        y_path = './train_B/'+str(i)+'.png'
        image_x = Image.open(x_path)
        image_y = Image.open(y_path)
        train_x +=image_x.getdata()
        train_y +=image_y.getdata()

    test_x = []
    test_y = []
    for i in range(400,400+cfg.test_size):
        x_path = './test_A/'+str(i)+'.png'
        y_path = './test_B/'+str(i)+'.png'
        image_x = Image.open(x_path)
        image_y = Image.open(y_path)
        test_x.append(list(image_x.getdata()))
        test_y.append(list(image_y.getdata()))
    return np.array(train_x),np.array(train_y),np.array(test_x),np.array(test_y)
    
print("开始读取数据")
s = datetime.datetime.now()
x_train,y_train,x_test,y_test = get_train_data()
print("读取结束")
x_train = x_train.reshape(cfg.train_size,cfg.channel,cfg.image_width,cfg.image_height)
y_train = y_train.reshape(cfg.train_size,cfg.channel,cfg.image_width,cfg.image_height)
x_test = x_test.reshape(cfg.test_size,cfg.channel,cfg.image_width,cfg.image_height)
y_test = y_test.reshape(cfg.test_size,cfg.channel,cfg.image_width,cfg.image_height)
e = datetime.datetime.now()
print("用时:",e-s)

def gen_data(X,Y):
    X = Tensor.from_numpy(X).astype(np.float32)
    Y = Tensor.from_numpy(Y).astype(np.float32)
    XY = list(zip(X,Y))
    dataset = ds.GeneratorDataset(XY,["input_images", "target_images"])
    return dataset

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
        )
        
        self.decoder = nn.SequentialCell(
            nn.Conv2dTranspose(num_filters*8, num_filters*8,2,dilation=1),
            nn.BatchNorm2d(num_filters*8),nn.ReLU(),
            nn.Conv2dTranspose(num_filters*8, num_filters*8*2,2,dilation=1),
            nn.BatchNorm2d(num_filters*16),nn.ReLU(),
            nn.Conv2dTranspose(num_filters*8*2, num_filters*8*2,2,dilation=1),
            nn.BatchNorm2d(num_filters*16),nn.ReLU(),
            nn.Conv2dTranspose(num_filters*8*2, num_filters*4,2,dilation=1),
            nn.BatchNorm2d(num_filters*4),nn.ReLU(),
            nn.Conv2dTranspose(num_filters*4, num_filters,2,dilation=1),
            nn.BatchNorm2d(num_filters),nn.ReLU(),
            nn.Conv2dTranspose(num_filters, 3,2,dilation=1),
            nn.BatchNorm2d(3),nn.ReLU(),
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

net_generator = Generator()
net_discriminator = Discriminator()
adversarial_loss = nn.BCELoss(reduction='mean')

d_opt = nn.Adam(net_discriminator.trainable_params(), learning_rate=cfg.learn_rate,
                beta1=0.5, beta2=0.999, loss_scale=1)
d_opt.update_parameters_name('optim_d')
g_opt = nn.Adam(net_generator.trainable_params(), learning_rate=cfg.learn_rate,
                beta1=0.5, beta2=0.999, loss_scale=1)
g_opt.update_parameters_name('optim_g')

def generator_forward(real_a,valid):
    gen_imgs = net_generator(real_a)
    g_loss = adversarial_loss(net_discriminator(gen_imgs), valid)
    return g_loss, gen_imgs

def discriminator_forward(gen_imgs, real_b,valid,fake):
    real_loss = adversarial_loss(net_discriminator(real_b), valid)
    fake_loss = adversarial_loss(net_discriminator(gen_imgs), fake)
    d_loss = (real_loss + fake_loss) / 2
    return d_loss

grad_generator_fn = ops.value_and_grad(generator_forward, None,
                                       g_opt.parameters,
                                       has_aux=True)
grad_discriminator_fn = ops.value_and_grad(discriminator_forward, None,
                                           d_opt.parameters)

def train_step(real_a,real_b):
    real_a = real_a.reshape(1,3,512,512)
    real_b = real_b.reshape(1,3,512,512)
    valid = Tensor(np.ones(real_a.shape).astype(np.float32))
    fake = Tensor(np.zeros(real_a.shape).astype(np.float32))
    (g_loss, gen_imgs), g_grads = grad_generator_fn(real_a,valid)
    g_opt(g_grads)
    d_loss, d_grads = grad_discriminator_fn(gen_imgs, real_b,valid,fake)
    d_opt(d_grads)
    return g_loss, d_loss

g_losses = []
d_losses = []

dataset = gen_data(x_train,y_train)
data_loader = dataset.create_dict_iterator(output_numpy=True, num_epochs=cfg.epoch_size)

for epoch in range(cfg.epoch_num):
    net_generator.set_train()
    net_discriminator.set_train()
    # 为每轮训练读入数据
    for i, data in enumerate(data_loader):
        start_time = datetime.datetime.now()
        input_image = Tensor(data["input_images"])
        target_image = Tensor(data["target_images"])
        g_loss, d_loss = train_step(input_image,target_image)
        end_time = datetime.datetime.now()
        d_losses.append(d_loss.asnumpy())
        g_losses.append(g_loss.asnumpy())
        delta = (end_time - start_time).microseconds
        if i % 2 == 1:
            print("per step:{:.2f}ms  epoch:{}/{} step:{}/{} Dloss:{:.6f}  Gloss:{:.6f} ".format(
                (delta / 1000), (epoch + 1), (cfg.epoch_num), (i+1)*2, cfg.epoch_size, float(d_loss), float(g_loss)))
        d_losses.append(d_loss)
        g_losses.append(g_loss)
    if (epoch + 1) == cfg.epoch_num:
        mindspore.save_checkpoint(net_generator, "Generator.ckpt")