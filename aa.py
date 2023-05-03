import os
import datetime
import numpy as np

from easydict import EasyDict as edict
from PIL import Image

import mindspore
import mindspore.ops as ops
import mindspore.dataset
import mindspore.nn as nn
from mindspore import dataset as ds
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.common.initializer import Normal
from mindspore import Tensor

cfg = edict({
    'train_size': 400,  # 训练集大小
    'test_size': 100,  # 测试集大小
    'channel': 3,  # 图片通道数
    'image_height': 512,  # 图片高度
    'image_width': 512,  # 图片宽度
    'epoch_size': 20,  # 训练次数
    'epoch_num': 10,
    'learn_rate': 1e-7,
    'ckpt_dir':  "ckpt_dir"
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
    

x_train,y_train,x_test,y_test = get_train_data()
x_train = x_train.reshape(cfg.train_size,cfg.channel,cfg.image_width,cfg.image_height)
y_train = y_train.reshape(cfg.train_size,cfg.channel,cfg.image_width,cfg.image_height)
x_test = x_test.reshape(cfg.test_size,cfg.channel,cfg.image_width,cfg.image_height)
y_test = y_test.reshape(cfg.test_size,cfg.channel,cfg.image_width,cfg.image_height)

def gen_data(X,Y):
    X = Tensor.from_numpy(X).astype(np.float32)
    Y = Tensor.from_numpy(Y).astype(np.float32)
    XY = list(zip(X,Y))
    dataset = ds.GeneratorDataset(XY,["input_images", "target_images"])
    return dataset

class Generator(nn.Cell):
    def __init__(self, num_filters=64):
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
            nn.BatchNorm2d(num_filters*8),nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.SequentialCell(
            nn.Conv2dTranspose(num_filters*8, num_filters*8,2,dilation=1),
            nn.BatchNorm2d(num_filters*8),nn.ReLU(),
            nn.Conv2dTranspose(num_filters*8, num_filters*8*2,2,dilation=1),
            nn.BatchNorm2d(num_filters*16),nn.ReLU(),
            nn.Conv2dTranspose(num_filters*8*2, num_filters*8*2,2,dilation=1),
            nn.BatchNorm2d(num_filters*16),nn.ReLU(),
            nn.Conv2dTranspose(num_filters*8*2, num_filters*8,2,dilation=1),
            nn.BatchNorm2d(num_filters*8),nn.ReLU(),
            nn.Conv2dTranspose(num_filters*8, num_filters*4,2,dilation=1),
            nn.BatchNorm2d(num_filters*4),nn.ReLU(),
            nn.Conv2dTranspose(num_filters*4, num_filters*2,2,dilation=1),
            nn.BatchNorm2d(num_filters*2),nn.ReLU(),
            nn.Conv2dTranspose(num_filters*2, num_filters,2,dilation=1),
            nn.BatchNorm2d(num_filters),nn.ReLU(),
            nn.Conv2dTranspose(num_filters, 3, 2,dilation=1),
            nn.BatchNorm2d(3),nn.ReLU(),
        )

    def construct(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Cell):
    def __init__(self, num_filters=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, 2)
        self.conv2 = nn.Conv2d(num_filters, num_filters*2, 2)
        self.bn2 = nn.BatchNorm2d(num_filters*2)
        self.conv3 = nn.Conv2d(num_filters*2, num_filters*4, 2)
        self.bn3 = nn.BatchNorm2d(num_filters*4)
        self.conv4 = nn.Conv2d(num_filters*4, num_filters*8, 2)
        self.bn4 = nn.BatchNorm2d(num_filters*8)
        self.conv5 = nn.Conv2d(num_filters*8, 1, 2)

    def construct(self, x):
        x = nn.LeakyReLU(0.2)(self.conv1(x))
        x = nn.LeakyReLU(0.2)(self.bn2(self.conv2(x)))
        x = nn.LeakyReLU(0.2)(self.bn3(self.conv3(x)))
        x = nn.LeakyReLU(0.2)(self.bn4(self.conv4(x)))
        x = self.conv5(x)
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
loss_f = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

def forword_dis(reala, realb):
    lambda_dis = 0.5
    fake = net_generator(reala)
    pred0 = net_discriminator(reala)
    pred1 = net_discriminator(realb)
    pred2 = net_discriminator(fake)
    loss_d = loss_f(pred0, pred1) + loss_f(pred0, pred2)
    loss_dis = loss_d * lambda_dis
    return loss_dis

def forword_gan(reala, realb):
    lambda_gan = 0.5
    lambda_l1 = 100
    fakeb = net_generator(reala)
    loss_1 = loss_f(reala, ops.ones_like(reala))
    loss_2 = l1_loss(fakeb, realb)
    loss_gan = loss_1 * lambda_gan + loss_2 * lambda_l1
    return loss_gan

d_opt = nn.Adam(net_discriminator.trainable_params(), learning_rate=cfg.learn_rate,
                beta1=0.5, beta2=0.999, loss_scale=1)
d_opt.update_parameters_name('optim_d')
g_opt = nn.Adam(net_generator.trainable_params(), learning_rate=cfg.learn_rate,
                beta1=0.5, beta2=0.999, loss_scale=1)
g_opt.update_parameters_name('optim_g')

grad_d = ops.value_and_grad(forword_dis, None, d_opt.parameters)
grad_g = ops.value_and_grad(forword_gan, None, g_opt.parameters)

def train_step(reala, realb):
    loss_dis, d_grads = grad_d(reala, realb)
    loss_gan, g_grads = grad_g(reala, realb)
    d_opt(d_grads)
    g_opt(g_grads)
    return loss_dis, loss_gan


g_losses = []
d_losses = []
dataset = gen_data(x_train,y_train)
data_loader = dataset.create_dict_iterator(output_numpy=True, num_epochs=cfg.epoch_num)

if not os.path.isdir(cfg.ckpt_dir):
    os.makedirs(cfg.ckpt_dir)

for epoch in range(cfg.epoch_num):
    for i, data in enumerate(data_loader):
        start_time = datetime.datetime.now()
        input_image = Tensor(data["input_images"])
        target_image = Tensor(data["target_images"])
        input_image = input_image.reshape(1,3,512,512)
        target_image = target_image.reshape(1,3,512,512)
        dis_loss, gen_loss = train_step(input_image, target_image)
        end_time = datetime.datetime.now()
        delta = (end_time - start_time).microseconds
        if i % 2 == 0:
            print("ms per step:{:.2f}  epoch:{}/{}  step:{}/{}  Dloss:{:.4f}  Gloss:{:.4f} ".format((delta / 1000), (epoch + 1), (cfg.epoch_num), i, float(dis_loss), float(gen_loss)))
        d_losses.append(dis_loss)
        g_losses.append(gen_loss)
    if (epoch + 1) == cfg.epoch_num:
        mindspore.save_checkpoint(net_generator, cfg.ckpt_dir + "Generator.ckpt")