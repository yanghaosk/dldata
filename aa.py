import datetime
import numpy as np

from cfg import cfg
from PIL import Image

from net import Generator,Discriminator

import mindspore
from mindspore import ops
import mindspore.nn as nn
from mindspore import dataset as ds
from mindspore.common.initializer import Normal
from mindspore import Tensor

# context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

weight_init = Normal(mean=0, sigma=0.02)
gamma_init = Normal(mean=1, sigma=0.02)

def get_train_data():
    train_x = []
    train_y = []
    for i in range(cfg.train_size):
        x_path = f'./train_A/{str(i)}.png'
        y_path = f'./train_B/{str(i)}.png'
        image_x = Image.open(x_path)
        train_x.append(np.array(image_x))
        image_y = Image.open(y_path)
        train_y.append(np.array(image_y))
    return np.array(train_x),np.array(train_y)

print("开始读取数据")
s = datetime.datetime.now()
x_train,y_train = get_train_data()
print("读取结束")
x_train = x_train.reshape(cfg.train_size,cfg.channel,cfg.image_width,cfg.image_height)
y_train = y_train.reshape(cfg.train_size,cfg.channel,cfg.image_width,cfg.image_height)
x_train = x_train / 127.5 - 1
y_train = y_train / 127.5 - 1
e = datetime.datetime.now()
print("用时:",e-s)

def gen_data(X,Y):
    X = Tensor.from_numpy(X).astype(np.float32)
    Y = Tensor.from_numpy(Y).astype(np.float32)
    XY = list(zip(X,Y))
    return ds.GeneratorDataset(XY,["input_images", "target_images"])

net_generator = Generator()
net_discriminator = Discriminator()
adversarial_loss = nn.BCELoss(reduction='mean')
L1_loss = nn.loss.L1Loss()
color_loss = nn.loss.MSELoss()

d_opt = nn.Adam(net_discriminator.trainable_params(), learning_rate=cfg.learn_rate)
d_opt.update_parameters_name('optim_d')
g_opt = nn.Adam(net_generator.trainable_params(), learning_rate=cfg.learn_rate*2)
g_opt.update_parameters_name('optim_g')

def generator_forward(real_a,real_b,valid):
    gen_imgs = net_generator(real_a)
    g_loss1 = adversarial_loss(net_discriminator(gen_imgs), valid)
    g_loss2 = L1_loss(gen_imgs,real_b)
    g_loss3 = color_loss(gen_imgs,real_b)
    g_loss = g_loss1 + g_loss2*100 + g_loss3*30
    return g_loss, gen_imgs

def discriminator_forward(gen_imgs, real_b,valid,fake):
    real_loss = adversarial_loss(net_discriminator(real_b), valid)
    fake_loss = adversarial_loss(net_discriminator(gen_imgs), fake)
    return (real_loss + fake_loss) / 2

grad_generator_fn = ops.value_and_grad(generator_forward, None,
                                       g_opt.parameters)
grad_discriminator_fn = ops.value_and_grad(discriminator_forward, None,
                                           d_opt.parameters)

valid = Tensor(np.ones((1,1,1,1)).astype(np.float32))
fake = Tensor(np.zeros((1,1,1,1)).astype(np.float32))
def train_step(real_a,real_b):
    real_a = real_a.reshape(1,3,512,512)
    real_b = real_b.reshape(1,3,512,512)
    (g_loss, gen_imgs), g_grads = grad_generator_fn(real_a,real_b,valid)
    g_opt(g_grads)
    d_loss, d_grads = grad_discriminator_fn(gen_imgs, real_b,valid,fake)
    d_opt(d_grads)
    return g_loss, d_loss

dataset = gen_data(x_train,y_train)

# for epoch in range(cfg.epoch_num):
#     net_generator.set_train()
#     net_discriminator.set_train()
#     # 为每轮训练读入数据
#     for i, data in enumerate(data_loader):
#         input_image = Tensor(data["input_images"])
#         target_image = Tensor(data["target_images"])
#         g_loss, d_loss = train_step(input_image,target_image)
#         if i % (cfg.train_size/20) == 1:
#             print("epoch:{}/{} step:{}/{} Dloss:{:.6f}  Gloss:{:.6f} ".format(
#                 (epoch + 1), (cfg.epoch_num), int(i/20), cfg.epoch_size, float(d_loss), float(g_loss)))
#     if (epoch + 1) == cfg.epoch_num:
#         mindspore.save_checkpoint(net_generator, "./model/Generator"+".ckpt")

epoch = 0
while 1:
    net_generator.set_train()
    net_discriminator.set_train()
    # 为每轮训练读入数据
    data_loader = dataset.create_dict_iterator(output_numpy=True, num_epochs=cfg.epoch_size)
    for i, data in enumerate(data_loader):
        input_image = Tensor(data["input_images"])
        target_image = Tensor(data["target_images"])
        g_loss, d_loss = train_step(input_image,target_image)
        if i % (cfg.train_size/20) == 1:
            print("epoch:{}/{} step:{}/{} Dloss:{:.6f}  Gloss:{:.6f} ".format(
                (epoch + 1), (cfg.epoch_num), int(i/20), 20, float(d_loss), float(g_loss)))
    if int((epoch + 1)%cfg.epoch_num) == 0:
        mindspore.save_checkpoint(
            net_generator,
            f"./model/Generator{str((epoch + 1) / cfg.epoch_num)}.ckpt",
        )
    epoch += 1
