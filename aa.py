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
from mindspore import Tensor,context

# context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

weight_init = Normal(mean=0, sigma=0.02)
gamma_init = Normal(mean=1, sigma=0.02)

def get_train_data():
    train_x = []
    train_y = []
    for i in range(cfg.train_size):
        x_path = './train_A/'+str(i)+'.png'
        y_path = './train_B/'+str(i)+'.png'
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
y_train = y_train / 127.5 - 1
e = datetime.datetime.now()
print("用时:",e-s)

def gen_data(X,Y):
    X = Tensor.from_numpy(X).astype(np.float32)
    Y = Tensor.from_numpy(Y).astype(np.float32)
    XY = list(zip(X,Y))
    dataset = ds.GeneratorDataset(XY,["input_images", "target_images"])
    return dataset

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

# g_grads = grads_gen(generator_forward,g_opt.parameters)(real_a,valid)
# g_opt(g_grads) 报错

# grads_gen = ops.GradOperation(get_all=True)
# grads_dis = ops.GradOperation(get_all=True)
# def train_step2(real_a,real_b):
#     real_a = real_a.reshape(1,3,512,512)
#     real_b = real_b.reshape(1,3,512,512)
#     valid = Tensor(np.ones((1,1,1,1)).astype(np.float32))
#     fake = Tensor(np.zeros((1,1,1,1)).astype(np.float32))
#     (g_loss, gen_imgs) = generator_forward(real_a,valid)
#     g_grads = grads_gen(generator_forward,g_opt.parameters)(real_a,valid)
#     g_opt(g_grads)
#     d_loss = discriminator_forward(gen_imgs,real_b,valid,fake)
#     d_grads = grads_dis(discriminator_forward,d_opt.parameters)(gen_imgs,real_b,valid,fake)
#     d_opt(d_grads)
#     return g_loss, d_loss

grad_generator_fn = ops.value_and_grad(generator_forward, None,
                                       g_opt.parameters)
grad_discriminator_fn = ops.value_and_grad(discriminator_forward, None,
                                           d_opt.parameters)

def train_step(real_a,real_b):
    real_a = real_a.reshape(1,3,512,512)
    real_b = real_b.reshape(1,3,512,512)
    valid = Tensor(np.ones((1,1,1,1)).astype(np.float32))
    fake = Tensor(np.zeros((1,1,1,1)).astype(np.float32))
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