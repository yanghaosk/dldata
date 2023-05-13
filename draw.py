import matplotlib.pyplot as plt

import mindspore
import numpy as np
from net import Generator
from cfg import cfg
from PIL import Image
from mindspore import Tensor
from mindspore import load_checkpoint, load_param_into_net

def draw(rgb_image):
    # 绘制RGB图像
    rgb_image = rgb_image.astype(int).reshape(512,512,3).asnumpy()
    plt.imshow(rgb_image, cmap=None)
    plt.show()

def draw_test(image):
    plt.imshow(image.reshape(512,512,3))
    plt.show()

def get_test_data():
    test_x = []
    test_y = []
    for i in range(400,400+cfg.test_size):
        x_path = './test_A/'+str(i)+'.png'
        y_path = './test_B/'+str(i)+'.png'
        image_x = Image.open(x_path)
        test_x.append(np.array(image_x))
        image_y = Image.open(y_path)
        test_y.append(np.array(image_y))
    return np.array(test_x),np.array(test_y)

x_test,y_test = get_test_data()
x_test = x_test.reshape(cfg.test_size,cfg.channel,cfg.image_width,cfg.image_height)
y_test = y_test.reshape(cfg.test_size,cfg.channel,cfg.image_width,cfg.image_height)

# param_dict = load_checkpoint("./model/Generator1.0.ckpt")
gen = Generator()
# load_param_into_net(gen, param_dict)
y = gen(Tensor.from_numpy(x_test).astype(mindspore.float32))
y = (y+1)*127.5

for image in y:
    draw(image)