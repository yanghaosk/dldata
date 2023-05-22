from net import Generator
import numpy as np
import mindspore
from mindspore import Tensor

gen = Generator()
x = np.ones((1,3,512,512))
y = gen.construct(Tensor.from_numpy(x).astype(mindspore.float32))
print(y)