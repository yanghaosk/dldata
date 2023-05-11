from easydict import EasyDict as edict

cfg = edict({
    'train_size': 400,  # 训练集大小
    'test_size': 20,  # 测试集大小
    'channel': 3,  # 图片通道数
    'image_height': 512,  # 图片高度
    'image_width': 512,  # 图片宽度
    'epoch_size': 20,  # 训练次数
    'epoch_num': 10,
    'learn_rate': 1e-7,
    'num_filters': 3
})