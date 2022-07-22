from easydict import EasyDict

cfg = EasyDict()

cfg.TRAIN = EasyDict()
cfg.TRAIN.GRAY2RGB_USE_BATCHNORM = False
cfg.TRAIN.GLOBAL_AE_LR = 1e-4
# cfg.TRAIN.GLOBAL_AE_LR = 0.002
cfg.TRAIN.MSE_MARGIN_SCALE = 0.05

cfg.TEST = EasyDict()
