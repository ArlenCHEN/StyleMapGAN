from easydict import EasyDict

cfg = EasyDict()

cfg.TRAIN = EasyDict()
cfg.TRAIN.GRAY2RGB_USE_BATCHNORM = False
cfg.TRAIN.GRAY2RGB_LR = 1e-4
cfg.TRAIN.MSE_MARGIN_SCALE = 0.05