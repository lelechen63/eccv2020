from options.train_options import TrainOptions
from dataset import LRSLmark2rgbDataset
opt = TrainOptions().parse()
dataset = LRSLmark2rgbDataset(opt)
sample = dataset[0]
print (sample)