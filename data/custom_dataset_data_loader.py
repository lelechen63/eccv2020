import torch.utils.data
from data.base_data_loader import BaseDataLoader
import os

def CreateDataset(opt):
    dataset = None
    if opt.dataname == 'lrs':
        from data.dataset import LRSLmark2rgbDataset
        # opt.dataroot = os.path.join(opt.dataroot , 'lrs3/lrs3_v0.4')
        dataset = LRSLmark2rgbDataset(opt)
    elif opt.dataname == 'facefor':
        from data.dataset import FaceForensicsLmark2rgbDataset
        # opt.dataroot = os.path.join(opt.dataroot , 'lrs3/lrs3_v0.4')
        dataset = FaceForensicsLmark2rgbDataset(opt)
    

    print("dataset [%s] was created" % (dataset.name()))
    # dataset.__init__(opt)
    return dataset
def collate_fn(batch):
    batch = list(filter(lambda x : x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, pin_memory= True,
            batch_size=opt.batchSize,
            shuffle = not opt.serial_batches,
            num_workers=int(opt.nThreads),
             drop_last=True, collate_fn= collate_fn)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
