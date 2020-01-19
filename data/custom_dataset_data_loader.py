import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    if opt.use_lstm:
        from data.dataset import LRSLmark2rgbDataset
        dataset = LRSLmark2rgbDataset(opt)
    else:
        from data.dataset import LRSLmark2rgbDataset
        dataset = LRSLmark2rgbDataset(opt)
    

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
