import torch.utils.data

from data.audio_dataset import AudioDataLoader, BucketingSampler
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt, argsDict):
    from data.audio_dataset import AudioDataset
    dataset = AudioDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt, argsDict)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt=None, argsDict=dict()):
        BaseDataLoader.initialize(self, opt)

        self.dataset = CreateDataset(opt, argsDict)
        self.mode = argsDict['mode']
        if argsDict['mode'] == 'Train':
            train_sampler = BucketingSampler(self.dataset, batch_size=opt.batchSize)
            self.dataloader = AudioDataLoader(
                self.dataset,
                num_workers=int(opt.nThreads),
                batch_sampler=train_sampler
                )
        elif argsDict['mode'] == 'Test':
            self.dataloader = AudioDataLoader(self.dataset, batch_size=2 * opt.batchSize, num_workers=int(opt.nThreads))
        else:
            raise 'Undefined Mode Error'

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
