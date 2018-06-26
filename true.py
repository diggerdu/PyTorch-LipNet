import sys
import time

import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from data.data_loader import CreateDataLoader
from models.models import create_model
from options.train_options import TrainOptions

cudnn.benchmark = True

opt = TrainOptions().parse()
ROOT = '/data1'
testDataLoader = CreateDataLoader(opt, {'mode':'Test', 'labelFn':'in5008.txt', 'rootPath':ROOT, 'subDir':'eval'})
#testDataLoader = CreateDataLoader(opt, {'mode':'Train', 'labelFn':'in5008.txt', 'rootPath':ROOT, 'subDir':'train'})
 

testDataset = testDataLoader.load_data()
print('#testing images = %d' % len(testDataLoader))

model = create_model(opt)

totalWer = list()
totalCer = list()
for i, data in tqdm(enumerate(testDataset), total=len(testDataset)):
    model.set_input(data)
    results = model.test()
    totalWer.append(results['wer'])
    totalCer.append(results['cer'])

cer = sum(totalCer) * 100 / len(totalCer)
wer = sum(totalWer) * 100 / len(totalWer)
print('############################')
print('epoch {}, wer is {}, cer is {}'.format(model.epoch, wer, cer))
print('############################')
