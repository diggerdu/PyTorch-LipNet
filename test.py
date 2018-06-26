import sys
import time

import numpy as np
import torch.backends.cudnn as cudnn
from torchsummary import summary
from tqdm import tqdm

from data.data_loader import CreateDataLoader
from models.models import create_model
from options.train_options import TrainOptions

cudnn.benchmark = True

opt = TrainOptions().parse()
testDataLoader = CreateDataLoader(opt, {'mode':'Test', 'manifestFn':'/home/caspardu/data/LipReadProject/LipNetData/manifestFiles/wellDone_test.list','labelFn':'/home/caspardu/data/LipReadProject/LipNetData/manifestFiles/label.txt'})

testDataset = testDataLoader.load_data()
print('#testing images = %d' % len(testDataLoader))

model = create_model(opt)
summary(model.netG, input_size=(3,150, 100, 50))


for i, data in tqdm(enumerate(testDataset), total=len(testDataset)):
    model.set_input(data)
    model.visSaliency()
    __import__('ipdb').set_trace()
