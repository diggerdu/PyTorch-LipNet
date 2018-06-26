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
data_loader = CreateDataLoader(opt, {'mode':'Train', 'manifestFn':'/home/caspardu/data/LipReadProject/LipNetData/manifestFiles/wellDone_train.list','labelFn':'/home/caspardu/data/LipReadProject/LipNetData/manifestFiles/label.txt'})
testDataLoader = CreateDataLoader(opt, {'mode':'Test', 'manifestFn':'/home/caspardu/data/LipReadProject/LipNetData/manifestFiles/wellDone_test.list','labelFn':'/home/caspardu/data/LipReadProject/LipNetData/manifestFiles/label.txt'})

dataset = data_loader.load_data()
testDataset = testDataLoader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
print('#testing images = %d' % len(testDataLoader))

model = create_model(opt)
#summary(model.netG, input_size=(3,150, 100, 50))


initEpoch = model.epoch + 1
for epoch in range(initEpoch, opt.niter):
    lossList = []
    for i, data in tqdm(enumerate(dataset), total=len(dataset)):
        __import__('ipdb').set_trace()
    #for i, data in enumerate(dataset):
        model.set_input(data)
        model.optimize_parameters()
        currentLoss = model.get_current_errors()['G_LOSS']
        if not np.isinf(currentLoss):
            lossList.append(currentLoss)
    loss = sum(lossList) / len(lossList)
    model.annealLR(loss)
    model.epoch += 1
    print('epoch {}, loss is {}'.format(epoch, loss))
    if epoch % 5 == 2 or True: 
        flag = False
        if loss < .5:
            flag = True
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
        print('epoch {}, wer is {}, cer is {}'.format(epoch, wer, cer))
        print('############################')
        model.save()
    if epoch % 9 == 6:
        flag = False
        if loss < .5:
            flag = True
        totalWer = list()
        totalCer = list()
        for i, data in tqdm(enumerate(dataset), total=len(dataset)):
            model.set_input(data)
            results = model.test()
            totalWer.append(results['wer'])
            totalCer.append(results['cer'])
        
        cer = sum(totalCer) * 100 / len(totalCer)
        wer = sum(totalWer) * 100 / len(totalWer)
        print('############################')
        print('#####TrainingSet epoch {}, wer is {}, cer is {}'.format(epoch, wer, cer))
        print('############################')
        model.save()
