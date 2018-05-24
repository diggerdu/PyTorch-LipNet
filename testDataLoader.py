import time
import torch.backends.cudnn as cudnn
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import sys
import numpy as np
from tqdm import tqdm

cudnn.benchmark = True

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt, {'mode':'Train', 'manifestFn':'/home/caspardu/data/LipNetData/manifestFiles/wellDone_train.list','labelFn':'/home/caspardu/data/LipNetData/manifestFiles/label.txt'})
testDataLoader = CreateDataLoader(opt, {'mode':'Test', 'manifestFn':'/home/caspardu/data/LipNetData/manifestFiles/wellDone_test.list','labelFn':'/home/caspardu/data/LipNetData/manifestFiles/label.txt'})

dataset = data_loader.load_data()
testDataset = testDataLoader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
print('#testing images = %d' % len(testDataLoader))

model = create_model(opt)
from torchsummary import summary
summary(model.netG, input_size=(3,150, 100, 50))


initEpoch = model.epoch + 1
for epoch in range(initEpoch, opt.niter):
    lossList = []
    for i, data in tqdm(enumerate(dataset), total=len(dataset)):
    #for i, data in enumerate(dataset):
        model.set_input(data)
        model.optimize_parameters()
        lossList.append(model.get_current_errors()['G_LOSS'])
    loss = sum(lossList) / len(lossList)
    model.annealLR(loss)
    model.epoch += 1
    print('epoch {}, loss is {}'.format(epoch, loss))
    if epoch % 5 == 2:
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

    
