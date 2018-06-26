import sys
import time

import numpy as np
import torch.backends.cudnn as cudnn

from data.data_loader import CreateDataLoader
from models.models import create_model
from options.train_options import TrainOptions

#from util.visualizer import Visualizer


cudnn.benchmark = True

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
# visualizer = Visualizer(opt)

total_steps = 0

embark_time = time.time()
for epoch in range(1, opt.niter + 1):
    epoch_start_time = time.time()
    errorEpoch = list()
    for i, data in tqdm(enumerate(dataset)):
        total_steps += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()
        error = model.get_current_errors()['G_LOSS']
        errorEpoch.append(error)
    t = time.time() - epoch_start_time
    print('epoch ', epoch, ', current error is ', np.mean(errorEpoch), ' cost time is ', t)
    if time.time() - embark_time > 10:
        model.test()
        model.save('latest')
        embark_time = time.time()
