import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules import Module
import os
import numpy as np
from collections import OrderedDict
from . import densenet_efficient as dens
from . import time_frequence as tf

###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data)
    elif classname.find('GRU') != -1:
        for weight in m.parameters():
            nn.init.orthogonal(weight)
    elif classname.find('Linear') != -1:
        for weight in m.parameters():
            nn.init.kaiming_normal(weight)

    elif classname.find('BatchNorm2d') != -1 or classname.find(
            'InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    else:
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer
# return None


def define_G(gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert (torch.cuda.is_available())
    netG = torch.nn.DataParallel(AuFCN())
    #netG = AuFCN()
    #netG = AuFCNWrapper(gpu_ids=gpu_ids)
    if len(gpu_ids) > 0:
        #netG.cuda(device=gpu_ids[0])
        netG.cuda(device=gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc,
        ndf,
        which_model_netD,
        n_layers_D=3,
        norm='batch',
        use_sigmoid=False,
        gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(
                input_nc,
                ndf,
                n_layers=3,
                norm_layer=norm_layer,
                use_sigmoid=use_sigmoid,
                gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(
                input_nc,
                ndf,
                n_layers_D,
                norm_layer=norm_layer,
                use_sigmoid=use_sigmoid,
                gpu_ids=gpu_ids)
    else:
        print('Discriminator model name [%s] is not recognized' %
                which_model_netD)
        if use_gpu:
            netD.cuda(device_id=gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################

class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x.contiguous()
        x = x.view(t * n, -1)
        try:
            x = self.module(x)
        except:
            __import__('ipdb').set_trace()
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.GRU, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                bidirectional=bidirectional, bias=False)
        self.num_directions = 2 if bidirectional else 1
        #self.rnn.flatten_parameters()

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        self.rnn.flatten_parameters()
        x, hidden = self.rnn(x)
        hidden.detach()
        #if self.bidirectional:
        #    x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x

class AuFCNWrapper(nn.Module):
    def __init__(self, numClasses=11, gpu_ids=[0,1]):
        super(AuFCNWrapper, self).__init__()
        self.gpu_ids = gpu_ids
        self.model = AuFCN(numClasses)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            #output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            return output
        else:
            return self.model(input)

try:
    exec(open(os.path.join(os.getenv("expPath"), 'networks_AuFCN.py')).read())
except:
    __import__('ipdb').set_trace()
    class AuFCN(nn.Module):
        def __init__(self, numClasses=11):
            super(AuFCN, self).__init__()
            modList = list()
            modList = [
                    nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(2, 2, 2), padding=(1, 2, 2)),
                    nn.BatchNorm3d(32),
                    nn.ReLU(inplace=True),
                    #nn.Dropout3d(),
                    nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2)),
                    nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
                    nn.BatchNorm3d(64),
                    nn.ReLU(inplace=True),
                    #nn.Dropout3d(),
                    nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2)),
                    nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                    nn.BatchNorm3d(96),
                    nn.ReLU(inplace=True),
                    #nn.Dropout3d(),
                    nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2))
                    ]
            #self.conv = nn.ModuleList(modList)
            self.conv = nn.Sequential(*modList)

            ## TODO ADD OPTIONS

            rnnList = [
                    BatchRNN(input_size=1728, hidden_size=256, bidirectional=True, batch_norm=True),
                    BatchRNN(input_size=512, hidden_size=256, bidirectional=True, batch_norm=True),
                    ]
            self.rnn = nn.Sequential(*rnnList)
            fcBlock = nn.Sequential(
                    #nn.BatchNorm1d(512),
                    nn.Dropout(),
                    nn.Linear(512, numClasses, bias=False)
                    )
            self.dense = nn.Sequential(
                    SequenceWise(fcBlock)
                    )
            self.inference_softmax = InferenceBatchSoftmax()



        def forward(self, sample):
            ## sample shape Batch x Channel x Time x H x w
            ## output of conv shape: Batch x Channel x time x H x w
            output = self.conv(sample)
            ## flatten: batch x time x 1728
            output = output.view(output.size(0), output.size(2), -1)

            ## transpose time x batch x 1728
            output = output.transpose(1, 0)
            try:
                assert output.size(-1) == 1728
            except:
                __import__('ipdb')
            output = self.rnn(output)
            ## output of rnn time x batch x 512
            output = self.dense(output)
            ## output of dense time x batch x nclasses
            output = output.transpose(1, 0)
            ## output batch x time x nclasses
            output = self.inference_softmax(output)

            return output


class Tanh_rescale(Module):
    def forward(self, input):
        return torch.div(
                torch.add(torch.tanh(torch.mul(input, 2.0)), 1.0), 2.0)

        def __repr__(self):
            return self.__class__.__name__ + ' ()'
