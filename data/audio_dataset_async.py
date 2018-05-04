import os.path
import random
import torch

from tqdm import tqdm
from collections import defaultdict
import librosa
import numpy as np
from skimage import io
from skimage.transform import resize
import multiprocessing
import h5py

import soundfile as sf
from data.audio_folder import make_dataset
from data.base_dataset import BaseDataset
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
import pdb
import sys

class fdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
def writeToDisk(args):
    hf = h5py.File(args[-1], 'a')
    fn = args[-2].split('/')[-1]
    hf.create_dataset('{}_data'.format(fn), data=args[0], compression="gzip", compression_opts=9)
    hf.create_dataset('{}_label'.format(fn), data=args[1], compression="gzip", compression_opts=9)
    hf.close()


class AudioDataset(BaseDataset):
    def initialize(self, opt, argsDict):
        self.opt = opt
        for k,v in argsDict.items():
            setattr(self, k, v)

        labelStr = '_0123456789'
        self.labels_map = dict(zip(labelStr, range(len(labelStr))))
        self.dumpPath = opt.dumpPath + '_' + self.mode + '.h5'
        self.loadData()

    def __getitem__(self, index):
        sampleInfo = self.fileInfo[index] 
        data, label, imagePath = self.loadSample(sampleInfo)
        assert data.shape[1] == 50
        assert data.shape[2] == 100
        return {
            ## channels x time x h x w
            'data': data.transpose((3, 0, 1, 2)) / 255.,
            'label': label,
            'path': imagePath
        }

    def __len__(self):
        # return len(self.FilesClean)
        # return 64
        return len(self.fileInfo)


    def addnoise(self, clean, noise):
        # print(clean.dtype, noise.dtype)
        assert clean.shape == noise.shape
        noiseAmp = np.mean(np.square(clean)) / np.power(10, self.snr / 10.0)
        scale = np.sqrt(noiseAmp / np.clip(np.mean(np.square(noise)), a_min=1e-7, a_max=1e8))
        return clean + scale * noise

    def name(self):
        return "AudioDataset"
    
    def loadData(self):
        ROOT = self.rootPath

        labelDict = dict()
        for entry in open(os.path.join(ROOT, self.labelFn)).readlines():
            idd, label = entry.split('\t')
            labelDict.update({idd: label})


        fileInfo = list()
        mouthDir = self.subDir
        for sampleFn in os.listdir(os.path.join(ROOT, mouthDir)):
            imagePath = os.path.join(ROOT, mouthDir, sampleFn)
            label = labelDict.get(sampleFn.split('-')[0])
            if label is None:
                continue
            if 'N' in label:
                continue
            label = ''.join(filter(lambda c:c.isdigit(), label))
            if len(label) != 4:
                continue
            if label is not None and len(os.listdir(imagePath)) < 149:
                fileInfo.append((imagePath, label))

        self.fileInfo = fileInfo[::]

    def loadSample(self, sampleInfo):
        assert len(sampleInfo) == 2
        imagePath, label = sampleInfo 
        return self.parseImage(imagePath), self.parseTranscript(label), imagePath
            

    def parseTranscript(self, label):
        #transcript = open(path, 'r').readlines()
        #transcript = ' '.join(list(map(lambda s:s.split(' ')[-1].strip(), transcript)))
        transcript = label
        transcript = list(filter(lambda x:x is not None, [self.labels_map.get(x, None) for x in transcript]))
        return transcript 
   
    #To Do normalization on whole dataset
    def parseImage(self, path):   
        ## return format: Time x Height x Width x Channel
        ## mount_xxx.png
        #return np.array([io.imread(os.path.join(path, fn)) for fn in list(os.listdir(path)).sort(key= lambda x:int(x[-7:-4]))])
        return np.array([resize(io.imread(os.path.join(path, fn), mode="constant"), (50, 100)) for fn in sorted(os.listdir(path), key=lambda x:int(x[-7:-4]))])


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)


class AudioDataLoader(DataLoader):
     def __init__(self, *args, **kwargs):
         """
         Creates a data loader for AudioDatasets.
         """
         super(AudioDataLoader, self).__init__(*args, **kwargs)
         self.collate_fn = _collate_fn


def _collate_fn(batch):
    def func(p):
        return p['data'].shape[1]
    #fdb().set_trace()
    longestSample = max(batch, key=func)['data']
    nChannel, nTimeSteps, height, width = longestSample.shape
    ## ugly workaround
    nTimeSteps = 150
    minibatchSize = len(batch)

    ### TODO:ARGIFY
    inputs = np.zeros((minibatchSize, nChannel, nTimeSteps, height, width), dtype=np.float32)
    input_percentages = torch.FloatTensor(minibatchSize)
    target_sizes = torch.IntTensor(minibatchSize)
    targets = []
    fnList = list()
    for x in range(minibatchSize):
        sample = batch[x]
        data = sample['data']
        target = sample['label']
        seqLength = data.shape[1]
        inputs[x][::, :seqLength, ::, ::] = data
        input_percentages[x] = seqLength / float(nTimeSteps)
        target_sizes[x] = len(target)
        targets.extend(target)
        fnList.append(sample['path'])
    return {
            'inputs': torch.from_numpy(inputs),
            'labels': torch.IntTensor(targets),
            'inputPercentages': input_percentages,
            'labelSize': target_sizes,
            'fileInfo': fnList
            } 

