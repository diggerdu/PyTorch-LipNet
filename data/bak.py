import multiprocessing
import os.path
import pdb
import random
import sys
from collections import defaultdict

import h5py
import librosa
import numpy as np
import soundfile as sf
import torch
from skimage import io
from skimage.transform import resize
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

from data.audio_folder import make_dataset
from data.base_dataset import BaseDataset


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

class AudioDataset(BaseDataset):
    def initialize(self, opt, argsDict):
        self.opt = opt
        for k,v in argsDict.items():
            setattr(self, k, v)

        labelStr = '_0123456789'
        self.labels_map = dict(zip(labelStr, range(len(labelStr))))
        self.preloadData = list()
        self.loadData(opt.dumpPath + '_' + self.mode + '.h5')

    def __getitem__(self, index):
        assert self.preloadData[index][0].shape[1] == 50
        assert self.preloadData[index][0].shape[2] == 100
        return {
            ## channels x time x h x w
            'data': self.preloadData[index][0].transpose((3, 0, 1, 2)) / 255.,
            'label': self.preloadData[index][1],
            'path': self.fileInfo[index]
        }

    def __len__(self):
        # return len(self.FilesClean)
        # return 64
        return len(self.preloadData)


    def addnoise(self, clean, noise):
        # print(clean.dtype, noise.dtype)
        assert clean.shape == noise.shape
        noiseAmp = np.mean(np.square(clean)) / np.power(10, self.snr / 10.0)
        scale = np.sqrt(noiseAmp / np.clip(np.mean(np.square(noise)), a_min=1e-7, a_max=1e8))
        return clean + scale * noise

    def name(self):
        return "AudioDataset"
    
    def loadData(self, dumpPath):
        #ROOT = "/home/dxj/corpus/GRID"
        #ROOT = "/data2/dxj/smallSet"
        if os.path.isfile(dumpPath): 
            hf = h5py.File(dumpPath, 'r')
            self.fileInfo = eval(hf.get('fileList').value)
            for sample in tqdm(self.fileInfo):
                fn = sample[0].split('/')[-1] 
                self.preloadData.append((hf.get('{}_data'.format(fn)).value, hf.get('{}_label'.format(fn)).value.tolist()))
            assert len(self.fileInfo) == len(self.preloadData) 
            hf.close()
        else:
            try:
                os.remove(dumpPath)
            except:
                pass

            ROOT = self.rootPath

            hf = h5py.File(dumpPath, 'w')
            labelDict = dict()
            for entry in open(os.path.join(ROOT, self.labelFn)).readlines():
                idd, label = entry.split('\t')
                labelDict.update({idd: label})


            fileInfo = list()

            mouthDir = self.subDir
            for sampleFn in os.listdir(os.path.join(ROOT, mouthDir)):
                imagePath = os.path.join(ROOT, mouthDir, sampleFn)
                #transPath = os.path.join(ROOT, 'align', sampleFn + '.align')
                label = labelDict.get(sampleFn.split('-')[0])
                if label is not None and len(os.listdir(imagePath)) < 149:
                    fileInfo.append((imagePath, label))

            self.fileInfo = fileInfo[::]
            hf.create_dataset('fileList', data=repr(self.fileInfo))   
            pool = multiprocessing.Pool(processes=128)
            self.preloadData = list(tqdm(pool.imap(self.loadSample, self.fileInfo), total=len(self.fileInfo)))
            pool.close()
            pool.join()

            #self.preloadData = list(map(self.loadSample, self.fileInfo))
            print('Dumping') 
            ## Dump Data
            assert len(self.preloadData) == len(self.fileInfo)
            for i in range(len(self.preloadData)):
                fn = self.fileInfo[i][0].split('/')[-1]
                hf.create_dataset('{}_data'.format(fn), data=self.preloadData[i][0])
                hf.create_dataset('{}_label'.format(fn), data=self.preloadData[i][1])
            hf.close()

    def loadSample(self, sampleInfo):
        assert len(sampleInfo) == 2
        imagePath, transPath = sampleInfo 
        return self.parseImage(imagePath), self.parseTranscript(transPath)
            

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
