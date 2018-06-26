import multiprocessing
import os.path
import pdb
import random
import sys
from collections import defaultdict

import h5py
import ipdb
import librosa
import numpy as np
import soundfile
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
        self.labels_map['4'] = 2
        self.labels_map['7'] = 2
        self.labels_map['8'] = 3
        self.labels_map['9'] = 7
        self.dumpPath = opt.dumpPath + '_' + self.mode + '.h5'
        self.loadData()

    def __getitem__(self, index):
        sampleInfo = self.fileInfo[index] 
        lipData, speechData, label, imagePath = self.loadSample(sampleInfo)
        assert data.shape[1] == 50
        assert data.shape[2] == 100
        return {
            ## channels x time x h x w
            'lipData': data.transpose((3, 0, 1, 2)) / 255.,
            'speechData': speechData,
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
        labelDict = dict()
        for entry in open(self.labelFn).read().splitlines():
            try:
                idd, label = entry.split('\t')
            except:
                print(entry)
                continue
            labelDict.update({idd: label})


        fileInfo = list()
        for imagePath in open(self.manifestFn, 'r').read().splitlines():
            sampleFn = imagePath.split('/')[-1]
            label = labelDict.get(sampleFn.split('-')[0])
            if label is None or len(label) == 0:  
                continue
            if 'N' in label:
                continue
            label = ''.join(filter(lambda c:c.isdigit(), label))
            if len(label) != 4:
                continue
            if label is not None and len(os.listdir(imagePath)) < 149:
                fileID = imagePath.split('/')[-1]
                ## TODO:argify this argument
                audioPath = "/home/caspardu/data/LipReadProject/oriAudio/{}.ogg".format(fileID)
                fileInfo.append((imagePath, audioPath, label))

        self.fileInfo = fileInfo[::]

    def loadSample(self, sampleInfo):
        assert len(sampleInfo) == 3
        imagePath, audioPath, label = sampleInfo 
        return self.parseImage(imagePath), self.parseAudio(audioPath), self.parseTranscript(label), imagePath

            

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

    def _loadAudio(self, path):
        try:
            #sound, _ = librosa.load(path)
            sound, _ = soundfile.read(path)
        except:
            print(path)
        if len(sound.shape) > 1:
            assert False
            sound = sound.T
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
        return sound
 
        
    def parseAudio(self, path):
        y = self._loadAudio(path)
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=self.window)
        spect, phase = librosa.magphase(D)
        spect = np.log1p(spect)

        mean = spect.mean()
        std = spect.std()
        spect += -mean
        if (std - 0.0) > 1e-4:
            spect = np.divide(spect, std)
        if std == float(0):
            print(audio_path)
        assert not np.isnan(spect).any()
        assert not np.isinf(spect).any()

        return spect

    


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
    longestSample = max(batch, key=func)['data']
    nChannel, nTimeSteps, height, width = longestSample.shape
    ## ugly workaround
    #nTimeSteps = len(longestSample)
    minibatchSize = len(batch)

    ### TODO:ARGIFY
    inputs = np.zeros((minibatchSize, nChannel, nTimeSteps, height, width), dtype=np.float32)
    input_percentages = torch.FloatTensor(minibatchSize)
    target_sizes = torch.IntTensor(minibatchSize)
    targets = []
    fnList = list()


    #fdb().set_trace()
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
