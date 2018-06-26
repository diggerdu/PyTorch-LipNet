import os
import time
from pdb import set_trace as st

import librosa
# audio process
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.autograd import Variable

from data.data_loader import CreateDataLoader
from models.models import create_model
from options.test_options import TestOptions
from util import html
from util.audio_process import recovery_phase
from util.visualizer import Visualizer


def CalSNR(ref, sig):
    ref_p = np.mean(np.square(ref))
    noi_p = np.mean(np.square(sig - ref))
    return 10 * (np.log10(ref_p) - np.log10(noi_p))



def loadAudio(path, SR):
    data, sr = sf.read(path)
    print(path, data.shape, sr)
    try:
        assert sr == SR
    except AssertionError:
        data = librosa.resample(data, sr, opt.SR)

    data = data / np.max(np.abs(data))
    return data - np.mean(data)

def eval(model, cleanPath, noisePath, opt):
    leng = opt.nfft + (opt.nFrames - 1) * opt.hop
    #clean = loadAudio(cleanPath, opt.SR)[::,0][212442:]
    clean = loadAudio(cleanPath, opt.SR)
    noise = loadAudio(noisePath, opt.SR)
    assert clean.shape[0] > leng

    noise = np.tile(noise, clean.shape[0] // noise.shape[0] + 1)[:clean.shape[0]]
    noiseAmp = np.mean(np.square(clean)) / np.power(10, opt.snr / 10.0)
    scale = np.sqrt(noiseAmp / np.mean(np.square(noise)))
    mix = clean + scale * noise
    print(opt.nfft, opt.nFrames, opt.hop)

    input_ = {'A' : torch.from_numpy(mix[:leng][np.newaxis, :]).float()}
    model.set_input(input_)
    model.test()
    output = model.get_current_visuals().data.cpu().numpy().flatten()
    target = clean[:leng]

    output = output[1024:-1024]
    target = target[1024:-1024]
    mix = mix[1024:-1024]
    print(CalSNR(target, output), 'dB')
    sf.write('clean.wav', target, opt.SR)
    sf.write('enhance.wav', output, opt.SR)
    sf.write('mix.wav', mix, opt.SR)


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.model = "test"

model = create_model(opt)

# test

#cleanPath = "/home/diggerdu/dataset/Large/clean/p240/p240_013.wav"
#cleanPath = "/home/diggerdu/dataset/chineseMix/li2.wav"
cleanPath = "/mnt/dataset/audio/ZTE/CHANNEL0/WAVE/SPEAKER0001/SESSION0/000010187.WAV"
#cleanPath = "/home/diggerdu/manifestdestiny/Cranberry/checkpoints/chinese-0db/enhance.wav"
noisePath = "/home/diggerdu/dataset/Large/babble/59899-robinhood76-00309crowd2-116.wav"
#noisePath = "/home/diggerdu/dataset/speech/14.wav"
#cleanPath = "/home/diggerdu/dataset/speech/14.wav"

eval(model, cleanPath, noisePath, opt)
