import gc
import os
from functools import reduce

import numpy as np
import soundfile as sf

import SharedArray as sa

stereoPath = '/seagate2t/rec/stereo/' 
monoPath = '/seagate2t/rec/mono/' 

def loadFile(path):
    y, sr = sf.read(path)
    assert sr == 16000
    return y

def listFpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def getData(path):
    return reduce(lambda x,y:np.concatenate((x, y)), map(loadFile, listFpath(path)))
    #return list(map(loadFile, listFpath(path)))

def prepareData(monoPath=monoPath, stereoPath=stereoPath):
    monoArray = getData(monoPath)
    np.save("mono.npy", monoArray)
    Length = monoArray.shape[0]
    #sharedMono = sa.create("shm://mono", monoArray.shape)
    print("shared Mono created")
    #sharedMono[::] = monoArray[::]
    del monoArray
    gc.collect()
    stereoArray = getData(stereoPath)
    assert stereoArray.shape[0] == Length
    np.save("stereo.npy", stereoArray)

    '''
    print(Length)
    try:
        sharedStereo = sa.create("shm://stereo", stereoArray.shape)
        print("shared Stereo created")
        sharedStereo[::] = stereoArray[::]
    except:
        pass
        #sa.delete("stereo")
        #sharedStereo = sa.create("shm://stereo", stereoArray.shape)
    '''


if __name__ == '__main__':
    prepareData()
