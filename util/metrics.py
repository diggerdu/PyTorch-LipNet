#!/usr/bin/env python
"""
 - Set of functions for converting between different speech quality estimation
   metrics such as PESQ MOS, MOS LQO, R-factor.
 - Python wrapper for ITU-T pesq utlity.
 - Helper class to define Speex codec parameters based on other options:
    - mapping between speex "quality" and "mode" option
    - size (in bits) for earch speex frame with given mode
    - required bandwidth estimation
"""
from __future__ import division
import sys, os
from math import sqrt, pi, atan2, log, pow, cos, log, exp

__all__ = 'SpeexMetric mos2r r2mos delay2id pesq2mos mos2pesq pesq'.split()

class SpeexMetric(object):
    """
    SpeexMetric class

    >>> m = SpeexMetric(quality=7)
    >>> m.mode
    5
    >>> m.size
    300

    >>> m = SpeexMetric(mode=5)
    >>> m.quality
    8
    >>> m.size
    300
    >>> m.get_bandwidth(1)
    31000
    >>> m.get_bandwidth(2)
    23000
    >>> m.get_bandwidth(3)
    20333
    """

    def __init__(self, quality=None, mode=None):
        if quality is None and mode is None:
            raise ValueError('Speex quality or mode must be set up')
        if quality is not None and mode is not None:
            raise ValueError('You must set up just one option: quality or mode')
        if quality:
            self.quality = quality
                       # 0  1  2  3  4  5  6  7  8  9  10
            self.mode = (1, 8, 2, 3, 3, 4, 4, 5, 5, 6, 7)[self.quality]
        else:
            self.mode = mode
            self.quality =  {
                1: 0, 8: 1, 2: 2, 3: 4, 4: 6, 5: 8, 6: 9, 7: 10,}[self.mode]
        self.size = {
                1: 43, 8: 79, 2: 119, 3: 160, 4: 220,
                5: 300, 6: 364, 7: 492, }[self.mode]

    def get_bandwidth(self, fpp=1):
        """
        Return bandwidth value (bits per second) required to transmit the
        speech encoded with given speex settings and given frames per packet.

        Assume that speech is transmitted over RTP/UDP/IP stack with 12+8+20=40
        bytes in the header.
        """
        ip_udp_rtp_hdr = (20 + 8 + 12) * 8
        size = self.size  * fpp + ip_udp_rtp_hdr
        # (50 packets with fpp=1)
        packets_per_second = 50.0 / fpp
        return int(packets_per_second * size)


def mos2r(mos):
    """ With given MOS LQO return R-factor  (1 < MOS < 4.5) """
    D = -903522 + 1113960 * mos - 202500 * mos * mos
    if D < 0:
        D = 0
    h = 1/3 * atan2(15*sqrt(D), 18556-6750*mos)
    R = 20/3 * (8 - sqrt(226) * cos(h+pi/3))
    return R > 100 and 100.0 or R


def r2mos(r):
    """ With given R-factor return MOS """
    if r < 0:
        return 1
    if r > 100:
        return 4.5
    return 1 + 0.035 * r  + r * (r - 60) * (100 - r) * 7e-6


def delay2id(Ta):
    """ Delay Ta (ms) render to Id penalty according to ITU-T G.107 and G.108
    recommendations. """
    if Ta < 100:
        Id = 0
    else:
        X = log(Ta/100) / log(2)
        Id = 25.0 * (
               pow((1 + pow(X, 6)), 1.0/6) - \
           3 * pow(1+ pow(X/3, 6), 1.0/6 ) + 2
        )
    return Id


def pesq2mos(pesq):
    """ Return MOS LQO value (within 1..4.5) on PESQ value (within -0.5..4.5).
    Mapping function given from P.862.1 (11/2003) """
    return 0.999 + (4.999-0.999) / (1+exp(-1.4945*pesq+4.6607))


def mos2pesq(mos):
    """ Return PESQ value (within -0.5..4.5) on MOS LQO value (within 1..4.5).
    Mapping function given from P.862.1 (11/2003) """
    inlog =(4.999-mos)/(mos-0.999)
    return (4.6607-log(inlog)) / 1.4945


def pesq(reference, degraded, sample_rate=None, program='pesq'):
    """ Return PESQ quality estimation (two values: PESQ MOS and MOS LQO) based
    on reference and degraded speech samples comparison.

    Sample rate must be 8000 or 16000 (or can be defined reading reference file
    header).

    PESQ utility must be installed.
    """
    if not os.path.isfile(reference) or not os.path.isfile(degraded):
        raise ValueError('reference or degraded file does not exist')
    if not sample_rate:
        import wave
        w = wave.open(reference, 'r')
        sample_rate = w.getframerate()
        w.close()
    if sample_rate not in (8000, 16000):
        raise ValueError('sample rate must be 8000 or 16000')
    import subprocess
    args = [ program, '+%d' % sample_rate, reference, degraded  ]
    pipe = subprocess.Popen(args, stdout=subprocess.PIPE)
    out = pipe.communicate()[0].decode("utf-8")
    last_line = out.split('\n')[-2]
    if not last_line.startswith('P.862 Prediction'):
        raise ValueError(last_line)
    return tuple(map(float, last_line.split()[-2:]))




if __name__ == '__main__':
    #import doctest
    #doctest.testmod(optionflags=doctest.ELLIPSIS)
    results = pesq("/home/diggerdu/manifestdestiny/Cranberry/checkpoints/babble-0db/clean.wav", "/home/diggerdu/manifestdestiny/Cranberry/checkpoints/babble-0db/enhance.wav", sample_rate=16000)
    print(results)

