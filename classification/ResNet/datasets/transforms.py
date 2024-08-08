import numpy as np
import random
from scipy import interpolate
import scipy.signal as signal
import matplotlib.pyplot as plt

# Data is expected to be in [channels, samples]
# Notes: some methods apply randomly to channels and some same for all channels
# randomization ranges not carefully checked

class Compose(object):
    def __init__(self, transforms, p = 0.5):
        self.transforms = transforms
        self.all_p = p
        
    def __call__(self, mseq):
        if self.all_p < np.random.rand(1):
            return mseq
        for t in self.transforms:
            mseq = t(mseq)
        return mseq

    
class Retype(object):
    def __call__(self, mseq):
        return mseq.astype(np.float32)
    

class Resample(object):
    def __init__(self, fs_new, fs_old):
        self.fs_new = fs_new
        self.fs_old = fs_old

    def __call__(self, mseq):
        num = int(mseq.shape[1]*self.fs_new/self.fs_old)
        mseq_rs = np.zeros([mseq.shape[0], num])
        for i, row in enumerate(mseq):            
            mseq_rs[i,:] = signal.resample(row, num)
        return mseq_rs

    
class SplineInterpolation(object):
    def __init__(self, fs_new, fs_old):
        self.fs_new = fs_new
        self.fs_old = fs_old

    def spliner(self, ind_orig, val_orig, ind_new):
        spline_fn = interpolate.interp1d(ind_orig, val_orig, kind='cubic')
        return spline_fn(ind_new)

    def __call__(self, mseq):
        n_old = mseq.shape[1]
        n_new = int(n_old*self.fs_new/self.fs_old)
        
        T = n_old/self.fs_old
        ind_orig = np.linspace(0,T,n_old)
        ind_new = np.linspace(ind_orig[0],ind_orig[-1],n_new)
        
        mseq_rs = np.zeros([mseq.shape[0], n_new])
        for i, row in enumerate(mseq):
            mseq_rs[i,:] = self.spliner(ind_orig, row, ind_new)
        return mseq_rs

       
class BandPassFilter(object):
    def __init__(self, fs, lf=0.5, hf=50, order=2):
        self.fs = fs
        self.lf = lf
        self.hf = hf
        self.order = order
        
    def bpf(self, arr, fs, lf=0.5, hf=50, order=2):
        wbut = [2*lf/fs, 2*hf/fs]
        sos = signal.butter(order, wbut, btype = 'bandpass', output = 'sos')       
        return signal.sosfiltfilt(sos, arr, padlen=250, padtype='even') 
                
    def __call__(self, mseq):
        for i, row in enumerate(mseq):
            mseq[i,:] = self.bpf(row, self.fs, self.lf, self.hf, self.order)
        return mseq    
    
    
class Normalize(object):
    def __init__(self, type="0-1"):
        self.type = type

    def __call__(self, mseq):
        if self.type == "0-1":
            for i, row in enumerate(mseq):
                if sum(mseq[i, :])  == 0:
                    mseq[i, :] = mseq[i, :]
                else:
                    mseq[i,:] = (row - np.min(row)) / (np.max(row) - np.min(row))             
        elif self.type == "mean-std":
            for i, row in enumerate(mseq):
                mseq[i,:] = (row - np.mean(row)) / np.std(row)
        elif self.type == "none":
            mseq = mseq
        else:
            raise NameError('This normalization is not included!')
        return mseq


class AddNoise(object):
    
    def __init__(self, sigma=0.05, p = 0.5):
        self.sigma = sigma
        self.addnoise_p = p
        
    def __call__(self, mseq):
        if self.addnoise_p < np.random.rand(1):
            return mseq
        sigma = np.random.uniform(0,self.sigma)        
        mseq = mseq + np.random.normal(loc=0, scale=sigma, size=mseq.shape)
        return mseq

    
class Roll(object):
    
    def __init__(self, n = 250, p=0.5):
        self.n = n
        self.roll_p = p   
        
    def __call__(self, mseq):
        if self.roll_p < np.random.rand(1):
            return mseq
        sign = np.random.choice([-1,1])
        n = np.random.randint(0, self.n)
        for i, row in enumerate(mseq):
            mseq[i,:] = np.roll(row, sign*n)
        return mseq

    
class Flipy(object):
    
    def __init__(self, p = 0.5):
        self.flipy_p = p
        
    def __call__(self, mseq):
        if self.flipy_p < np.random.rand(1):
            return mseq
        for i, row in enumerate(mseq):
            mseq[i,:] = np.multiply(row,-1)
        return mseq


class Flipx(object):
    
    def __init__(self, p = 0.5):
        self.flipx_p = p
        
    def __call__(self, mseq):
        if self.flipx_p < np.random.rand(1):
            return mseq
        return np.fliplr(mseq)

    
class MultiplySine(object):
    
    def __init__(self, fs = 250, f = 2, a = 1, p = 0.5):
        self.fs = 250
        self.f = f
        self.a = a
        self.multiply_sine_p = p
                
    def __call__(self, mseq):
        if self.multiply_sine_p < np.random.rand(1):
            return mseq
        t = np.arange(mseq.shape[1])/self.fs          
        for i, row in enumerate(mseq):
            f, a = np.random.uniform(0,self.f), np.random.uniform(0,self.a)
            mseq[i,:] = row*(1 + a*np.sin(2*np.pi*f*t))     
        return mseq

    
class MultiplyLinear(object):
    
    def __init__ (self, multiplier = 5, p = 0.5):
        self.multiply_linear_p = p
        self.multiplier = multiplier
        
    def __call__(self, mseq):
        if self.multiply_linear_p < np.random.rand(1):
            return mseq
        n = mseq.shape[1]
        for i, row in enumerate(mseq):            
            m = np.random.uniform(1,self.multiplier,2)        
            v = np.linspace(m[0],m[1],n)
            mseq[i,:] = np.multiply(row, v) 
        return mseq

    
class MultiplyTriangle(object):
    
    def __init__(self, scale = 2.0, p = 0.5):
        self.multiply_triangle_p = p
        self.scale = scale
        
    def __call__(self, mseq):
        if self.multiply_triangle_p < np.random.rand(1):
            return mseq
        n_samples = mseq.shape[1]
        for i, row in enumerate(mseq):
            n_turning_point = int(np.random.uniform(0,1)*n_samples)
            m = np.random.uniform(1/self.scale, self.scale)   
            v1 = np.linspace(1,m,n_turning_point)
            v2 = np.linspace(m,1,n_samples - n_turning_point)
            v = np.concatenate([v1,v2])
            mseq[i,:] = np.multiply(row, v) 
        return mseq

    
class RandomClip(object):
    def __init__(self, w=1000):
        self.w = w

    def __call__(self, mseq):
        if mseq.shape[1] >= self.w:
            start = random.randint(0, mseq.shape[1] - self.w)
            mseq = mseq[:, start:start + self.w]
        else:
            left = random.randint(0, self.w - mseq.shape[1])
            right = self.w - mseq.shape[1] - left
            zeros_padding1 = np.zeros(shape=(mseq.shape[0], left))
            zeros_padding2 = np.zeros(shape=(mseq.shape[0], right))
            mseq = np.hstack((zeros_padding1, mseq, zeros_padding2))
        return mseq

    
class RandomStretch(object):
    def __init__(self, scale=1.5, p = 0.5):
        self.scale = scale
        self.p = p

    def __call__(self, mseq):
        if self.p < np.random.rand(1):
            return mseq
        m = np.random.uniform(1/self.scale, self.scale)
        num = int(mseq.shape[1]*m)
        for i, row in enumerate(mseq):
            y = signal.resample(row, num)
            if len(y) < len(row):
                mseq[i,:len(y)] = y
            else:
                mseq[i,:] = y[:len(row)]
            return mseq

        
class ResampleSine(object):
    def __init__(self, fs = 250, freq_lo = 0.0, freq_hi = 0.3, 
                 scale_lo = 0.0, scale_hi = 0.5, p=0.5):
        self.fs = fs
        self.freq_lo = freq_lo
        self.freq_hi = freq_hi
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi
        self.p = p
        
    def __call__(self, mseq):
        if self.p < np.random.rand(1):
            return mseq        
        scale = np.random.uniform(self.scale_lo, self.scale_hi) 
        freq = np.random.uniform(self.freq_lo, self.freq_hi)
        x_orig = np.arange(0, mseq.shape[1])/self.fs
        x_new = x_orig + scale*np.sin(2*np.pi*freq*x_orig)
        for i, row in enumerate(mseq):            
            mseq[i,:] = np.interp(x_new, x_orig, row)
        return mseq

    
class ResampleLinear(object):
    def __init__(self, scale = 2, p=0.5):
        self.scale = scale
        self.p = p
        
    def __call__(self, mseq):
        if self.p < np.random.rand(1):
            return mseq    
        x_orig = np.arange(0, mseq.shape[1])
        scale = np.random.uniform(self.scale, self.scale) 
        scale = np.linspace(1,scale,len(x_orig))
        x_new = x_orig*scale
        x_new = x_new*(x_orig[-1]/x_new[-1])
        for i, row in enumerate(mseq):            
            mseq[i,:] = np.interp(x_new, x_orig, row)
        return mseq

    
class NotchFilter(object):
    def __init__(self, fs, Q = 1, p = 0.5):
        self.fs = fs
        self.Q = Q
        self.p = p
        
    def nf(self, arr, fs, f0, Q):
        b, a = signal.iirnotch(f0, Q, fs)    
        return signal.filtfilt(b, a, arr) 
                
    def __call__(self, mseq):
        if self.p < np.random.rand(1):
            return mseq  
        f0 = np.random.uniform(1, int(self.fs/2)) 
        for i, row in enumerate(mseq):
            mseq[i,:] = self.nf(row, self.fs, f0, self.Q)
        return mseq  

    
class ValClip(object):
    def __init__(self, w=72000):
        self.w = w

    def __call__(self, seq):
        if seq.shape[1] >= self.w:
            seq = seq
        else:
            zeros_padding = np.zeros(shape=(seq.shape[0], self.w - seq.shape[1]), dtype=np.float32)
            seq = np.hstack((seq, zeros_padding))
        return seq

'''
clips ECG channels so that 2.5 seconds of lead each lead is available in the following order:
    [[Lead I, Lead II, Lead III],[aVL, aVR, aVF],[Lead V1, Lead V2, Lead V3],[Lead V4, Lead V5, Lead V6]]'''
class DigitizationClip(object):
    # TODO: check - ideally w = 5000, but not sure if using something that isn't 2^n will cause problems for resnet
    def __init__(self, w=4096, fs = 500): 
        self.w = w
        self.fs = fs

    def __call__(self, mseq):

        if mseq.shape[1] >= self.w:
            
            start = random.randint(0, mseq.shape[1] - self.w)
            mseq = mseq[:, start:start + self.w]
        else: # if total ECG length is too short
            left = random.randint(0, self.w - mseq.shape[1])
            right = self.w - mseq.shape[1] - left
            zeros_padding1 = np.zeros(shape=(mseq.shape[0], left))
            zeros_padding2 = np.zeros(shape=(mseq.shape[0], right))
            mseq = np.hstack((zeros_padding1, mseq, zeros_padding2))

            
        # zero padding - mseq should already be a 12 x w tensor
        # TODO: if there are multiple rhythm strips from digitization - throw away 2nd rhythm strip so it doesn't confuse the resnet
        # TODO: train second model for cabrera layout training option.
        # TODO: add switch between cabrera and normal layout models during model_run
        
        #hard code these all for the moment
        w_col = int(np.round(2.5 * self.fs)) 
        
        # column 1
        right_padding = np.zeros(int(self.w - w_col))
        mseq[0] = np.hstack((mseq[0][0:w_col], right_padding)) 
        mseq[2] = np.hstack((mseq[2][0:w_col], right_padding))
        
        # column 2
        left_padding = np.zeros(w_col)
        right_padding = np.zeros(self.w - 2*w_col)
        mseq[3] = np.hstack((left_padding, mseq[3][(w_col):(2*w_col)], right_padding))
        mseq[4] = np.hstack((left_padding, mseq[4][(w_col):(2*w_col)], right_padding))
        mseq[5] = np.hstack((left_padding, mseq[5][(w_col):(2*w_col)], right_padding))
        
        # column 3
        left_padding = np.zeros(w_col*2)
        right_padding = np.zeros(self.w - 3*w_col)
        mseq[6] = np.hstack((left_padding, mseq[6][(2*w_col):(3*w_col)], right_padding))
        mseq[7] = np.hstack((left_padding, mseq[7][(2*w_col):(3*w_col)], right_padding))
        mseq[8] = np.hstack((left_padding, mseq[8][(2*w_col):(3*w_col)], right_padding))
        
        # column 4
        left_padding = np.zeros(w_col*3)
        right_padding = np.zeros(self.w - 4*w_col)
        mseq[9] = np.hstack((left_padding, mseq[9][(3*w_col):(4*w_col)], right_padding))
        mseq[10] = np.hstack((left_padding, mseq[10][(3*w_col):(4*w_col)], right_padding))
        mseq[11] = np.hstack((left_padding, mseq[11][(3*w_col):(4*w_col)], right_padding))
        
        # assume that rhythm strip is lead 2 - no changes
        mseq[1] = np.hstack((mseq[1][0:(4*w_col)], right_padding))
        
        return mseq
    
'''
clips ECG channels so that 2.5 seconds of lead each lead is available in the following order:
    [[Lead I, Lead II, Lead III],[aVL, aVR, aVF],[Lead V1, Lead V2, Lead V3],[Lead V4, Lead V5, Lead V6]]'''
class DigitizationClipAugment(object):
    # TODO: check - ideally w = 5000, but not sure if using something that isn't 2^n will cause problems for resnet
    def __init__(self, w=4096, fs = 500): 
        self.w = w
        self.fs = fs

    def __call__(self, mseq):

        if mseq.shape[1] >= self.w:
            start = random.randint(0, mseq.shape[1] - self.w)
            mseq = mseq[:, start:start + self.w]
        else: # if total ECG length is too short
            left = random.randint(0, self.w - mseq.shape[1])
            right = self.w - mseq.shape[1] - left
            zeros_padding1 = np.zeros(shape=(mseq.shape[0], left))
            zeros_padding2 = np.zeros(shape=(mseq.shape[0], right))
            mseq = np.hstack((zeros_padding1, mseq, zeros_padding2))

        #hard code these all for the moment
        w_col = int(np.round(2.5 * self.fs)) 
        
        # column 1
        right_padding = np.zeros(int(self.w - w_col))
        mseq[0] = np.hstack((mseq[0][0:w_col], right_padding)) 
        mseq[2] = np.hstack((mseq[2][0:w_col], right_padding))
        
        # column 2
        left_padding = np.zeros(w_col)
        right_padding = np.zeros(self.w - 2*w_col)
        mseq[3] = np.hstack((left_padding, mseq[3][(w_col):(2*w_col)], right_padding))
        mseq[4] = np.hstack((left_padding, mseq[4][(w_col):(2*w_col)], right_padding))
        mseq[5] = np.hstack((left_padding, mseq[5][(w_col):(2*w_col)], right_padding))
        
        # column 3
        left_padding = np.zeros(w_col*2)
        right_padding = np.zeros(self.w - 3*w_col)
        mseq[6] = np.hstack((left_padding, mseq[6][(2*w_col):(3*w_col)], right_padding))
        mseq[7] = np.hstack((left_padding, mseq[7][(2*w_col):(3*w_col)], right_padding))
        mseq[8] = np.hstack((left_padding, mseq[8][(2*w_col):(3*w_col)], right_padding))
        
        # column 4
        left_padding = np.zeros(w_col*3)
        right_padding = np.zeros(self.w - 4*w_col)
        mseq[9] = np.hstack((left_padding, mseq[9][(3*w_col):(4*w_col)], right_padding))
        mseq[10] = np.hstack((left_padding, mseq[10][(3*w_col):(4*w_col)], right_padding))
        mseq[11] = np.hstack((left_padding, mseq[11][(3*w_col):(4*w_col)], right_padding))
        
        # assume that rhythm strip is lead 2 - no changes
        mseq[1] = np.hstack((mseq[1][0:(4*w_col)], right_padding))
        
        idx = np.random.randint(0, 11)
        # TODO: THIS PROBABLY WON'T WORK - need to replace with a row of zeros
        mseq[idx, :] = 0
        
        # randomly remove another lead 5% of the time
        if np.random.uniform() < 0.05:
            idx = np.random.randint(0, 11)
            # TODO: THIS PROBABLY WON'T WORK - need to replace with a row of zeros
            mseq[idx, :] = 0
        
        return mseq