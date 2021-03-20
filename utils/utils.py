import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter

from scipy.optimize import curve_fit as CF
from scipy.fft import fft, fftfreq,ifft


def deconv_f(dt,sig,imp,freq):
    sig1 = np.zeros([len(sig)+len(imp)-1])
    sig1[:len(sig)] = sig
    sigf = fft(sig1)
    imp1 = np.zeros_like(sig1)
    imp1[:len(imp)] = imp
    impf = fft(imp1)
    xf = fftfreq(len(sig1), dt)/1e6    
    f_ind = np.abs(xf)<freq
    decf = sigf.copy()
    decf[f_ind] = decf[f_ind]/impf[f_ind]
    dec = ifft(decf)
    return dec

def deconv(sig,imp):
    sig1 = np.zeros([len(sig)+len(imp)-1])
    sig1[:len(sig)] = sig
    sigf = fft(sig1)
    imp1 = np.zeros_like(sig1)
    imp1[:len(imp)] = imp
    impf = fft(imp1)
    decf = sigf/impf
    dec = ifft(decf)
    return dec

def line(x,a,b):
    return a*x+b

def T2SMQ(ts,smqs):
    a,b = CF(line,ts,smqs)
    return a,b

def SMQ2T(smqs,ts):
    a,b = CF(line,smqs,ts)
    
    plt.figure()
    plt.plot(smqs,ts,'o')
    xs = np.linspace(smqs[0],smqs[-1],100)
    plt.plot(xs,line(xs,*a),'r-.')   
    
    return a,b

def FitErrs(m,ts,qs0,num):
    
    perrs = []
    for i in range(num):
        qsi = qs0+i
        smq = np.sqrt(m/qsi)
        a,b=CF(line,ts,smq)
        perr = np.sum((line(ts,*a)-smq)**2)
        perrs.append(perr)
    return perrs

def rebin1dx(arr,factor):
    if factor == 1:
        return arr
    as1 = len(arr)
    s1 = as1 - (as1%factor) + factor    
    newarr = np.zeros((s1,))
#    arrX = arrX[:l//factor*factor]    
    newarr[:as1] = arr
    sh = int(s1/factor),factor
    return np.mean(newarr.reshape(sh),axis=-1)[:-1]
    
    
def rebin1d(arr,factor):
    if factor == 1:
        return arr    
    as1 = len(arr)
    s1 = as1 - (as1%factor) + factor    
    newarr = np.zeros((s1,))
#    arrX = arrX[:l//factor*factor]    
    newarr[:as1] = arr
    sh = int(s1/factor),factor
    return np.sum(newarr.reshape(sh),axis=-1)[:-1]

def rebin2d(a, factor):
    s0 = a.shape[0] - (a.shape[0]%factor[0]) + factor[0]
    s1 = a.shape[1] - (a.shape[1]%factor[1]) + factor[1]    
    sh = int(s0/factor[0]),factor[0],int(s1/factor[1]),factor[1]
    newa = np.zeros((s0,s1))
    newa[:a.shape[0],:a.shape[1]] = a
    return newa.reshape(sh).sum(-1).sum(1)
    
def rebin3d(a, factor):

    s0 = a.shape[0] - (a.shape[0]%factor[0]) + factor[0]
    s1 = a.shape[1] - (a.shape[1]%factor[1]) + factor[1]    
    s2 = a.shape[2] - (a.shape[2]%factor[2]) + factor[2]  
        
    sh = s0/factor[0],factor[0],s1/factor[1],factor[1],s2/factor[2],factor[2]
    newa = np.zeros((s0,s1,s2))
    newa[:a.shape[0],:a.shape[1],:a.shape[2]] = a
    return newa.reshape(sh).sum(-1).sum(-2).sum(1)    
    
