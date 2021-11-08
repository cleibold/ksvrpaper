import recsvr as rs
import matplotlib.pyplot as plt
import numpy as np
import gammatonebank as gtb

def kerfunc(x,S,c):
  
  r=(np.abs(x)/S);
  r=np.exp(-r)
  r=r*np.cos(2.*np.pi*x*c)
  
  return r

##
## turn on the radio
##
##http://ccmixter.org/files/texasradiofish/63300
##
## CC BY NC
##
eps=0.000
Noff=50000
N=1040001
import wave
obj = wave.open('texas.wav','rb')
fs=obj.getframerate()
raw=obj.readframes(N+Noff)
y = np.frombuffer(raw,np.int16)
y = np.reshape(y,(N+Noff,obj.getnchannels()))
obj.close()
y=y[Noff:Noff+N,:]
groundtruth=np.mean(y.transpose(),axis=0,keepdims=True)


################ parameter definitions
cfs=np.array([[200, 400, 800, 1600, 3200, 6400, 12800]])
q=np.array([6,4,3,3,1.5,1,.25]);


######### "cochlea" model
y=gtb.filterbank(groundtruth,fs,cfs)
ymean=np.mean(y,axis=1,keepdims=True)
y=y-ymean


## Compute filters kappa
wax=2.*np.pi*fs/N*(np.arange(0,N)-np.floor(N/2))
kappa=np.sum(gtb.gammahat(wax,cfs),axis=0)+eps



##### fit individual channels

yfit=np.zeros((cfs.size,N))
dof=np.zeros(cfs.size)
tlist=[]
for nchan in range(cfs.size):
  Tmem=N/fs*cfs[0,nchan]*q[nchan]#  q per cycle
  dof[nchan]=Tmem
  print(nchan, Tmem)
  t=np.int32(np.arange(0,Tmem)/(Tmem)*N)
  tlist.append({'tpnts' : t})
  a=np.ones_like(t)
  signal=rs.subsig(y[nchan,[t]],t,a)

  c=cfs[0,nchan]/fs
  S=5./c
  #uest=rs.recsvr(signal,kerfunc, (S,c))
  #yfit[nchan,:]=rs.funcfit(uest,signal,N,kerfunc, (S,c))
  yfit[nchan,:], dmy = rs.contrecsvr(signal,N,kerfunc, (S,c), 500)


##### print compression:
print(np.sum(dof),N)


##### plot some examples
for nplt in range(6):
    t=np.array(tlist[nplt]['tpnts'])
    plt.subplot(3,2,nplt+1)
    plt.plot(t/fs,y[nplt,t],'o')
    plt.plot(np.arange(y.shape[1])/fs, y[nplt,:])
    plt.plot(np.arange(N)/fs, yfit[nplt,:])

plt.show()



yfinal=yfit+ymean

### reconstruct sound
import scipy.fft as fourier
tst=np.sum(fourier.fftshift(fourier.fft(yfinal)), axis=0)
tstorig=np.sum(fourier.fftshift(fourier.fft(y)), axis=0)

freqs=np.abs(wax/2/np.pi)
highpass10=np.where( np.bitwise_or(freqs<10, freqs>20000))[0]
tst[highpass10]=0

tmp=fourier.ifft(fourier.ifftshift(tst/kappa)).real
tmporig=fourier.ifft(fourier.ifftshift(tstorig/kappa)).real
tmp=tmp/np.std(tmp)*np.std(groundtruth)
tmporig=tmporig/np.std(tmporig)*np.std(groundtruth)


data=np.zeros((N,2), dtype=np.int16)
data[:,0]=np.int16(tmp)
data[:,1]=np.int16(tmp)

### save reconstruction
obj = wave.open('Reconstruction.wav','wb')
obj.setnchannels(2)
obj.setframerate(fs)
obj.setnframes(N)
obj.setsampwidth(2)
obj.writeframes(data)
obj.close()

#### save for plotting
from scipy.io import savemat
mdic = {"y" : y, "yfinal" : yfinal, "groundtruth" : groundtruth[0,:], "dof" : dof, "tlist" : tlist, "fs" : fs, "N" : N, "cfs" : cfs, "reconstruct" : np.int16(tmp)}
savemat("audioexample.mat", mdic)
