import recsvr as rs
import matplotlib.pyplot as plt
import numpy as np


def kerfunc(x):
  S=50
  c=0
  r=1.-(np.abs(x)/S);
  r=(r>0)*r
  r=r*np.cos(2.*np.pi*x*c)
  
  return r




##
## boring noise
##
N=20000;
nlow=50;
y1=np.convolve(np.random.randn(N),np.ones(nlow)/nlow, 'same')
y2=np.convolve(np.random.randn(N),np.ones(nlow)/nlow, 'same')
y=np.zeros((2,N))
y[0,:]=y1
y[1,:]=y2

Tmem=500;
#t=(np.random.permutation(N)[0:Tmem])
t=np.int32(np.arange(0,Tmem)/(Tmem)*N)

##### choose importance values
a=np.random.rand(Tmem)*0.5+.5
idorder=np.argsort(a)
torder=t[idorder]
aorder=a[idorder]


#### fit
signal=rs.subsig(y[:,torder],torder,aorder)
uest=rs.recsvr(signal,kerfunc,())
yfit=rs.funcfit(uest,signal,N,kerfunc,())



#### plot examples
pxmpl=[0, 1]
for nplt in range(2):
    
    plt.subplot(2,2,nplt+1)
    plt.plot(t,y[pxmpl[nplt],t],'o')
    plt.plot(range(y.shape[1]),y[pxmpl[nplt],:])
    plt.plot(range(N),yfit[pxmpl[nplt],:])


plt.subplot(2,2,3)    
plt.plot(aorder, np.abs(yfit[0,torder]-y[0,torder]), '.')
plt.plot(aorder, np.abs(yfit[1,torder]-y[1,torder]), '.')

plt.subplot(2,2,4)
plt.plot(torder, np.abs(yfit[0,torder]-y[0,torder]), '.')
plt.plot(torder, np.abs(yfit[1,torder]-y[1,torder]), '.')

plt.show()



### for plotting
from scipy.io import savemat
mdic = {"y" : y, "yfit" : yfit, "torder" : torder, "aorder" : aorder}
savemat("fig3_0.mat", mdic)
