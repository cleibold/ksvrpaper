import recsvr as rs
import matplotlib.pyplot as plt
import numpy as np


def kerfunc(x):
  S=25
  c=0
  r=1.-(np.abs(x)/S);
  r=(r>0)*r
  r=r*np.cos(2.*np.pi*x*c)
  
  return r




##
## boring noise
##
N=50000;
nlow=100;
Tmem=25000;
t=np.int32(np.arange(0,Tmem)/(Tmem)*N)
a=np.ones_like(t)*.95+(t/N)*.05


no_repetitions=20
allerr=[]

for nrep in range(no_repetitions):

  y1=np.convolve(np.random.randn(N),np.ones(nlow)/nlow, 'same')
  y2=np.convolve(np.random.randn(N),np.ones(nlow)/nlow, 'same')
  y=np.zeros((2,N))
  y[0,:]=y1
  y[1,:]=y2
  
  
  
  err=[]

  for ncut in [50,100,200, 300, 500]:

    if ncut<=200:
      Marr=[50,100,200,500,1000,2000,3000]
    elif ncut>=300:
      Marr=[50, 100,200,500,1000,2000,3000,5000,10000,15000]
    else:
      Marr=[50, 100, 200, 300, 500, 1000, 1500, 2000, 3000, 4000,5000,10000,15000,20000]
      
    erract=[]
    for M in Marr:
      idsort=np.argsort(a[0:M])
      torder=t[0:M]
      aorder=a[0:M]
      tloop=torder[idsort]
      aloop=aorder[idsort]
  
          
      signal=rs.subsig(y[:,idsort],tloop,aloop)
      uest=rs.recsvr(signal,kerfunc,(),ncut)
      yfit=rs.funcfit(uest,signal,M*2,kerfunc,())

      mse=np.mean((yfit[:,torder]-y[:,0:M])**2)
      print(nrep, ncut,M,np.sqrt(mse))
      erract.append(mse)

    err.append(erract)
  

    
  allerr.append(err)



np.save('allerr.npy',allerr)
