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
## going to the movies
##
import imageio
vid = imageio.get_reader('potemkinsnip.mp4',  'ffmpeg')#re_potempkin
print(vid.get_meta_data() )

count = 0
vecout=[]
change=[]
imprev=np.zeros((576,768,3))
try:
    for _ in vid:
        imraw=np.array(vid.get_data(count), dtype=np.int32)
        vecout.append(np.reshape(imraw[:,:,:],(576*768*3,1)))
        iprev=imraw
        count += 1
except RuntimeError:
    print('something went wront in iterating, maybe wrong fps number')
finally:
    print('number of frames counted {}, number of frames in metada {}'.format(count, vid.get_meta_data()['nframes']))

vid.close()

y=np.squeeze(np.array(vecout)).transpose()
#y=y[:,0:150]
N=y.shape[1]
ymean=np.mean(y,axis=1,keepdims=True)
y=y-ymean




#### fit the weights
Tmem=20;
t=np.int32(np.arange(0,Tmem)/(Tmem)*N)
a=np.ones_like(t)

signal=rs.subsig(y[:,t],t,a)
uest=rs.recsvr(signal,kerfunc,())
yfit=rs.funcfit(uest,signal,N,kerfunc,())


##### show examples
pxmpl=[32, 12144, 23212, 30210]
for nplt in range(4):
    
    plt.subplot(2,2,nplt+1)
    plt.plot(t,y[pxmpl[nplt],t],'o')
    plt.plot(range(y.shape[1]),y[pxmpl[nplt],:])
    plt.plot(range(N),yfit[pxmpl[nplt],:])

plt.show()



### save the reconstruction
yfinal=yfit+ymean
corr=(yfinal>0.)*yfinal
idS=np.where(yfinal>255.)
corr[idS[0],idS[1]]=255.

txmpl=np.array([10, 30, 50, 70, 90, 110])
still=np.zeros((576,768,3,txmpl.size))
writer = imageio.get_writer('fileout.mp4', fps=25)
counto=0
ns=0
while counto<N:
    vout=np.reshape(corr[:,counto], (576,768,3))
    writer.append_data(np.array(vout,dtype=np.uint8))
    
    if np.any(txmpl == counto):
      #print(ns)
      still[:,:,:,ns]=np.array(vout,dtype=np.uint8)
      ns +=1
      
    counto += 1
    
writer.close()


##### for plotting
from scipy.io import savemat
mdic = {"y" : y, "yfit" : yfit, "yfinal" : yfinal, "t" : t, 'examples' : still}
savemat("movieexample.mat", mdic)
