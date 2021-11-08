import numpy as np

#### 4th order gammatone filter effective algorithm in the time domain

### bandwidth scale
bwfac=2

def erb(cf):

    return 24.7 * (4.37*cf/1000 + 1)

def filterbank(y,fs,cf):

    nfreqs=cf.shape[1]
    N=y.shape[1]
  
    b=1.019*erb(cf)*bwfac
    a=((2*np.pi*b.transpose())**4)/6
    c6=a/fs
    
    g=np.zeros((nfreqs,N),dtype=complex)
    y2=np.zeros((nfreqs,1),dtype=complex)
    y1=np.zeros((nfreqs,1),dtype=complex)
    y0=y[0,0]*c6
    
  
    c2=3./fs
    c1=3./(fs**2)
    c0=1./(fs**3)
    
    c3=2./fs
    c4=1./(fs**2)
  
    c5=1./fs
    
  
    decay=np.exp(-2.*np.pi*(b+1j*cf)/fs).transpose()


    for nt in np.arange(1,N):
        
        g[:,[nt]] = decay*(g[:,[nt-1]] + c2*y2 + c1*y1 + c0*y0)
        
        y2 = decay*(y2 + c3*y1 + c4*y0)
        y1 = decay*(y1 + c5*y0)
        y0 = decay*y0 + y[0,nt]*c6;
 

        
    r=g.real

    return r


## the theoretrical frequency domain filter
def gammahat(w,cf):
    
  b=1.019*erb(cf).transpose()*bwfac
  a=((2*np.pi*b)**4)
  
  z=2*np.pi*(cf.transpose()+1j*b)
  
    
  kappa= 0.5*a*(1./(w-z)**4 + 1./(w+z.conj())**4);
  return kappa
