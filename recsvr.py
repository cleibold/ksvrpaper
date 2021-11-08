import numpy as np

class subsig:
    def __init__(self,y,t,a):
        self.y=y;
        self.t=t;
        self.a=a;

def modelfunc(uu,mk):
    return np.sum(uu*mk,axis=1)


######### recursive fit
def recsvr(signal,myk,par,Ncut=300):

   
    y=signal.y
    
    Tmem=y.shape[1]
    ndim=y.shape[0]

    #print(Tmem,ndim)
    
    ismp=np.append(signal.t,0)
    att=np.append(signal.a,0)
    
    ids=np.zeros(Tmem)
    iprev=0
    errest=np.zeros((ndim,Tmem))
    
    uest=np.zeros((ndim,Tmem))
    Npest=myk(0,*par)*att[0]**2
    
    Gest=np.zeros((Tmem,Tmem)) 
    dvest=np.zeros((Tmem,1))
    upest=np.zeros((ndim,1))

    #always add the pth pattern
    for P in range(Tmem): 
        
        
        ids[0:P]=ids[0:P]+ismp[P]-iprev
    
        errest[:,P]=y[:,P]-modelfunc(uest[:,0:P]*att[0:P],myk(ids[0:P],*par))
        upest[:,0] = errest[:,P]/Npest

        
        
        uest[:,0:P]=uest[:,0:P]-upest@dvest[0:P].transpose()

        ids[P]=0
        uest[:,P]=upest[:,0]
    
        Ktmp=att[P+1]*(att[0:P+1]*myk(ids[0:P+1]+ismp[P+1]-ismp[P],*par))
        Kp=np.zeros((Ktmp.size,1))
        Kp[:,0]=Ktmp
        
        dtmp=dvest[0:P]


        if P< Ncut:
            Gest[0:P,0:P] += np.outer(dtmp,dtmp)/Npest
            Gest[P,0:P] = -dtmp.transpose()/Npest
            Gest[0:P,P] = -dvest[0:P,0]/Npest
            Gest[P,P]   = 1./Npest
        else:
            dvs=dvest[P-Ncut:P];
            dvl=np.zeros((P,1));
            dvl[P-Ncut:P]=dvs;
            Gest[P-Ncut:P,P-Ncut:P] += np.outer(dvs,dvs)/Npest
            Gest[P,0:P] = -dvl.transpose()/Npest
            Gest[0:P,P] = -dvl[:,0]/Npest
            Gest[P,P]   = 1./Npest

            
        dvest[0:P+1]=Gest[0:P+1,0:P+1]@Kp

        Npest=(att[P+1]**2)*myk(0,*par)- Kp.transpose()@dvest[0:P+1]#Gest[0:P+1,0:P+1]@Kp
    
        iprev=ismp[P]


    return uest

### reconstruction
def funcfit(uest,signal,N,myk,par):

    yfinal=np.zeros((signal.y.shape[0],N))
    
    for nt in range(N-1):
        yfinal[:,nt]=modelfunc(uest,myk(nt-signal.t,*par));  

    return yfinal



#######################
# for longer signals (see audio example) split up into folds and fit foldwise
#######################

def contrecsvr(signal,N,myk,par,Ncut=300):

    y=signal.y
    t=signal.t
    a=signal.a
    
    L=y.shape[1]
    ndim=y.shape[0]

    uest=np.zeros((ndim,L))
    yfit=np.zeros((ndim,N))
    
    folds=np.int(L/Ncut)+1

    n0=0
    m0=0
    for nf in range(folds):
        n1=np.min([n0+Ncut, L])
        Nloc=np.int(N/L*(n1-n0))
        idloc=range(n0,n1)
        sigloc=subsig(y[:,idloc],t[idloc]-m0,a[idloc])
        uest[:,idloc]=recsvr(sigloc,myk,par)

        tmp=funcfit(uest[:,idloc],sigloc,Nloc,myk,par)
        m1=m0+tmp.shape[1]
        yfit[:,m0:m1]=tmp
        
        n0=n1
        m0=m1
        

    return yfit, uest
