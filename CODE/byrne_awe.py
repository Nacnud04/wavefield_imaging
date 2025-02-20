import os, sys

# get GPU count
nGPU = int(os.environ.get('nGPU'))
gpu = 0 # default gpu to use

# Import RSF Stuff
try:    from rsf.cluster import *
except: from rsf.proj    import *
import fdmod, random, functools, operator

random.seed(1002)
def add(x,y): return x+y
def myid(n): return '_'+functools.reduce(operator.add,['%d'%random.randint(0,9) for i in range(n)])

path = "/home/byrne/WORK/CODE/"

# -------------------------------------------------
# Some default parameters

def param(par):
    if 'nb'     not in par: par['nb']=0
    if 'nbell'  not in par: par['nbell']='y'
    if 'snap'   not in par: par['snap']='y'
    if 'jsnap'  not in par: par['jsnap']=100
    if 'jdata'  not in par: par['jdata']=1

    if 'ompchunk'  not in par: par['ompchunk']=1
    if 'ompnth'    not in par: par['ompnth']=0

    if 'dabc'      not in par: par['dabc']='y'
    if 'verb'      not in par: par['verb']='n'
    if 'expl'      not in par: par['expl']='n'
    if 'gaus'      not in par: par['gaus']='y'
    if 'sinc'      not in par: par['sinc']='n'
    if 'adj'       not in par: par['adj'] ='n'

    if 'fdorder'      not in par: par['fdorder']=4
    if 'optfd'        not in par: par['optfd']='n'
    if 'hybridbc'     not in par: par['hybridbc']='n'

    # only do this if in cartesian coordinate system
    if 'nx' in par:
        if 'nqx'      not in par: par['nqx']=par['nx']
        if 'oqx'      not in par: par['oqx']=par['ox']
        if 'dqx'      not in par: par['dqx']=par['dx']
        if 'nqy'      not in par: par['nqy']=par['ny']
        if 'oqy'      not in par: par['oqy']=par['oy']
        if 'dqy'      not in par: par['dqy']=par['dy']
        if 'nqz'      not in par: par['nqz']=par['nz']
        if 'oqz'      not in par: par['oqz']=par['oz']
        if 'dqz'      not in par: par['dqz']=par['dz']

    # Free surface can be defined two ways
    if 'free'     not in par: 
        if 'fsrf' not in par:
            par['fsrf']='n'
            par['free']='n'
        else:
            par['free']=par['fsrf']
    else :
        if 'fsrf' not in par:
            par['fsrf']='n'
            par['free']='n'
        else:
            par['free'] = par['fsrf']

    # Capture absorbing boundry condition with wavefield?
    if 'bnds'      not in par: par['bnds']='n'

    # Default gpu
    if 'gpu'       not in par: par['gpu']=gpu

    if nGPU > 0:
        par['useGPU'] = 'y'
    else:
        par['useGPU'] = 'n'

    # Wavefield axis:
    if 'xz'      not in par: par['xz']='n'



# -------------------------------------------------
# Read dict params

def awepar(par):

    param(par)

    awe = ' ' + \
          '''
          ompchunk=%(ompchunk)d ompnth=%(ompnth)d
          verb=%(verb)s fsrf=%(fsrf)s
          dabc=%(dabc)s nb=%(nb)d
          snap=%(snap)s jsnap=%(jsnap)d
          bnds=%(bnds)s expl=%(expl)s 
          adj=%(adj)s
          '''%par + ' '
    return awe

def awepargpu(par):
    awe = ' ' + \
          '''
          jdata=%(jdata)d dabc=%(dabc)s free=%(free)s
          snap=%(snap)s bnds=%(bnds)s jsnap=%(jsnap)d
          nb=%(nb)d gpu=%(gpu)d adj=%(adj)s
          '''%par + ' '
    return awe

def eicpar(par):
    eic = ' ' + \
          '''
          nhx=%(nhx)d nhy=%(nhy)d nhz=%(nhz)d nht=%(nht)d
          gaus=%(gaus)s verb=%(verb)s
          '''%par + ' '
    return eic

def iwindow(par):
    win = ' ' + \
          '''
          nqz=%(nqz)d oqz=%(oqz)g dqz=%(dqz)g
          nqx=%(nqx)d oqx=%(oqx)g dqx=%(dqx)g
          ''' % par + ' '
    return win

def aweoptpar(par):
    aweopt = ' ' + \
          '''
          verb=%(verb)s
          sinc=%(sinc)s expl=%(expl)s
          fdorder=%(fdorder)d optfd=%(optfd)s
          dabc=%(dabc)s nb=%(nb)d hybridbc=%(hybridbc)s
          snap=%(snap)s jsnap=%(jsnap)d
          '''%par + ' '
    return aweopt


# ------------------------------------------------------------
# wavelet
def wavelet(wav,frq,custom,par):
    
    Flow(wav,None,
         '''
         spike nsp=1 mag=1 n1=%(nt)d d1=%(dt)g o1=%(ot)g k1=%(kt)d |
         pad end1=%(nt)d |
         '''%par +
         '''
         ricker1 frequency=%g |
         '''%frq +
         '''
         window n1=%(nt)d |
         scale axis=123 |
         put label1=%(lt)s unit1=%(ut)s |
         transp
         '''%par)

# ---------------------------------------------------
# Cartesian Simulators

# variable-density acoustic FD modeling
def awefd2d(odat, owfl, idat, velo, dens, sou, rec, custom, par):
    
    if nGPU > 0: # if gpu's are detected run the gpu code

        Flow([odat, owfl], [idat, velo, sou, rec],
            f"{path}sfAWEFDcart2Dgpu" + 
            '''
            vel=${SOURCES[1]} sou=${SOURCES[2]} rec=${SOURCES[3]}
            wfl=${TARGETS[1]}
            '''  + ' ' + awepargpu(par) + ' ' + custom)

    # otherwise run the non GPU version
    else:
        Flow([odat,owfl],[idat,velo,dens,sou,rec],
            f"{path}sfAWEFDcart2Dcpu" + '''
            cden=n
            vel=${SOURCES[1]} den=${SOURCES[2]}
            sou=${SOURCES[3]} rec=${SOURCES[4]}
            wfl=${TARGETS[1]}
            ''' + ' ' + awepar(par) + ' ' + custom)


def awefd3d(odat, owfl, idat, velo, dens, sou, rec, custom, par):

    if nGPU > 0: # if gpu's are detected run the gpu code

        Flow([odat, owfl], [idat, velo, sou, rec],
            f"{path}sfAWEFDcart3Dgpu" +  
            '''
            vel=${SOURCES[1]} sou=${SOURCES[2]} rec=${SOURCES[3]} wfl=${TARGETS[1]}
            ''' + ' ' + awepargpu(par) +  ' ' + custom)

    # otherwise run the non GPU version
    else:
        Flow([odat,owfl],[idat,velo,dens,sou,rec],
            f"{path}sfAWEFD3D" +  
            ''' 
            cden=n
            vel=${SOURCES[1]} den=${SOURCES[2]}
            sou=${SOURCES[3]} rec=${SOURCES[4]}
            wfl=${TARGETS[1]}
            ''' + ' ' + awepar(par) + ' ' + custom)



# -------------------------------------------------------------------
# Spherical Simulators

def spawefd2d(odat, owfl, idat, velo, dens, sou, rec, custom, par):

    if nGPU > 0: # if gpu's are detected run the gpu code

        Flow([odat, owfl], [idat, velo, sou, rec], 
            f"{path}sfAWEFDspher2Dgpu" + 
            '''
            vel=${SOURCES[1]} sou=${SOURCES[2]} rec=${SOURCES[3]}
            wfl=${TARGETS[1]}
            ''' + ' ' + awepargpu(par) + ' ' + custom)

    # otherwise run the non GPU version
    else:
        
        Flow([odat, owfl], [idat, velo, sou, rec], 
            f"{path}sfAWEFDspher2Dcpu" + 
            '''
            vel=${SOURCES[1]} sou=${SOURCES[2]} rec=${SOURCES[3]}
            wfl=${TARGETS[1]}
            ''' + ' ' + awepar(par) + ' ' + custom)

def spawefd3d(odat, owfl, idat, velo, dens, sou, rec, custom, par):
    
    if nGPU > 0: # if gpu's are detected run the gpu code
        
        Flow([odat, owfl], [idat, velo, sou, rec], 
            f"{path}sfAWEFDspher3Dgpu" + 
            '''
            vel=${SOURCES[1]} sou=${SOURCES[2]} rec=${SOURCES[3]}
            wfl=${TARGETS[1]}
            ''' + ' ' + awepargpu(par) + ' ' + custom)

    else: # otherwise run the cpu version

        Flow([odat, owfl], [idat, velo, sou, rec], 
            f"{path}sfAWEFDspher3Dcpu" + 
            '''
            vel=${SOURCES[1]} sou=${SOURCES[2]} rec=${SOURCES[3]}
            wfl=${TARGETS[1]}
            ''' + ' ' + awepar(par) + ' ' + custom)



# ------------------------------------------------------------
# constant-density acoustic FD modeling
def cdafd2d(odat,owfl,idat,velo,sou,rec,custom,par):
    Flow([odat,owfl],[idat,velo,sou,rec],
         '''
         awefd2d cden=y
         vel=${SOURCES[1]}
         sou=${SOURCES[2]} rec=${SOURCES[3]}
         wfl=${TARGETS[1]}
         ''' + ' ' + aweoptpar(par) + ' ' + custom)

def cdafd3d(odat,owfl,idat,velo,sou,rec,custom,par):
    Flow([odat,owfl],[idat,velo,sou,rec],
         '''
         awefd3d cden=y
         vel=${SOURCES[1]}
         sou=${SOURCES[2]} rec=${SOURCES[3]}
         wfl=${TARGETS[1]}
         ''' + ' ' + aweoptpar(par) + ' ' + custom)

# ------------------------------------------------------------
# variable-density acoustic FD modeling with optimized fd and hybrid bc
def awefd2dopt(odat,owfl,idat,velo,dens,sou,rec,custom,par):
    Flow([odat,owfl],[idat,velo,dens,sou,rec],
         '''
         awefd2dopt cden=n
         vel=${SOURCES[1]} den=${SOURCES[2]}
         sou=${SOURCES[3]} rec=${SOURCES[4]}
         wfl=${TARGETS[1]}
         ''' + ' ' + awepar(par) + ' ' + custom)

def awefd3dopt(odat,owfl,idat,velo,dens,sou,rec,custom,par):
    Flow([odat,owfl],[idat,velo,dens,sou,rec],
         '''
         awefd3dopt cden=n
         vel=${SOURCES[1]} den=${SOURCES[2]}
         sou=${SOURCES[3]} rec=${SOURCES[4]}
         wfl=${TARGETS[1]}
         ''' + ' ' + awepar(par) + ' ' + custom)

# ------------------------------------------------------------
# constant-density acoustic FD modeling with optimized fd and hybrid bc
def cdafd2dopt(odat,owfl,idat,velo,sou,rec,custom,par):
    Flow([odat,owfl],[idat,velo,sou,rec],
         '''
         awefd2dopt cden=y
         vel=${SOURCES[1]}
         sou=${SOURCES[2]} rec=${SOURCES[3]}
         wfl=${TARGETS[1]}
         ''' + ' ' + aweoptpar(par) + ' ' + custom)

def cdafd3dopt(odat,owfl,idat,velo,sou,rec,custom,par):
    Flow([odat,owfl],[idat,velo,sou,rec],
         '''
         awefd3dopt cden=y
         vel=${SOURCES[1]}
         sou=${SOURCES[2]} rec=${SOURCES[3]}
         wfl=${TARGETS[1]}
         ''' + ' ' + aweoptpar(par) + ' ' + custom)

# ------------------------------------------------------------
# AWE modeling operator
def aweop2d(wfl,sou,vel,custom,par):
    Flow(wfl,[sou,vel],
         '''
         aweop2d
         vel=${SOURCES[1]}
         '''+ ' ' + awepar(par) + ' ' + custom)

def aweop3d(wfl,sou,vel,custom,par):
    Flow(wfl,[sou,vel],
         '''
         aweop3d
         vel=${SOURCES[1]}
         '''+ ' ' + awepar(par) + ' ' + custom)

# ------------------------------------------------------------
# CIC forward: opr * wfl = img
def cic2dF(img,wfl,opr,custom,par):
    Flow(img,[wfl,opr],
         '''
         cicop2d adj=n wflcausal=y oprcausal=n
         opr=${SOURCES[1]}
         ''' + ' ' + custom)
def cic3dF(img,wfl,opr,custom,par):
    Flow(img,[wfl,opr],
         '''
         cicop3d adj=n wflcausal=y oprcausal=n
         opr=${SOURCES[1]}
         ''' + ' ' + custom)

# CIC adjoint: opr * img = wfl
def cic2dA(wfl,img,opr,custom,par):
    Flow(wfl,[img,opr],
         '''
         cicop2d adj=y wflcausal=n oprcausal=n
         opr=${SOURCES[1]}
         ''' + ' ' + custom)
def cic3dA(wfl,img,opr,custom,par):
    Flow(wfl,[img,opr],
         '''
         cicop3d adj=y wflcausal=n oprcausal=n
         opr=${SOURCES[1]}
         ''' + ' ' + custom)

# ------------------------------------------------------------
# EIC forward: opr * wfl = img
def eic2dF(img,wfl,opr,cip,custom,par):
    Flow(img,[wfl,opr,cip],
         '''
         eicop2d adj=n wflcausal=y oprcausal=n
         opr=${SOURCES[1]} cip=${SOURCES[2]}
         ''' + eicpar(par) + ' ' + custom )
def eic3dF(img,wfl,opr,cip,custom,par):
    Flow(img,[wfl,opr,cip],
         '''
         eicop3d adj=n wflcausal=y oprcausal=n
         opr=${SOURCES[1]} cip=${SOURCES[2]}
         ''' + eicpar(par) + ' ' + custom )

# EIC adjoint: opr * img = wfl
def eic2dA(wfl,img,opr,cip,custom,par):
    Flow(wfl,[img,opr,cip],
         '''
         eicop2d adj=y wflcausal=n oprcausal=n
         opr=${SOURCES[1]} cip=${SOURCES[2]}
         ''' + ' ' + custom )
def eic3dA(wfl,img,opr,cip,custom,par):
    Flow(wfl,[img,opr,cip],
         '''
         eicop3d adj=y wflcausal=n oprcausal=n
         opr=${SOURCES[1]} cip=${SOURCES[2]}
         ''' + ' ' + custom )

# ------------------------------------------------------------
# EIC 'source' kernel
def kerS2d(ker,uS,uR,res,cip,vel,custom,par):
    eic2dA(ker+'_gS',res,uR,cip,'gaus=y wflcausal=y oprcausal=n'+' '+custom,par)
    aweop2d(ker+'_aS',ker+'_gS',vel,'adj=y'+' '+custom,par)
    cic2dF(ker,uS,ker+'_aS','wflcausal=y oprcausal=y'+' '+custom,par)

def kerS3d(ker,uS,uR,res,cip,vel,custom,par):
    eic3dA(ker+'_gS',res,uR,cip,'gaus=y wflcausal=y oprcausal=n'+' '+custom,par)
    aweop3d(ker+'_aS',ker+'_gS',vel,'adj=y'+' '+custom,par)
    cic3dF(ker,uS,ker+'_aS','wflcausal=y oprcausal=y'+' '+custom,par)

# ------------------------------------------------------------
# EIC 'receiver' kernel
def kerR2d(ker,uS,uR,res,cip,vel,custom,par):
    Flow(  res+'_',res,'reverse which=7 opt=i')
    eic2dA(ker+'_gR',res+'_',uS,cip,'gaus=y wflcausal=y oprcausal=y'+' '+custom,par)
    aweop2d(ker+'_aR',ker+'_gR',vel,'adj=n'+' '+custom,par)
    cic2dF(ker,uR,ker+'_aR','wflcausal=y oprcausal=n'+' '+custom,par)

def kerR3d(ker,uS,uR,res,cip,vel,custom,par):
    Flow(  res+'_',res,'reverse which=15 opt=i')
    eic3dA(ker+'_gR',res+'_',uS,cip,'gaus=y wflcausal=y oprcausal=y'+' '+custom,par)
    aweop3d(ker+'_aR',ker+'_gR',vel,'adj=n'+' '+custom,par)
    cic3dF(ker,uR,ker+'_aR','wflcausal=y oprcausal=n'+' '+custom,par)

# ------------------------------------------------------------
# EIC total kernel
def kerT2d(ker,uS,uR,res,cip,vel,custom,par):
    kerS2d(ker+'_S',uS,uR,res,cip,vel,custom,par)
    kerR2d(ker+'_R',uS,uR,res,cip,vel,custom,par)
    Flow(ker,[ker+'_S',ker+'_R'],'add ${SOURCES[1]}')

def kerT3d(ker,uS,uR,res,cip,vel,custom,par):
    kerS3d(ker+'_S',uS,uR,res,cip,vel,custom,par)
    kerR3d(ker+'_R',uS,uR,res,cip,vel,custom,par)
    Flow(ker,[ker+'_S',ker+'_R'],'add ${SOURCES[1]}')

# ------------------------------------------------------------
# variable-density RTM w/ CIC
def cicmig(icic,
           sdat,scoo,
           rdat,rcoo,
           velo,dens,
           custom,par):

    M8R='$RSFROOT/bin/sf'
    DPT=os.environ.get('TMPDATAPATH')

    swfl=icic+'swfl'+myid(16)
    rdrv=icic+'rdrv'+myid(16)
    rwfl=icic+'rwfl'+myid(16)

    Flow(icic,[sdat,scoo,rdat,rcoo,velo,dens],
         '''
         %sawefd2d < ${SOURCES[0]} cden=n %s verb=n
         sou=${SOURCES[1]}
         rec=${SOURCES[1]}
         vel=${SOURCES[4]}
         den=${SOURCES[5]}
         wfl=%s datapath=%s/
         >/dev/null;
         '''%(M8R,iwindow(par)+awepar(par)+custom,swfl,DPT) +
         '''
         %sreverse < ${SOURCES[2]} which=2 opt=i verb=n >%s datapath=%s/;
         '''%(M8R,rdrv,DPT) +
         '''
         %sawefd2d < %s cden=n %s verb=n
         sou=${SOURCES[3]}
         rec=${SOURCES[3]}
         vel=${SOURCES[4]}
         den=${SOURCES[5]}
         wfl=%s datapath=%s/
         >/dev/null;
         '''%(M8R,rdrv,iwindow(par)+awepar(par)+custom,rwfl,DPT) +
         '''
         %scicop2d <%s adj=n wflcausal=y oprcausal=n %s
         opr=%s
         >${TARGETS[0]};
         '''%(M8R,swfl,custom,rwfl) +
         '''
         %srm %s %s %s
         '''%(M8R,swfl,rdrv,rwfl),
              stdin=0,
              stdout=0)

# ------------------------------------------------------------
# constant-density RTM w/ CIC
def cicmigCD(icic,
             sdat,scoo,
             rdat,rcoo,
             velo,
             custom,par):

    M8R='$RSFROOT/bin/sf'
    DPT=os.environ.get('TMPDATAPATH')

    swfl=icic+'swfl'+myid(16)
    rdrv=icic+'rdrv'+myid(16)
    rwfl=icic+'rwfl'+myid(16)

    Flow(icic,[sdat,scoo,rdat,rcoo,velo],
         '''
         %sawefd2d < ${SOURCES[0]} cden=y %s verb=n
         sou=${SOURCES[1]}
         rec=${SOURCES[1]}
         vel=${SOURCES[4]}
         wfl=%s datapath=%s/
         >/dev/null;
         '''%(M8R,iwindow(par)+awepar(par)+custom,swfl,DPT) +
         '''
         %sreverse < ${SOURCES[2]} which=2 opt=i verb=n >%s datapath=%s/;
         '''%(M8R,rdrv,DPT) +
         '''
         %sawefd2d < %s cden=y %s verb=n
         sou=${SOURCES[3]}
         rec=${SOURCES[3]}
         vel=${SOURCES[4]}
         wfl=%s datapath=%s/
         >/dev/null;
         '''%(M8R,rdrv,iwindow(par)+awepar(par)+custom,rwfl,DPT) +
         '''
         %scicop2d <%s adj=n wflcausal=y oprcausal=n %s
         opr=%s
         >${TARGETS[0]};
         '''%(M8R,swfl,custom,rwfl) +
         '''
         %srm %s %s %s
         '''%(M8R,swfl,rdrv,rwfl),
              stdin=0,
              stdout=0)

# ------------------------------------------------------------
# veriable-density RTM w/ CIC and EIC
def eicmig(icic,
           ieic,icoo,
           sdat,scoo,
           rdat,rcoo,
           velo,dens,
           custom,par):

    M8R='$RSFROOT/bin/sf'
    DPT=os.environ.get('TMPDATAPATH')

    swfl=ieic+'swfl'+myid(16)
    rdrv=ieic+'rdrv'+myid(16)
    rwfl=ieic+'rwfl'+myid(16)

    Flow([icic,ieic],[sdat,scoo,rdat,rcoo,icoo,velo,dens],
         '''
         %sawefd2d < ${SOURCES[0]} cden=n %s verb=n
         sou=${SOURCES[1]}
         rec=${SOURCES[1]}
         vel=${SOURCES[5]}
         den=${SOURCES[6]}
         wfl=%s datapath=%s/
         >/dev/null;
         '''%(M8R,iwindow(par)+awepar(par)+custom,swfl,DPT) +
         '''
         %sreverse < ${SOURCES[2]} which=2 opt=i verb=n >%s datapath=%s/;
         '''%(M8R,rdrv,DPT) +
         '''
         %sawefd2d < %s cden=n %s verb=n
         sou=${SOURCES[3]}
         rec=${SOURCES[3]}
         vel=${SOURCES[5]}
         den=${SOURCES[6]}
         wfl=%s datapath=%s/
         >/dev/null;
         '''%(M8R,rdrv,iwindow(par)+awepar(par)+custom,rwfl,DPT) +
         '''
         %scicop2d <%s adj=n wflcausal=y oprcausal=n %s
         opr=%s
         >${TARGETS[0]};
         '''%(M8R,swfl,custom,rwfl) +
         '''
         %seicop2d <%s adj=n wflcausal=y oprcausal=n %s
         opr=%s cip=${SOURCES[4]}
         >${TARGETS[1]};
         '''%(M8R,swfl,eicpar(par)+custom,rwfl) +
         '''
         %srm %s %s %s
         '''%(M8R,swfl,rdrv,rwfl),
              stdin=0,
              stdout=0)

# ------------------------------------------------------------
# constant-density RTM w/ CIC and EIC
def eicmigCD(icic,
             ieic,icoo,
             sdat,scoo,
             rdat,rcoo,
             velo,
             custom,par):

    M8R='$RSFROOT/bin/sf'
    DPT=os.environ.get('TMPDATAPATH')

    swfl=ieic+'swfl'+myid(16)
    rdrv=ieic+'rdrv'+myid(16)
    rwfl=ieic+'rwfl'+myid(16)

    Flow([icic,ieic],[sdat,scoo,rdat,rcoo,icoo,velo],
         '''
         %sawefd2d < ${SOURCES[0]} cden=y %s verb=n
         sou=${SOURCES[1]}
         rec=${SOURCES[1]}
         vel=${SOURCES[5]}
         wfl=%s datapath=%s/
         >/dev/null;
         '''%(M8R,iwindow(par)+awepar(par)+custom,swfl,DPT) +
         '''
         %sreverse < ${SOURCES[2]} which=2 opt=i verb=n >%s datapath=%s/;
         '''%(M8R,rdrv,DPT) +
         '''
         %sawefd2d < %s cden=y %s verb=n
         sou=${SOURCES[3]}
         rec=${SOURCES[3]}
         vel=${SOURCES[5]}
         wfl=%s datapath=%s/
         >/dev/null;
         '''%(M8R,rdrv,iwindow(par)+awepar(par)+custom,rwfl,DPT) +
         '''
         %scicop2d <%s adj=n wflcausal=y oprcausal=n %s
         opr=%s
         >${TARGETS[0]};
         '''%(M8R,swfl,custom,rwfl) +
         '''
         %seicop2d <%s adj=n wflcausal=y oprcausal=n %s
         opr=%s cip=${SOURCES[4]}
         >${TARGETS[1]};
         '''%(M8R,swfl,eicpar(par)+custom,rwfl) +
         '''
         %srm %s %s %s
         '''%(M8R,swfl,rdrv,rwfl),
              stdin=0,
              stdout=0)

# ------------------------------------------------------------
# zero-offset RTM - variable density
def awertm2d(imag,data,rcoo,velo,dens,custom,par):
    M8R='$RSFROOT/bin/sf'
    DPT=os.environ.get('TMPDATAPATH',os.environ.get('DATAPATH'))

    rwfl = imag+'wwfl'+myid(16)

    Flow(imag,[data,rcoo,velo,dens],
         '''
         %sawefd2d < ${SOURCES[0]} adj=y cden=n %s verb=n
         sou=${SOURCES[1]}
         rec=${SOURCES[1]}
         vel=${SOURCES[2]}
         den=${SOURCES[3]}
         wfl=%s datapath=%s/ %s
         >/dev/null;
         '''%(M8R,awepar(par)+' jsnap=%d'%(par['nt']-1),rwfl,DPT,custom) +
         '''
         %swindow < %s n3=1 f3=1 >${TARGETS[0]};
         '''%(M8R,rwfl) +
         '''
         %srm %s
         '''%(M8R,rwfl),
              stdin=0,
              stdout=0)

def awertm3d(imag,data,rcoo,velo,dens,custom,par):
    M8R='$RSFROOT/bin/sf'
    DPT=os.environ.get('TMPDATAPATH',os.environ.get('DATAPATH'))

    rwfl = imag+'wwfl'+myid(16)

    Flow(imag,[data,rcoo,velo,dens],
         '''
         %sawefd3d < ${SOURCES[0]} adj=y cden=n %s verb=n
         sou=${SOURCES[1]}
         rec=${SOURCES[1]}
         vel=${SOURCES[2]}
         den=${SOURCES[3]}
         wfl=%s datapath=%s/ %s
         >/dev/null;
         '''%(M8R,awepar(par)+' jsnap=%d'%(par['nt']-1),rwfl,DPT,custom) +
         '''
         %swindow < %s n4=1 f4=1 >${TARGETS[0]};
         '''%(M8R,rwfl) +
         '''
         %srm %s
         '''%(M8R,rwfl),
              stdin=0,
              stdout=0)

# zero-offset RTM - constant density
def cdartm2d(imag,data,rcoo,velo,custom,par):
    M8R='$RSFROOT/bin/sf'
    DPT=os.environ.get('TMPDATAPATH',os.environ.get('DATAPATH'))

    rwfl = imag+'wwfl'+myid(16)

    Flow(imag,[data,rcoo,velo],
         '''
         %sawefd2d < ${SOURCES[0]} adj=y cden=y %s verb=n
         sou=${SOURCES[1]}
         rec=${SOURCES[1]}
         vel=${SOURCES[2]}
         wfl=%s datapath=%s/ %s
         >/dev/null;
         '''%(M8R,awepar(par)+' jsnap=%d'%(par['nt']-1),rwfl,DPT,custom) +
         '''
         %swindow < %s n3=1 f3=1 >${TARGETS[0]};
         '''%(M8R,rwfl) +
         '''
         %srm %s
         '''%(M8R,rwfl),
              stdin=0,
              stdout=0)

def cdartm3d(imag,data,rcoo,velo,custom,par):
    M8R='$RSFROOT/bin/sf'
    DPT=os.environ.get('TMPDATAPATH',os.environ.get('DATAPATH'))

    rwfl = imag+'wwfl'+myid(16)

    Flow(imag,[data,rcoo,velo],
         '''
         %sawefd3d < ${SOURCES[0]} adj=y cden=y %s verb=n
         sou=${SOURCES[1]}
         rec=${SOURCES[1]}
         vel=${SOURCES[2]}
         wfl=%s datapath=%s/ %s
         >/dev/null;
         '''%(M8R,awepar(par)+' jsnap=%d'%(par['nt']-1),rwfl,DPT,custom) +
         '''
         %swindow < %s n4=1 f4=1 >${TARGETS[0]};
         '''%(M8R,rwfl) +
         '''
         %srm %s
         '''%(M8R,rwfl),
              stdin=0,
              stdout=0)

# ------------------------------------------------------------
# zero-offset RTM with awefd2dopt/awefd3dopt - constant density
def cdartm2dopt(imag,data,rcoo,velo,custom,par):
    M8R='$RSFROOT/bin/sf'
    DPT=os.environ.get('TMPDATAPATH',os.environ.get('DATAPATH'))

    rwfl = imag+'wwfl'+myid(16)

    Flow(imag,[data,rcoo,velo],
         '''
         %sawefd2dopt < ${SOURCES[0]} adj=y %s verb=n
         sou=${SOURCES[1]}
         rec=${SOURCES[1]}
         vel=${SOURCES[2]}
         wfl=%s datapath=%s/ %s
         >/dev/null;
         '''%(M8R,awepar(par)+' jsnap=%d'%(par['nt']-1),rwfl,DPT,custom) +
         '''
         %swindow < %s n3=1 f3=1 >${TARGETS[0]};
         '''%(M8R,rwfl) +
         '''
         %srm %s
         '''%(M8R,rwfl),
              stdin=0,
              stdout=0)

def cdartm3dopt(imag,data,rcoo,velo,custom,par):
    M8R='$RSFROOT/bin/sf'
    DPT=os.environ.get('TMPDATAPATH',os.environ.get('DATAPATH'))

    rwfl = imag+'wwfl'+myid(16)

    Flow(imag,[data,rcoo,velo],
         '''
         %sawefd3dopt < ${SOURCES[0]} adj=y %s verb=n
         sou=${SOURCES[1]}
         rec=${SOURCES[1]}
         vel=${SOURCES[2]}
         wfl=%s datapath=%s/ %s
         >/dev/null;
         '''%(M8R,awepar(par)+' jsnap=%d'%(par['nt']-1),rwfl,DPT,custom) +
         '''
         %swindow < %s n4=1 f4=1 >${TARGETS[0]};
         '''%(M8R,rwfl) +
         '''
         %srm %s
         '''%(M8R,rwfl),
              stdin=0,
              stdout=0)

# ------------------------------------------------------------
def dPAD2d(wfld,trac,ix,iz,par):
    Flow(wfld,trac,
        '''
        transp plane=23 |
        pad beg1=%d n1out=%d beg2=%d n2out=%d |
        put o1=%g d1=%g o2=%g d2=%g
        '''%(iz,par['nz'],ix,par['nx'],
             par['oz'],par['dz'],
             par['ox'],par['dx']))

def dWIN2d(trac,wfld,ix,iz,par):
    Flow(trac,wfld,
         '''
         window n1=1 f1=%d n2=1 f2=%d |
         transp
         '''%(iz,ix))

# ------------------------------------------------------------
# wavefield-over-model plot
def wom(wom,wfld,velo,vmean,par):
    M8R='$RSFROOT/bin/sf'
    DPT=os.environ.get('TMPDATAPATH',os.environ.get('DATAPATH'))

    #if(not par.has_key('wweight')): par['wweight']=1
    if 'wweight' not in par: par['wweight']=1

    wtmp = wfld + 'tmp'+myid(16)
    vtmp = wfld + 'vel'+myid(16)

    Flow(wom,[velo,wfld],
        '''
        %sscale < ${SOURCES[1]} axis=123 >%s datapath=%s/;
        '''%(M8R,wtmp,DPT)
        +
        '''
        %sadd < ${SOURCES[0]} add=-%g |
        scale axis=123 |
        spray axis=3 n=%d o=%g d=%g
        >%s datapath=%s/;
        '''%(M8R,vmean,
             (par['nt']-1)/par['jsnap']+1,
             par['ot'],
             par['dt']*par['jsnap'],
             vtmp,DPT)
        +
        '''
        %sadd scale=1,%g <%s %s >${TARGETS[0]};
        '''%(M8R,par['wweight'],vtmp,wtmp)
        +
        '''
        %srm %s %s
        '''%(M8R,wtmp,vtmp),
        stdin=0,
        stdout=0)
