import os, sys

# get GPU count
nGPU = int(os.environ.get('nGPU'))
gpu = 1 # default gpu to use

# Import RSF Stuff
try:    from rsf.cluster import *
except: from rsf.proj    import *
import fdmod


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

    if 'fdorder'      not in par: par['fdorder']=4
    if 'optfd'        not in par: par['optfd']='n'
    if 'hybridbc'     not in par: par['hybridbc']='n'

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
        if 'frsf' not in par:
            par['fsrf']='n'
            par['free']='n'
        else:
            par['free']=par['frsf']

    # Capture absorbing boundry condition with wavefield?
    if 'bnds'      not in par: par['bnds']='n'

    # Default gpu
    if 'gpu'       not in par: par['gpu']=1 # default gpu is gpu 1

    # Wavefield axis:
    if 'xz'      not in par: par['xz']='n'



# -------------------------------------------------
# Read dict params

def awepar(par):
    awe = ' ' + \
          '''
          ompchunk=%(ompchunk)d ompnth=%(ompnth)d
          verb=%(verb)s fsrf=%(fsrf)s
          dabc=%(dabc)s nb=%(nb)d
          snap=%(snap)s jsnap=%(jsnap)d
          '''%par + ' '
    return awe


def awepargpu(par):
    awe = ' ' + \
          '''
          jdata=%(jdata)d dabc=%(dabc)s free=%(free)s
          snap=%(snap)s bnds=%(bnds)s jsnap=%(jsnap)d
          nb=%(nb)d gpu=%(gpu)d
          '''%par + ' '
    return awe


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


def awefd2d(odat, owfl, idat, velo, dens, sou, rec, custom, par):
    
    if nGPU > 0: # if gpu's are detected run the gpu code

        Flow([odat, owfl], [idat, velo, sou, rec], '''
            /home/byrne/WORK/CODE/sfAWEFDcart2d
            vel=${SOURCES[1]} sou=${SOURCES[2]} rec=${SOURCES[3]}
            wfl=${TARGETS[1]}
            '''  + ' ' + awepargpu(par) + ' ' + custom)

    # otherwise run the non GPU version
    else:
        Flow([odat,owfl],[idat,velo,dens,sou,rec],
            '''
            awefd2d cden=n
            vel=${SOURCES[1]} den=${SOURCES[2]}
            sou=${SOURCES[3]} rec=${SOURCES[4]}
            wfl=${TARGETS[1]}
            ''' + ' ' + awepar(par) + ' ' + custom)


def awefd3d(odat, owfl, idat, velo, dens, sou, rec, custom, par):

    if nGPU > 0: # if gpu's are detected run the gpu code

        Flow([odat, owfl], [idat, velo, sou, rec], '''
            /home/byrne/WORK/CODE/sfAWEFDcart3d 
            vel=${SOURCES[1]} sou=${SOURCES[2]} rec=${SOURCES[3]} wfl=${TARGETS[1]}
            ''' + ' ' + awepargpu(par) +  ' ' + custom)

    # otherwise run the non GPU version
    else:
        Flow([odat,owfl],[idat,velo,dens,sou,rec],
            '''
            awefd3d cden=n
            vel=${SOURCES[1]} den=${SOURCES[2]}
            sou=${SOURCES[3]} rec=${SOURCES[4]}
            wfl=${TARGETS[1]}
            ''' + ' ' + awepar(par) + ' ' + custom)



# -------------------------------------------------------------------
# Spherical Simulators

def spawefd2d(odat, owfl, idat, velo, dens, sou, rec, custom, par):

    if nGPU > 0: # if gpu's are detected run the gpu code

        Flow([odat, owfl], [idat, velo, sou, rec], '''
            /home/byrne/WORK/CODE/sfAWEFDpolar
            vel=${SOURCES[1]} sou=${SOURCES[2]} rec=${SOURCES[3]}
            wfl=${TARGETS[1]}
            ''' + ' ' + awepargpu(par) + ' ' + custom)

    # otherwise run the non GPU version
    else:
        raise NotImplementedError("CPU Code for 2d spherical acoustic model does not exist")

def spawefd3d(odat, owfl, idat, velo, dens, sou, rec, custom, par):
    
    if nGPU > 0:
        
        Flow([odat, owfl], [idat, velo, sou, rec], '''
        /home/byrne/WORK/CODE/sfAWEFDspher
        vel=${SOURCES[1]} sou=${SOURCES[2]} rec=${SOURCES[3]}
        wfl=${TARGETS[1]}
        ''' + ' ' + awepargpu(par) + ' ' + custom)

    else:
        raise NotImplementedError("CPU code for 3d spherical acoustic model does not exist")
