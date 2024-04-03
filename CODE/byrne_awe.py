import os, sys

# get GPU count
nGPU = int(os.environ.get('nGPU'))
gpu = 1 # default gpu to use

# Import RSF Stuff
try:    from rsf.cluster import *
except: from rsf.proj    import *
import fdmod


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
          nb=%(nb)d nc=%(nc)d gpu=%(gpu)d
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

def spawefd2dgpu(odat, owfl, idat, velo, dens, sou, rec, custom, par):

    # make wavelet
    fdmod.wavelet('wav',par['frq'],par)

    # run acoustic model
    Flow([odat, owfl], ['wav', velo, sou, rec], '''
        /home/byrne/WORK/CODE/sfAWEFDpolar
        jdata=%(jdata)d dabc=%(dabc)s free=%(free)s
        snap=%(snap)s bnds=%(bnds)s jsnap=%(jsnap)d
        nb=%(nb)d nbell=%(nbell)d nc=%(nc)d gpu=%(gpu)d
        vel=${SOURCES[1]} sou=${SOURCES[2]} rec=${SOURCES[3]}
        wfl=${TARGETS[1]}
        ''' % par)

def spawefd2d(odat, owfl, idat, velo, dens, sou, rec, custom, par):

    if nGPU > 0: # if gpu's are detected run the gpu code

        Flow([odat, owfl], [idat, velo, sou, rec], '''
            /home/byrne/WORK/CODE/sfAWEFDpolar
            vel=${SOURCES[1]} sou=${SOURCES[2]} rec=${SOURCES[3]}
            wfl=${TARGETS[1]}
            ''' + ' ' + awepargpu(par) + ' ' + custom)

        #spawefd2dgpu(odat, owfl, idat, velo, dens, sou, rec, custom, par)

    # otherwise run the non GPU version
    else:
        raise NotImplementedError("CPU Code for 2d spherical acoustic model does not exist")

def spawefd3dgpu(odat, owfl, idat, velo, dens, sou, rec, custom, par):

    # make wavelet
    fdmod.wavelet('wav',par['frq'],par)

    # run acoustic model
    Flow([odat, owfl], ['wav', velo, sou, rec], '''
        ~/WORK/CODE/sfAWEFDspher
        jdata=%(jdata)d free=%(free)s
        dabc=%(dabc)s  snap=%(snap)s bnds=%(bnds)s jsnap=%(jsnap)d
        nb=%(nb)d nbell=%(nbell)d nc=%(nc)d gpu=%(gpu)d
        vel=${SOURCES[1]} sou=${SOURCES[2]} rec=${SOURCES[3]}
        wfl=${TARGETS[1]}
        ''' % par)

def spawefd3d(odat, owfl, idat, velo, dens, sou, rec, custom, par):
    
    if nGPU > 0:
        spawefd3dgpu(odat, owfl, idat, velo, dens, sou, rec, custom, par)

    else:
        raise NotImplementedError("CPU code for 3d spherical acoustic model does not exist")
