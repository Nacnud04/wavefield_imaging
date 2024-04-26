try:    from rsf.cluster import *
except: from rsf.proj    import *
try:    from rsf.recipes import pplot
except: import pplot
import math, random
import functools, operator

random.seed(1006)

def add(x,y): return x+y
def myid(n): return '_'+functools.reduce(operator.add,['%d'%random.randint(0,9) for i in range(n)])
def Temp(o,i,r):
    Flow(o,i,r+ ' datapath=%s '%os.environ.get('TMPDATAPATH',os.environ.get('DATAPATH')))

def horizontal(cc,coord,par):
    M8R='$RSFROOT/bin/sf'
    DPT=os.environ.get('TMPDATAPATH',os.environ.get('DATAPATH'))

    cco=cc+'o'+myid(16)
    ccz=cc+'z'+myid(16)
    ccx=cc+'x'+myid(16)

    Flow(cc,None,
         '''
         %smath output=0 n1=%d o1=%g d1=%g >%s datapath=%s/;
         '''%(M8R,par['nx'],par['ox'],par['dx'],cco,DPT) +
         '''
         %smath <%s output="%g" >%s datapath=%s/;
         '''%(M8R,cco,coord,ccz,DPT) +
         '''
         %smath <%s output="x1" >%s datapath=%s/;
         '''%(M8R,cco,ccx,DPT) +
         '''
         %scat axis=2 space=n %s %s | transp | put label1="" unit1="" label2="" unit2="">${TARGETS[0]};
         '''%(M8R,ccz,ccx) +
         '''
         %srm %s %s %s
         '''%(M8R,cco,ccx,ccz),
              stdin=0,
              stdout=0)



def horizontal3d(cc,coord,par):
    Temp(cc+'_',None, 'math n1=%(nx)d d1=%(dx)g o1=%(ox)g n2=%(ny)d d2=%(dy)g o2=%(oy)g output=0' % par)
    Temp(cc+'_z',cc+'_','math output="%g" | put n1=%d n2=%d n3=1' % (coord,par['nx'],par['ny']) )

    if(par['nx']==1):
        Temp(cc+'_x',cc+'_',
             'math output="%g" | put n1=%d n2=%d n3=1' % (par['ox'],par['nx'],par['ny']) )
    else:
        Temp(cc+'_x',cc+'_',
             'math output="x1" | put n1=%d n2=%d n3=1' % (      par['nx'],par['ny']) )

    if(par['ny']==1):
        Temp(cc+'_y',cc+'_',
             'math output="%g" | put n1=%d n2=%d n3=1' % (par['oy'],par['nx'],par['ny']) )
    else:
        Temp(cc+'_y',cc+'_',
             'math output="x2" | put n1=%d n2=%d n3=1' % (          par['nx'],par['ny']) )

    Flow(cc,[cc+'_z',cc+'_x',cc+'_y'],
         '''
         cat axis=3 space=n ${SOURCES[1:3]} |
         transp plane=13 | transp plane=23 |
         put label1="" unit1="" label2="" unit2=""
         ''')

