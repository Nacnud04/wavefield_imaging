try:    from rsf.cluster import *
except: from rsf.proj    import *
try:    from rsf.recipes import pplot
except: import pplot
import math, random
import functools, operator

random.seed(1006)

def add(x,y): return x+y
def myid(n): return '_'+functools.reduce(operator.add,['%d'%random.randint(0,9) for i in range(n)])

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
