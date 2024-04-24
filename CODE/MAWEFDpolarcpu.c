#include <rsf.h>

#include "fdutil_old.c"
#include "cpupolarutils.c"

#define NOP 4

#ifdef _OPENMP
#include <omp.h>
#include "omputil.h"
#endif

// entry
int main(int argc, char*argv[]) {

    // define input variables from sconstruct
    bool fsrf, snap, bnds, dabc;
    int jsnap, jdata;

    // define IO files
    sf_file Fwav=NULL; //wavelet
    sf_file Fsou=NULL; //source
    sf_file Frec=NULL; //receivers
    sf_file Fvel=NULL; //velocity
    sf_file Fdat=NULL; //data
    sf_file Fwfl=NULL;

    // define axis
    sf_axis at, ara, ath, acra, acth; // time, radius, theta
    sf_axis as, ar;

    // define dimension sizes
    int nt, nra, nth, ns, nr, ncs, nb;
    int it, ira, ith;
    float dt, dra, dth;
    float ot, ora, oth;

    // FDM structure
    fdm2d fdm=NULL;

    // device and host velocity
    float *h_vel, *d_vel;
    // pressure
    float *h_po, *h_fpo, *h_ppo; // future, present, past
    
    float **oslice=NULL;
    float **po=NULL;

    // linear interpolation of weights and indicies
    lint2d cs, cr;

    sf_init(argc, argv);

    // exec flags
    if(! sf_getbool("free",&fsrf)) fsrf=false; /* free surface flag */
    if(! sf_getbool("dabc",&dabc)) dabc=false; /* absorbing BC */
    if(! sf_getbool("snap",&snap)) snap=true;
    if(! sf_getbool("bnds",&bnds)) bnds=true;
    sf_warning("Free Surface: %b", fsrf);
    sf_warning("Absorbing Boundaries: %b", dabc);

    // IO
    Fwav = sf_input("in");
    Fvel = sf_input("vel");
    Fsou = sf_input("sou");
    Frec = sf_input("rec");
    Fdat = sf_output("out");
    Fwfl = sf_output("wfl");



    // SET UP AXIS --------------------------------------------------------------------------------

    at  = sf_iaxa(Fwav,2); sf_setlabel(at ,"t" ); // time
    ara = sf_iaxa(Fvel,1); sf_setlabel(ara,"ra"); // radius
    ath = sf_iaxa(Fvel,2); sf_setlabel(ath,"th"); // theta

    as  = sf_iaxa(Fsou,2); sf_setlabel(as ,"s" ); // sources
    ar  = sf_iaxa(Frec,2); sf_setlabel(ar ,"r" ); // receivers
    sf_axis ar_3, as_3;
    ar_3 = sf_iaxa(Frec, 3);
    as_3 = sf_iaxa(Fsou, 3);

    nt  = sf_n(at ); dt  = sf_d(at );
    nra = sf_n(ara); dra = sf_d(ara);
    nth = sf_n(ath); dth = sf_d(ath);
    
    ns  = sf_n(as_3) * sf_n(as);
    nr  = sf_n(ar_3) * sf_n(ar);

    ora = sf_o(ara); oth = sf_o(ath); ot = sf_o(at);

    sf_warning("nra:%d|nth:%d|nt:%d|ns:%d|nr:%d",nra,nth,nt,ns,nr);
    sf_warning("dra:%f|dth:%f|dt:%f", dra, dth, dt);
    sf_warning("ora:%f|oth:%f|ot:%f", ora, oth, ot);



    // EXTRACTION RATES --------------------------------------------------------------------------------

    if(! sf_getint("jdata",&jdata)) jdata=1;
    sf_warning("extracting recevier data every %d times", jdata);

    // how many time steps in each extraction?
    int nsmp = (nt/jdata);
    sf_warning("therefore there are %d timesteps between extraction", nsmp);

    if(! sf_getint("jdata",&jdata)) jdata=1;    // extract receiver data every jdata time steps



    // define padding

    if( !sf_getint("nb",&nb) || nb<NOP) nb=NOP;
    
    // INITIALIZE FDM --------------------------------------------------------------------------
    // FDM is based on Z, X, Not polar. So we need to convert
    // to spherical. Z=Theta, X=Radius
    fdm = fdutil_init(false, fsrf, ath, ara, nb, 1);

    // origin is very slighly different under FDM due to gridsize.
    sf_warning("Adjusted Origins: oth %f, ora %f", fdm->ozpad, fdm->oxpad);
    oth = fdm->ozpad; ora = fdm->oxpad;

    // from x and z to ra and th
    int nrapad=fdm->nxpad; int nthpad=fdm->nzpad; 


    // READ SOURCE WAVELET --------------------------------------------------------------------
    ncs = 1;
    float *ww = NULL;
    ww = sf_floatalloc(nt); // allocate var for ncs dims over nt time
    sf_floatread(ww, nt, Fwav); // read wavelet into allocated mem

    float *h_ww;
    h_ww = (float*)malloc(1 * ncs * nt*sizeof(float));
    for (int t = 0; t < nt; t++) { 
        if (t < 0.5 * nt) {h_ww[t] = ww[t];}
        if (t > 0.5 * nt) {h_ww[t] = 0;} // temporary hack to remove later
    }

    // SET UP SOURCE AND RECEIVER COORDS ----------------------------------------------------------

    pt2d *ss=NULL;
    pt2d *rr=NULL;

    ss = (pt2d*) sf_alloc(ns, sizeof(*ss));
    rr = (pt2d*) sf_alloc(nr, sizeof(*rr));

    // SET UP VELOCITY ARRAY ------------------------------------------------------------------------

    float *tt1 = (float*)malloc(nra * nth * sizeof(float)); // velocity array for read in data
    h_vel = (float*)malloc(nrapad * nthpad * sizeof(float)); // velocity arrat for expanded domain

    sf_floatread(tt1, nra*nth, Fvel); // read data from file

    // expand to padded domain
    expand_cpu_2d(tt1, h_vel, fdm->nb, nra, nrapad, nth, nthpad);
    
    // SET UP DATA ARRAYS ----------------------------------------------------------------------
    float *h_dd_pp;
    h_dd_pp = (float*)malloc(nsmp * nr * sizeof(float));

    // SET UP PRESSURE ARRAYS ---------------------------------------------------------------------
    h_po  = (float*)malloc(nthpad*nrapad*sizeof(float));
    h_fpo = (float*)malloc(nthpad*nrapad*sizeof(float));
    h_ppo = (float*)malloc(nthpad*nrapad*sizeof(float));

    // SET UP WAVEFIELD EXTRACTION --------------------------------------------------------------
    if(snap) {

        if(! sf_getint("jsnap",&jsnap)) jsnap=nt;       // save wavefield every jsnap time steps

	    sf_warning("Jsnap: %d", jsnap);
        acth = sf_maxa(nth,oth,dth); 
        acra = sf_maxa(nra,ora,dra); 
    
        int ntsnap = 0;
	    for (it=0; it<nt; it++) {
	        if (it%jsnap==0) ntsnap++;
	    }

	    sf_setn(at,ntsnap);
        sf_setd(at,dt*jsnap);

        if (bnds) {
            sf_setn(acth, nthpad);
            sf_setn(acra, nrapad);
        }
        
        sf_oaxa(Fwfl,acth,1);
        sf_oaxa(Fwfl,acra,2);
        sf_oaxa(Fwfl,at,3);

        // stuff for wavefield extraction
        oslice = sf_floatalloc2(nth, nra);
        po     = sf_floatalloc2(nthpad, nrapad);

    }

    // SET UP ONE WAY BC's --------------------------------------------------------------------------------------
    float *one_bthl = sf_floatalloc(nrapad);
    float *one_bthh = sf_floatalloc(nrapad);
    float *one_bral = sf_floatalloc(nthpad);
    float *one_brah = sf_floatalloc(nthpad);

    float d;
    for (int ira=0; ira<nrapad; ira++) {
        d = h_vel[NOP * nrapad + ira] * (dt / dth);
	    one_bthl[ira] = (1-d)/(1+d);
	    d = h_vel[(nthpad-NOP-1)*nrapad + ira] * (dt / dth);
	    one_bthh[ira] = (1-d)/(1+d);
    }
    for (int ith=0; ith<nthpad; ith++) {
	    d = h_vel[ith * nrapad + NOP] * (dt / dra);
	    one_bral[ith] = (1-d)/(1+d);
	    d = h_vel[ith * nrapad + nrapad-NOP-1] * (dt / dra);
	    one_brah[ith] = (1-d)/(1+d);
    }

    // READ SOURCES ------------------------------------------------------
    pt2dread1(Fsou, ss, ns, 2); // read in sources

    sf_warning("Source location: ");
	printpt2d(*ss);

    // do 2d interpolation on source
    cs = lint2d_make(ns, ss, fdm);

    sf_warning("Source interp coeffs:");
    sf_warning("00:%f | 01:%f | 10:%f | 11:%f", cs->w00[0], cs->w01[0], cs->w10[0], cs->w11[0]);

    // READ RECEIVERS ------------------------------------------------------------------------------------------
    pt2dread1(Frec, rr, nr, 2); // read in receivers

    sf_warning("Receiver Count: %d", nr);
	cr = lint2d_make(nr, rr, fdm);

    // SET PRESSURE ARRAY AND DATA ARRAY TO 0 TO START ------------------------------------------------------------

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) private(ira, ith) shared(h_po, nrapad, nthpad)
#endif

    for (int ira=0; ira < nrapad; ira++) {
        for (int ith=0; ith < nthpad; ith++) {
            h_po[ith * nrapad + ira] = 0;
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) private(ira, ith) shared(h_po, nrapad, nthpad)
#endif

    for (int i=0; i < nsmp*nr; i++) {
	    h_dd_pp[i] = 0.f;
	}

    // TIME LOOP! ---------------------------------------------------------------------------------
    fprintf(stderr, "total number of time steps: %d \n", nt);

    for (it=0; it<nt; it++) {

        fprintf(stderr, "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\btime step: %d", it+1);

        // inject sources
        inject_sources(h_po, h_ww, cs->w00, cs->w01, cs->w10, cs->w11,
                       cs->jx, cs->jz, it, ns, nrapad, nthpad);

        // solve for next time step
        solve(h_fpo, h_po, h_ppo, h_vel, dra, dth, ora, oth, dt, nrapad, nthpad);

        // shift pressure fields in time
        shift(h_fpo, h_po, h_ppo, nrapad, nthpad);

        // apply one way boundary conditions
        onewayBC(h_po, h_ppo, one_bthl, one_bthh, one_bral, one_brah, nrapad, nthpad);

        // apply sponge twice
        spongeBC(h_po, nrapad, nthpad, nb);
        spongeBC(h_ppo, nrapad, nthpad, nb);

        // apply free surface if needed
        if (fsrf) {
            freeSurf(h_po, nrapad, nthpad, nb);
        }

        // extract data to receivers
        extract(h_dd_pp, it, nr, nrapad, nthpad,
                h_po, cr->jx, cr->jz, cr->w00, cr->w01, cr->w10, cr->w11);

        if (snap && it % jsnap == 0) {

            // this is quick enough it has not been multithreaded.
            for (int ra = 0; ra < nrapad; ra++) {
                for (int th = 0; th < nthpad; th++) {
                    po[ra][th] = h_po[th*nrapad + ra];
                }
            }

            if (bnds) {
                sf_floatwrite(po[0], nthpad*nrapad, Fwfl);
            } else {
                cut2d(po, oslice, fdm, ath, ara);
                sf_floatwrite(oslice[0], sf_n(ath)*sf_n(ara), Fwfl);
            }

        }

    }

    fprintf(stderr,"\n");

    // write data out
    sf_setn(ar, nr);
    sf_setn(at, nsmp);
    sf_setd(at, dt*jdata);
    sf_oaxa(Fdat, at, 2);
    sf_oaxa(Fdat, ar, 1);
    sf_floatwrite(h_dd_pp, nsmp*nr, Fdat); 

}