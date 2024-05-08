#include <rsf.h>
#include "fdutil.c"
#include "spher_utils.c"

#define NOP 4

#ifdef _OPENMP
#include <omp.h>
#include "omputil.h"
#endif

int main(int argc, char*argv[]) {

    // define input variables from sconstruct
    bool fsrf, snap, bnds, dabc;
    int jsnap, jdata;

    // define IO files
    sf_file Fwav=NULL; //wavelet
    sf_file Fsou=NULL; //sources
    sf_file Frec=NULL; //receivers
    sf_file Fvel=NULL; //velocity

    sf_file Fdat=NULL; //data
    sf_file Fwfl=NULL;

    // define axis
    sf_axis at, awt, ara, ath, aph, acra, acth, acph; // time, radius, theta, phi
    sf_axis as, ar;

    // define dimension sizes
    int nt, nra, nth, nph, ns, nr, ncs, nb;
    int it, ira, ith, iph;
    float dt, dra, dth, dph;

    // FDM structure
    fdm3d fdm=NULL;

    // device and host velocity
    float *h_vel, *d_vel;
    
    // pressure
    float *h_fpo, *h_po, *h_ppo; // future, present, past
    
    // vars for wavefield return
    float ***po=NULL;
    float ***oslice=NULL;

    // linear interpolation of weights and indicies
    lint3d cs, cr;

    int nbell; // gaussian bell dims

    sf_init(argc, argv);

    // exec flags
    if(! sf_getbool("free",&fsrf)) fsrf=false; /* free surface flag */
    if(! sf_getbool("snap",&snap)) snap=false;
    if(! sf_getbool("dabc",&dabc)) dabc=false; /* absorbing BC */
    if(! sf_getbool("bnds",&bnds)) bnds=false;
    sf_warning("Free Surface: %b", fsrf);
    sf_warning("Absorbing Boundaries: %b", dabc);
    sf_warning("Saving wavefield? %b", snap);

    // IO
    Fwav = sf_input("in");
    Fvel = sf_input("vel");
    Fsou = sf_input("sou");
    Frec = sf_input("rec");

    Fdat = sf_output("out");
    Fwfl = sf_output("wfl");


    // SET UP AXIS -----------------------------------------------------------------------
    at  = sf_iaxa(Fwav,2); sf_setlabel(at ,"t" ); // time
    ara = sf_iaxa(Fvel,1); sf_setlabel(ara,"ra"); // radius
    ath = sf_iaxa(Fvel,2); sf_setlabel(ath,"th"); // theta
    aph = sf_iaxa(Fvel,3); sf_setlabel(aph,"ph"); // phi

    as  = sf_iaxa(Fsou,2); sf_setlabel(as ,"s" ); // sources
    ar  = sf_iaxa(Frec,2); sf_setlabel(ar ,"r" ); // receivers
    sf_axis ar_3, as_3;
    ar_3 = sf_iaxa(Frec, 3);
    as_3 = sf_iaxa(Fsou, 3);

    awt = at;

    nt  = sf_n(at ); dt  = sf_d(at );
    nra = sf_n(ara); dra = sf_d(ara);
    nth = sf_n(ath); dth = sf_d(ath);
    nph = sf_n(aph); dph = sf_d(aph);
    
    ns  = sf_n(as_3) * sf_n(as);
    nr  = sf_n(ar_3) * sf_n(ar);

    sf_warning("nra:%d|nth:%d|nph:%d|nt:%d|ns:%d|nr:%d",nra,nth,nph,nt,ns,nr);
    sf_warning("dra:%f|dth:%f|dph:%f|dt:%f", dra, dth, dph, dt);

    
    // define padding
    if( !sf_getint("nb",&nb) || nb<NOP) nb=NOP;
    sf_warning("nb: %d", nb);


    // FDM ---------------------------------------------------------------------------
    // FDM is based on Z, X, Y. Not spherical. So we need to convert
    // to spherical. Z=Theta, X=Radius, Y=Phi

    fdm = fdutil3d_init(false, fsrf, ath, ara, aph, nb, 1);
    float oth, ora, oph;
    oth = fdm->ozpad; ora = fdm->oxpad; oph = fdm->oypad;
    sf_warning("oth %f, ora %f, oph %f", fdm->ozpad, fdm->oxpad, fdm->oypad);

    // x, y, z pad to nrapad, nthpad, nphpad
    int nrapad=fdm->nxpad; int nthpad=fdm->nzpad; int nphpad=fdm->nypad;
    sf_warning("nrapad: %d | nthpad: %d | nphpad: %d", nrapad, nthpad, nphpad);
    h_vel = (float*)malloc(nrapad * nthpad * nphpad * sizeof(float));

    // define bell size
    if(! sf_getint("nbell",&nbell)) nbell=5;  //bell size
    sf_warning("nbell=%d",nbell);
    

    // EXTRACTION RATES ------------------------------------------------------------------
    if(! sf_getint("jdata",&jdata)) jdata=1;
    int nsmp = (nt/jdata);
    sf_warning("extracting recevier %d times", nsmp);


    // DEFINE WAVEFIELD OUTPUT ------------------------------------------------------------
    if(snap) {

        if(! sf_getint("jsnap",&jsnap)) jsnap=nt; // save wavefield every nt timesteps

        acth = sf_maxa(nth, sf_o(ath), dth); sf_setlabel(acth,"lat/th (rad)");
        acra = sf_maxa(nra, sf_o(ara), dra); sf_setlabel(acra,"ra (km)"); // radius
        acph = sf_maxa(nph, sf_o(aph), dph); sf_setlabel(acph,"lon/ph (rad)"); 

        int ntsnap = 0;
        for (it=0; it<nt; it++) {
            if(it%jsnap==0) ntsnap++;
        }

        sf_warning("There are %d wavefield extractions", ntsnap);

        sf_setn(awt, ntsnap);
        sf_setd(awt, dt*jsnap);

        if (bnds) {
            
            sf_warning("Eextracting boundary conditions set to true");
            sf_setn(acth, nthpad);
            sf_setn(acra, nrapad);
            sf_setn(acph, nphpad);    
            
        }

        sf_oaxa(Fwfl, acth, 2);
        sf_oaxa(Fwfl, acra, 1);
        sf_oaxa(Fwfl, acph, 3);

        sf_oaxa(Fwfl, awt, 4);

        // define stuff for wavefield chopping
        oslice = sf_floatalloc3(sf_n(ara), sf_n(ath), sf_n(aph));
        po = sf_floatalloc3(nrapad, nthpad, nphpad);

    }


    // READ IN SOURCE WAVELET -------------------------------------------------------------------
    ncs = 1;
    float *ww = NULL;
    ww = sf_floatalloc(nt); // allocate var for ncs dims over nt time
    sf_floatread(ww, nt, Fwav); // read wavelet into allocated mem

    float *h_ww;
    h_ww = (float*)malloc(1 * ncs * nt*sizeof(float));
    for (int t = 0; t < nt; t++) { 
        h_ww[t] = ww[t];
    }


    // READ AND EXPAND VELOCITY ARRAY -------------------------------------------------------

    // allocate memory to read in velocities
    float *tt1 = (float*)malloc(nra * nth * nph * sizeof(float));
    sf_floatread(tt1, nra*nth*nph, Fvel); // read in data

    sf_warning("Expanding dimensions to allocate for bound. conditions");
    sf_warning("nrapad: %d | nthpad: %d | nphpad: %d", nrapad, nthpad, nphpad);

    // expand domain
    expand_cpu_3d(tt1, h_vel, fdm->nb, nra, nrapad, nph, nphpad, nth, nthpad);


    // CREATE DATA ARRAYS --------------------------------------------------------------------
    float *h_dd_pp;
    h_dd_pp = (float*)malloc(nsmp * nr * sizeof(float));

    // do same for wavefield arrays
    h_po  = (float*)malloc(nthpad*nrapad*nphpad*sizeof(float));
    h_fpo = (float*)malloc(nthpad*nrapad*nphpad*sizeof(float));
    h_ppo = (float*)malloc(nthpad*nrapad*nphpad*sizeof(float));


    // SET UP SOURCE COORDS AND FDM ---------------------------------------------------------
    pt3d *ss = NULL;
    ss = (pt3d*) sf_alloc(ns, sizeof(*ss)); // allocate memory

    pt3dread1(Fsou, ss, ns, 3); // read source coords
    cs = lint3d_make(ns, ss, fdm); // calc source weights


    // SET UP RECEIVER COORDS AND FDM -------------------------------------------------------
    pt3d *rr = NULL;
    rr = (pt3d*) sf_alloc(nr, sizeof(*rr));

    pt3dread1(Frec, rr, nr, 3); // read receiver coords
    cr = lint3d_make(nr, rr, fdm); // calc receiver weights


    // SET UP ONE WAY BOUNDARY CONDITIONS -----------------------------------------------------
    sf_warning("Defining one way BCs...");

    float *one_bthl = sf_floatalloc(nrapad * nphpad);
    float *one_bthh = sf_floatalloc(nrapad * nphpad);
    float *one_bral = sf_floatalloc(nthpad * nphpad);
    float *one_brah = sf_floatalloc(nthpad * nphpad);
    float *one_bphl = sf_floatalloc(nrapad * nthpad);
    float *one_bphh = sf_floatalloc(nrapad * nthpad);

    float d;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) private(ira, iph, d) shared(h_vel, nrapad, nthpad, dt, dth, one_bthl, one_bthh)
#endif
    for (ira=0; ira<nrapad; ira++) {
        for (iph=0; iph<nphpad; iph++) {
            d = h_vel[iph*nrapad*nthpad + NOP*nrapad + ira] * (dt / dth);
            one_bthl[iph*nrapad+ira] = (1-d)/(1+d);
            d = h_vel[iph*nrapad*nthpad + (nthpad-NOP-1)*nrapad + ira] * (dt / dth);
            one_bthh[iph*nrapad+ira] = (1-d)/(1+d);
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) private(ith, iph, d) shared(h_vel, nrapad, nthpad, dt, dra, one_bral, one_brah)
#endif
    for (ith=0; ith<nthpad; ith++) {
        for (iph=0; iph<nphpad; iph++) {
            d = h_vel[iph*nrapad*nthpad + ith*nrapad + NOP] * (dt / dra);
            one_bral[iph*nthpad+ith] = (1-d)/(1+d);
            d = h_vel[iph*nrapad*nthpad + ith*nrapad + nrapad-NOP-1] * (dt / dra);
            one_brah[iph*nthpad+ith] = (1-d)/(1+d);
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) private(ith, ira, d) shared(h_vel, nrapad, nthpad, dt, dph, one_bphl, one_bphh)
#endif
    for (ith=0; ith<nthpad; ith++) {
        for (ira=0; ira<nrapad; ira++) {
            d = h_vel[NOP*nrapad*nthpad + ith*nrapad + ira] * (dt / dph);
            one_bphl[ith*nrapad+ira] = (1-d)/(1+d);
            d = h_vel[(nthpad-NOP-1)*nrapad*nthpad + ith*nrapad + ira] * (dt / dph);
            one_bphh[ith*nrapad+ira] = (1-d)/(1+d);
        }
    }


    // SET PRESSURE AND DATA TO 0 TO START ----------------------------------------------
    sf_warning("Cleaning Data Array memspace...");
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) private(ira, ith, iph) shared(h_po, nrapad, nthpad)
#endif

    for (ira=0; ira < nrapad; ira++) {
        for (ith=0; ith < nthpad; ith++) {
            for (iph=0; iph < nphpad; iph++) {
                h_po[iph * nrapad * nthpad + ith * nrapad + ira] = 0;
            }
        }
    }


    int i;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) private(i) shared(nsmp, nr, h_dd_pp)
#endif

    for (i=0; i < nsmp*nr; i++) {
	    h_dd_pp[i] = 0.f;
	}


    // TIME LOOP -----------------------------------------------------------------------------
    fprintf(stderr,"total num of time steps: %d \n", nt);
	for (it=0; it<801; it++) {

        fprintf(stderr, "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\btime step: %d", it+1);

        // inject pressure source
        inject_sources_3D(h_po, h_ww, 
                       cs->w000, cs->w001, cs->w010, cs->w011, cs->w100, cs->w101, cs->w110, cs->w111,
                       cs->jx, cs->jy, cs->jz,
                       it, ns, nrapad, nphpad, nthpad);

        // solve wave equation
        solve_3D(h_fpo, h_po, h_ppo, h_vel,
              dra, dph, dth, ora, oph, oth, dt,
              nrapad, nphpad, nthpad);

        // shift pressure fields
        shift_3D(h_fpo, h_po, h_ppo, nrapad, nphpad, nthpad);

        // one way boundary conditions
        onewayBC_3D(h_po, h_ppo, one_bthl, 
                 one_bthh, one_brah, one_bral, one_bphl, one_bphh,
                 nrapad, nphpad, nthpad);

        // sponge
        spongeBC_3D(h_po, nrapad, nphpad, nthpad, nb);
        spongeBC_3D(h_ppo, nrapad, nphpad, nthpad, nb);

        // free surface
        if (fsrf) {
            freeBC_3D(h_po, nrapad, nphpad, nthpad, nb);
        }

        // extract data to receivers
        extract_3D(h_dd_pp, it, nr, nrapad, nphpad, nthpad, h_po,
                cr->jx, cr->jy, cr->jz, 
                cr->w000, cr->w001, cr->w010, cr->w011, cr->w100, cr->w101, cr->w110, cr->w111);

        // extract wavefield every jsnap timesteps
        if (snap && it % jsnap == 0) {

            if (bnds) {
                sf_floatwrite(h_po, nthpad*nrapad*nphpad, Fwfl);
            } else {
                cut3d(po, oslice, fdm, ath, ara, aph);
                sf_floatwrite(oslice[0][0], sf_n(ath)*sf_n(ara)*sf_n(aph), Fwfl);
            }


        }

    }

    sf_setn(ar, nr);
    sf_setn(at, nsmp);
    sf_setd(at, dt*jdata);

    sf_oaxa(Fdat, at, 2);
    sf_oaxa(Fdat, ar, 1);

    sf_floatwrite(h_dd_pp, nsmp*nr, Fdat);

}