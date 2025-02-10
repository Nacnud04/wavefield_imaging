#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C" {
    #include <rsf.h>
    #include <rsf_su.h>
}

#include "fdutil.c"

#include "spher_kernels.cu"

#define MIN(x, y) (((x) < (y)) ? (x): (y))
#define NOP 4

// funct to check gpu error
static void sf_check_gpu_error (const char *msg) {
    cudaError_t err = cudaGetLastError ();
    if (cudaSuccess != err)
	sf_error ("Cuda error: %s: %s", msg, cudaGetErrorString (err));
}

// entry
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
    int it;
    float dt, dra, dth, dph;

    // FDM structure
    fdm3d fdm=NULL;

    // device and host velocity
    float *h_vel, *d_vel;
    
    // pressure
    float *d_fpo, *d_po, *d_ppo; // future, present, past
    
    // vars for wavefield return
    float *h_po;
    float ***po=NULL;
    float ***oslice=NULL;

    // linear interpolation of weights and indicies
    lint3d cs, cr;

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

    // define gpu to be used
    int gpu;
    if (! sf_getint("gpu", &gpu)) gpu = 0;
    sf_warning("Using gpu #%d", gpu);
    cudaSetDevice(gpu);

    // set up axis
    at  = sf_iaxa(Fwav,2); sf_setlabel(at ,"t" ); // time
    ara = sf_iaxa(Fvel,1); sf_setlabel(ara,"ra"); // radius
    ath = sf_iaxa(Fvel,2); sf_setlabel(ath,"th"); // theta
    aph = sf_iaxa(Fvel,3); sf_setlabel(aph,"ph"); // phi

    as  = sf_iaxa(Fsou,2); sf_setlabel(as ,"s" ); // sources
    ar  = sf_iaxa(Frec,2); sf_setlabel(ar ,"r" ); // receivers

    awt = at;

    nt  = sf_n(at ); dt  = sf_d(at );
    nra = sf_n(ara); dra = sf_d(ara);
    nth = sf_n(ath); dth = sf_d(ath);
    nph = sf_n(aph); dph = sf_d(aph);
    
    ns  = sf_n(as);
    nr  = sf_n(ar);

    sf_warning("nra:%d|nth:%d|nph:%d|nt:%d|ns:%d|nr:%d",nra,nth,nph,nt,ns,nr);
    sf_warning("dra:%f|dth:%f|dph:%f|dt:%f", dra, dth, dph, dt);

    
    // define increase in domain of model for boundary conditions
    if( !sf_getint("nb",&nb) || nb<NOP) nb=NOP;

    // init FDM
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
    
    // how often to extract receiver data?
    if(! sf_getint("jdata",&jdata)) jdata=1;
    int nsmp = (nt/jdata);
    sf_warning("reading receiver data %d times", nsmp);

    sf_warning("nb: %d", nb);
    
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

            sf_setn(acth, nthpad);
            sf_setn(acra, nrapad);
            sf_setn(acph, nphpad);    
            
        }

        sf_oaxa(Fwfl, acth, 2);
        sf_oaxa(Fwfl, acra, 1);
        sf_oaxa(Fwfl, acph, 3);

        sf_oaxa(Fwfl, awt, 4);

    }

    // MOVE SOURCE WAVELET INTO THE GPU
    ncs = 1;
    float *ww = NULL;
    ww = sf_floatalloc(nt); // allocate var for ncs dims over nt time
    sf_floatread(ww, nt, Fwav); // read wavelet into allocated mem

    float *h_ww;
    h_ww = (float*)malloc(nt*sizeof(float));
    for (int t = 0; t < nt; t++) {
        h_ww[t] = ww[t];
    }

    float *d_ww;
    cudaMalloc((void**)&d_ww, ncs*nt*sizeof(float));
    sf_check_gpu_error("cudaMalloc source wavelet to device");
    cudaMemcpy(d_ww, h_ww, ncs*nt*sizeof(float), cudaMemcpyHostToDevice);

    // SET UP SOURCE / RECEIVER COORDS
    pt3d *ss=NULL;
    pt3d *rr=NULL;

    ss = (pt3d*) sf_alloc(ns, sizeof(*ss));
    rr = (pt3d*) sf_alloc(nr, sizeof(*rr));

    float *d_Sw000, *d_Sw001, *d_Sw010, *d_Sw011, *d_Sw100, *d_Sw101, *d_Sw110, *d_Sw111;
    cudaMalloc((void**)&d_Sw000, ns * sizeof(float));
    cudaMalloc((void**)&d_Sw001, ns * sizeof(float));
    cudaMalloc((void**)&d_Sw010, ns * sizeof(float));
    cudaMalloc((void**)&d_Sw011, ns * sizeof(float));
    cudaMalloc((void**)&d_Sw100, ns * sizeof(float));
    cudaMalloc((void**)&d_Sw101, ns * sizeof(float));
    cudaMalloc((void**)&d_Sw110, ns * sizeof(float));
    cudaMalloc((void**)&d_Sw111, ns * sizeof(float));
    sf_check_gpu_error("cudaMalloc source interpolation coefficients to device");

    // radal and theta, phi coordinates of each source
    int *d_Sjra, *d_Sjth, *d_Sjph;
    cudaMalloc((void**)&d_Sjra, ns * sizeof(int));
    cudaMalloc((void**)&d_Sjth, ns * sizeof(int));
    cudaMalloc((void**)&d_Sjph, ns * sizeof(int));
    sf_check_gpu_error("cudaMalloc source coords to device");

    float *d_Rw000, *d_Rw001, *d_Rw010, *d_Rw011, *d_Rw100, *d_Rw101, *d_Rw110, *d_Rw111;
    cudaMalloc((void**)&d_Rw000, nr * sizeof(float));
    cudaMalloc((void**)&d_Rw001, nr * sizeof(float));
    cudaMalloc((void**)&d_Rw010, nr * sizeof(float));
    cudaMalloc((void**)&d_Rw011, nr * sizeof(float));
    cudaMalloc((void**)&d_Rw100, nr * sizeof(float));
    cudaMalloc((void**)&d_Rw101, nr * sizeof(float));
    cudaMalloc((void**)&d_Rw110, nr * sizeof(float));
    cudaMalloc((void**)&d_Rw111, nr * sizeof(float));
    sf_check_gpu_error("cudaMalloc receiver interpolation coefficients to device");

    // radial, theta, and phi locations of each receiver
    int *d_Rjra, *d_Rjth, *d_Rjph;
    cudaMalloc((void**)&d_Rjra, nr * sizeof(int));
    cudaMalloc((void**)&d_Rjth, nr * sizeof(int));
    cudaMalloc((void**)&d_Rjph, nr * sizeof(int));
    sf_check_gpu_error("cudaMalloc receiver coords to device");

    // allocate memory to import velocity data
    float *tt1 = (float*)malloc(nra * nth * nph * sizeof(float));
    
    // read in velocity data & expand domain
    sf_floatread(tt1, nra*nth*nph, Fvel);
    expand_cpu_3D(tt1, h_vel, fdm->nb, nra, nrapad, nph, nphpad, nth, nthpad);
    cudaMalloc((void **)&d_vel, nthpad*nrapad*nphpad*sizeof(float));
    sf_check_gpu_error("allocated velocity to device");
    cudaMemcpy(d_vel, h_vel, nthpad*nrapad*nphpad*sizeof(float), cudaMemcpyHostToDevice);
    sf_check_gpu_error("copy velocity to device");

    // CREATE DATA ARRAYS FOR RECEIVERS
    float *d_dd_pp; float *h_dd_pp;
    h_dd_pp = (float*)malloc(nsmp * nr * sizeof(float));
    cudaMalloc((void**)&d_dd_pp, nsmp * nr * sizeof(float));
    sf_check_gpu_error("allocate data arrays");

    // allocate pressure arrays for past, present and future on GPU's
    cudaMalloc((void**)&d_ppo, nthpad*nphpad*nrapad*sizeof(float));
    cudaMalloc((void**)&d_po , nthpad*nphpad*nrapad*sizeof(float));
    cudaMalloc((void**)&d_fpo, nthpad*nphpad*nrapad*sizeof(float));
    h_po=(float*)malloc(nthpad * nrapad * nphpad * sizeof(float));
    sf_check_gpu_error("allocate pressure arrays");
    
    if (snap) {

        oslice = sf_floatalloc3(sf_n(ara), sf_n(ath), sf_n(aph));
        po = sf_floatalloc3(nrapad, nthpad, nphpad);
    
    }
    
    // SET UP ONE WAY BOUND CONDITIONS
    float *one_bthl = sf_floatalloc(nrapad * nphpad);
    float *one_bthh = sf_floatalloc(nrapad * nphpad);
    float *one_bral = sf_floatalloc(nthpad * nphpad);
    float *one_brah = sf_floatalloc(nthpad * nphpad);
    float *one_bphl = sf_floatalloc(nrapad * nthpad);
    float *one_bphh = sf_floatalloc(nrapad * nthpad);

    float d;
    for (int ira=0; ira<nrapad; ira++) {
        for (int iph=0; iph<nphpad; iph++) {
            d = h_vel[iph*nrapad*nthpad + NOP*nrapad + ira] * (dt / dth);
            one_bthl[iph*nrapad+ira] = (1-d)/(1+d);
            d = h_vel[iph*nrapad*nthpad + (nthpad-NOP-1)*nrapad + ira] * (dt / dth);
            one_bthh[iph*nrapad+ira] = (1-d)/(1+d);
        }
    }
    for (int ith=0; ith<nthpad; ith++) {
        for (int iph=0; iph<nphpad; iph++) {
            d = h_vel[iph*nrapad*nthpad + ith*nrapad + NOP] * (dt / dra);
            one_bral[iph*nthpad+ith] = (1-d)/(1+d);
            d = h_vel[iph*nrapad*nthpad + ith*nrapad + nrapad-NOP-1] * (dt / dra);
            one_brah[iph*nthpad+ith] = (1-d)/(1+d);
        }
    }
    for (int ith=0; ith<nthpad; ith++) {
        for (int ira=0; ira<nrapad; ira++) {
            d = h_vel[NOP*nrapad*nthpad + ith*nrapad + ira] * (dt / dph);
            one_bphl[ith*nrapad+ira] = (1-d)/(1+d);
            d = h_vel[(nthpad-NOP-1)*nrapad*nthpad + ith*nrapad + ira] * (dt / dph);
            one_bphh[ith*nrapad+ira] = (1-d)/(1+d);
        }
    }
    
    float *d_bthl, *d_bthh, *d_bral, *d_brah, *d_bphl, *d_bphh;
    cudaMalloc((void**)&d_bthl, nrapad*nphpad*sizeof(float));
    cudaMemcpy(d_bthl, one_bthl, nrapad*nphpad*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_bthh, nrapad*nphpad*sizeof(float));
    cudaMemcpy(d_bthh, one_bthh, nrapad*nphpad*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_bral, nthpad*nphpad*sizeof(float));
    cudaMemcpy(d_bral, one_bral, nthpad*nphpad*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_brah, nthpad*nphpad*sizeof(float));
    cudaMemcpy(d_brah, one_brah, nthpad*nphpad*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_bphl, nthpad*nrapad*sizeof(float));
    cudaMemcpy(d_bphl, one_bphl, nthpad*nrapad*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_bphh, nthpad*nrapad*sizeof(float));
    cudaMemcpy(d_bphh, one_bphh, nthpad*nrapad*sizeof(float), cudaMemcpyHostToDevice);
    sf_check_gpu_error("copy one way bc conditions to device");

    // ITERATE OVER SHOTS
    // CURRENTLY ONLY 1 TO SET ALL SHOTS AT ONCE
    for (int isrc = 0; isrc < 1; isrc ++) {

	// read source and receiver coordinates
	// in the pt3d struct there is X, Y and Z. The same convention is
	// used here to transform into spherical coordinates (X:Radius,
	// Y:Phi, Z:Theta)
	pt3dread1(Fsou, ss, ns, 3);
	pt3dread1(Frec, rr, nr, 3);

	// SET SOURCES ON GPU

	// perform 3d linear interpolation on source
	cs = lint3d_make(ns, ss, fdm);

    cudaMemcpy(d_Sw000, cs->w000, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sw001, cs->w001, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sw010, cs->w010, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sw011, cs->w011, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sw100, cs->w100, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sw101, cs->w101, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sw110, cs->w110, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sw111, cs->w111, ns * sizeof(float), cudaMemcpyHostToDevice);
    sf_check_gpu_error("copy source interpolation coefficients to device");

    cudaMemcpy(d_Sjth, cs->jz, ns * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Sjra, cs->jx, ns * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Sjph, cs->jy, ns * sizeof(int), cudaMemcpyHostToDevice);
	sf_check_gpu_error("copy source coords to device");

	// SET RECEIVERS ON THE GPU
	cr = lint3d_make(nr, rr, fdm);

	cudaMemcpy(d_Rw000, cr->w000, nr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rw001, cr->w001, nr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rw010, cr->w010, nr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rw011, cr->w011, nr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rw100, cr->w100, nr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rw101, cr->w101, nr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rw110, cr->w110, nr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rw111, cr->w111, nr * sizeof(float), cudaMemcpyHostToDevice);
    sf_check_gpu_error("copy receiver interpolation coefficients to device");

    cudaMemcpy(d_Rjth, cr->jz, nr * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rjra, cr->jx, nr * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rjph, cr->jy, nr * sizeof(int), cudaMemcpyHostToDevice);
    sf_check_gpu_error("copy receiver coords to device");


	// set pressure to 0 on gpu
	cudaMemset(d_ppo, 0, nthpad*nphpad*nrapad*sizeof(float));
	cudaMemset(d_po , 0, nthpad*nphpad*nrapad*sizeof(float));
	cudaMemset(d_fpo, 0, nthpad*nphpad*nrapad*sizeof(float));
	sf_check_gpu_error("Set pressure arrays to 0");

	// set receiver data to 0
	cudaMemset(d_dd_pp, 0, nsmp*nr*sizeof(float));

	for (int i=0; i < nsmp*nr; i++) {
	    h_dd_pp[i] = 0.f;
	}

	// TIME LOOP
	fprintf(stderr,"total num of time steps: %d \n", nt);
	for (it=0; it<nt; it++) {

	    fprintf(stderr, "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\btime step: %d", it+1);

	    // INJECT PRESSURE SOURCE
        
        dim3 dimGridS(MIN(ns, ceil(ns/1024.0f)), 1, 1);
	    dim3 dimBlockS(MIN(ns, 1024), 1, 1);
        inject_sources_3D<<<dimGridS, dimBlockS>>>(d_po, d_ww, 
			    d_Sw000, d_Sw001, d_Sw010, d_Sw011, 
			    d_Sw100, d_Sw101, d_Sw110, d_Sw111, 
			    d_Sjra, d_Sjph, d_Sjth, 
			    it, ns, nrapad, nphpad, nthpad);
        sf_check_gpu_error("inject_sources Kernel");

	    dim3 dimGrid2(ceil(nrapad/8.0f),ceil(nphpad/8.0f),ceil(nthpad/8.0f));
	    dim3 dimBlock2(8,8,8);
        
	    // APPLY WAVE EQUATION
	    solve_3D<<<dimGrid2, dimBlock2>>>(d_fpo, d_po, d_ppo,
                d_vel,
                dra, dph, dth, ora, oph, oth, dt,
                nrapad, nphpad, nthpad);
	    sf_check_gpu_error("solve Kernel");

	    // SHIFT PRESSURE FIELDS IN TIME
	    shift_3D<<<dimGrid2, dimBlock2>>>(d_fpo, d_po, d_ppo,
                nrapad, nphpad, nthpad);
	    sf_check_gpu_error("shift Kernel");

	    // ONE WAY BC
	    onewayBC_3D<<<dimGrid2,dimBlock2>>>(d_po, d_ppo,
                d_bthl, d_bthh, d_bral, d_brah, d_bphl, d_bphh,
                nrapad, nphpad, nthpad);

	    // SPONGE
	    spongeKernel_3D<<<dimGrid2, dimBlock2>>>(d_po, nrapad, nphpad, nthpad, nb);
	    sf_check_gpu_error("sponge Kernel1");
	    spongeKernel_3D<<<dimGrid2, dimBlock2>>>(d_ppo, nrapad, nphpad, nthpad, nb);
        sf_check_gpu_error("sponge Kernel2");

	    // FREE SURFACE
        if (fsrf) {
            freeSurf_3D<<<dimGrid2, dimBlock2>>>(d_po, nrapad, nphpad, nthpad, nb);
            sf_check_gpu_error("free surface Kernel");
        }
		
	    // RECEIVERS
	    dim3 dimGridR(MIN(nr, ceil(nr/1024.0f)), 1, 1);
	    dim3 dimBlockR(MIN(nr, 1024), 1, 1);
	    extract_3D<<<dimGridR, dimBlockR>>>(d_dd_pp, it, nr,
                nrapad, nphpad, nthpad, 
                d_po, d_Rjra, d_Rjph, d_Rjth,
                d_Rw000, d_Rw001, d_Rw010, d_Rw011,
                d_Rw100, d_Rw101, d_Rw110, d_Rw111);
        sf_check_gpu_error("lint3d_extract_gpu Kernel");

	    // EXTRACT WAVEFIELD EVERY JSNAP STEPS
	    if (snap && it % jsnap == 0) {

            cudaMemcpy(h_po, d_po, nrapad * nphpad * nthpad * sizeof(float), cudaMemcpyDefault);

            if (bnds) {
                sf_floatwrite(h_po, nthpad*nrapad*nphpad, Fwfl);
            } else {
                cut3d(po, oslice, fdm, ath, ara, aph);
                sf_floatwrite(oslice[0][0], sf_n(ath)*sf_n(ara)*sf_n(aph), Fwfl);
            }
            
	    }	    

	}

    }

    fprintf(stderr,"\n");

    cudaMemcpy(h_dd_pp, d_dd_pp, nsmp*nr*sizeof(float), cudaMemcpyDefault);

    sf_setn(ar, nr);
    sf_setn(at, nsmp);
    sf_setd(at, dt*jdata);

    sf_oaxa(Fdat, at, 2);
    sf_oaxa(Fdat, ar, 1);

    sf_floatwrite(h_dd_pp, nsmp*nr, Fdat);

    // FREE GPU MEMORY

    cudaFree(d_ww);

    cudaFree(d_Sw000); cudaFree(d_Sw001); cudaFree(d_Sw010); cudaFree(d_Sw011);
    cudaFree(d_Sw100); cudaFree(d_Sw101); cudaFree(d_Sw110); cudaFree(d_Sw111);
    cudaFree(d_Sjra); cudaFree(d_Sjth); cudaFree(d_Sjph);
    
    cudaFree(d_Rw000); cudaFree(d_Rw001); cudaFree(d_Rw010); cudaFree(d_Rw011);
    cudaFree(d_Rw100); cudaFree(d_Rw101); cudaFree(d_Rw110); cudaFree(d_Rw111);
    cudaFree(d_Rjra); cudaFree(d_Rjth); cudaFree(d_Rjph);

    cudaFree(d_dd_pp);
    cudaFree(d_ppo); cudaFree(d_po); cudaFree(d_fpo);

}

