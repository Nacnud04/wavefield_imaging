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
    int it;
    float dt, dra, dth;
    float ot, ora, oth;

    // FDM structure
    fdm2d fdm=NULL;

    // device and host velocity
    float *h_vel, *d_vel;
    // pressure
    float *h_po, *d_fpo, *d_po, *d_ppo; // future, present, past
    
    float **oslice=NULL;
    float **po=NULL;

    // linear interpolation of weights and indicies
    lint2d cs, cr;

    sf_init(argc, argv);

    // exec flags
    if(! sf_getbool("free",&fsrf)) fsrf=false; /* free surface flag */
    if(! sf_getbool("dabc",&dabc)) dabc=false; /* absorbing BC */
    if(! sf_getbool("snap",&snap)) snap=true;
    if(! sf_getbool("bnds",&bnds)) bnds=false;
    sf_warning("Free Surface: %b", fsrf);
    sf_warning("Absorbing Boundaries: %b", dabc);

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

    // how often to extract receiver data?
    if(! sf_getint("jdata",&jdata)) jdata=1;

    // how many time steps in each extraction?
    int nsmp = (nt/jdata);
    sf_warning("reading receiver data %d times", nsmp);

    if(! sf_getint("jdata",&jdata)) jdata=1;    // extract receiver data every jdata time steps

    // define increase in domain of model for boundary conditions
    if( !sf_getint("nb",&nb) || nb<NOP) nb=NOP;
    
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
        
        sf_oaxa(Fwfl,acth,1);
        sf_oaxa(Fwfl,acra,2);
        sf_oaxa(Fwfl,at,3);

    }
    

    // init FDM
    // FDM is based on Z, X, Not polar. So we need to convert
    // to spherical. Z=Theta, X=Radius
    fdm = fdutil_init(false, fsrf, ath, ara, nb, 1);
    // origin is very slighly different under FDM due to gridsize.
    sf_warning("Adjusted Origins: oth %f, ora %f", fdm->ozpad, fdm->oxpad);
    oth = fdm->ozpad; ora = fdm->oxpad;

    // MOVE SOURCE WAVELET INTO THE GPU
    ncs = 1;
    float *ww = NULL;
    ww = sf_floatalloc(nt); // allocate var for ncs dims over nt time
    sf_floatread(ww, nt, Fwav); // read wavelet into allocated mem

    float *h_ww;
    h_ww = (float*)malloc(1 * ncs * nt*sizeof(float));
    for (int t = 0; t < nt; t++) { 
        h_ww[t] = ww[t];
    }

    float *d_ww;
    cudaMalloc((void**)&d_ww, 1*ncs*nt*sizeof(float));
    sf_check_gpu_error("cudaMalloc source wavelet to device");
    cudaMemcpy(d_ww, h_ww, 1*ncs*nt*sizeof(float), cudaMemcpyHostToDevice);

    // SET UP SOURCE / RECEIVER COORDS
    pt2d *ss=NULL;
    pt2d *rr=NULL;

    ss = (pt2d*) sf_alloc(ns, sizeof(*ss));
    rr = (pt2d*) sf_alloc(nr, sizeof(*rr));

    float *d_Sw00, *d_Sw01, *d_Sw10, *d_Sw11;
    cudaMalloc((void**)&d_Sw00, ns * sizeof(float));
    cudaMalloc((void**)&d_Sw01, ns * sizeof(float));
    cudaMalloc((void**)&d_Sw10, ns * sizeof(float));
    cudaMalloc((void**)&d_Sw11, ns * sizeof(float));
    sf_check_gpu_error("cudaMalloc source interpolation coefficients to device");

    // radal and theta, phi coordinates of each source
    int *d_Sjra, *d_Sjth;
    cudaMalloc((void**)&d_Sjra, ns * sizeof(int));
    cudaMalloc((void**)&d_Sjth, ns * sizeof(int));
    sf_check_gpu_error("cudaMalloc source coords to device");

    float *d_Rw00, *d_Rw01, *d_Rw10, *d_Rw11;
    cudaMalloc((void**)&d_Rw00, nr * sizeof(float));
    cudaMalloc((void**)&d_Rw01, nr * sizeof(float));
    cudaMalloc((void**)&d_Rw10, nr * sizeof(float));
    cudaMalloc((void**)&d_Rw11, nr * sizeof(float));
    sf_check_gpu_error("cudaMalloc receiver interpolation coefficients to device");

    // radial, theta, and phi locations of each receiver
    int *d_Rjra, *d_Rjth;
    cudaMalloc((void**)&d_Rjra, nr * sizeof(int));
    cudaMalloc((void**)&d_Rjth, nr * sizeof(int));
    sf_check_gpu_error("cudaMalloc receiver coords to device");

    // allocate memory to import velocity data
    float *tt1 = (float*)malloc(nra * nth * sizeof(float));

    // x, y, z pad to nrapad, nthpad, nphpad
    int nrapad=fdm->nxpad; int nthpad=fdm->nzpad; 
    h_vel = (float*)malloc(nrapad * nthpad * sizeof(float));

    // expand dimensions to allow for absorbing boundary conditions
    sf_warning("Expanding dimensions to allocate for bound. conditions");
    sf_warning("nrapad: %d | nthpad: %d", nrapad, nthpad);
    
    // read in velocity data & expand domain
    sf_floatread(tt1, nra*nth, Fvel);
    expand_cpu_2d(tt1, h_vel, fdm->nb, nra, nrapad, nth, nthpad);
    cudaMalloc((void **)&d_vel, nthpad*nrapad*sizeof(float));
    sf_check_gpu_error("allocated velocity to device");
    cudaMemcpy(d_vel, h_vel, nthpad*nrapad*sizeof(float), cudaMemcpyHostToDevice);
    sf_check_gpu_error("copy velocity to device");

    // CREATE DATA ARRAYS FOR RECEIVERS
    float *d_dd_pp; float *h_dd_pp;
    h_dd_pp = (float*)malloc(nsmp * nr * sizeof(float));
    cudaMalloc((void**)&d_dd_pp, nsmp*nr*sizeof(float));
    sf_check_gpu_error("allocate data arrays");

    // allocate pressure arrays for past, present and future on GPU's
    cudaMalloc((void**)&d_ppo, nthpad*nrapad*sizeof(float));
    cudaMalloc((void**)&d_po , nthpad*nrapad*sizeof(float));
    cudaMalloc((void**)&d_fpo, nthpad*nrapad*sizeof(float));
    h_po = (float*)malloc(nthpad*nrapad*sizeof(float));
    sf_check_gpu_error("allocate pressure arrays");
 
    if (snap){
        oslice = sf_floatalloc2(nth,nra);
        po = sf_floatalloc2(nthpad,nrapad);
    }

    if (bnds){
        sf_setn(acth, nthpad);
        sf_setn(acra, nrapad);
        sf_oaxa(Fwfl,acth,1);
        sf_oaxa(Fwfl,acra,2);
    }

    // SET UP ONE WAY BC's
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

    float *d_bthl, *d_bthh, *d_bral, *d_brah;
    cudaMalloc((void**)&d_bthl, nrapad*sizeof(float));
    cudaMemcpy(d_bthl, one_bthl, nrapad*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_bthh, nrapad*sizeof(float));
    cudaMemcpy(d_bthh, one_bthh, nrapad*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_bral, nthpad*sizeof(float));
    cudaMemcpy(d_bral, one_bral, nthpad*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_brah, nthpad*sizeof(float));
    cudaMemcpy(d_brah, one_brah, nthpad*sizeof(float), cudaMemcpyHostToDevice);

    // ITERATE OVER SHOTS
    for (int isrc = 0; isrc < 1; isrc ++) {

	// read source and receiver coordinates
	// in the pt struct there is X and Z. The same convention is
	// used here to transform into spherical coordinates (X:Radius,
	// Z:Theta)
	pt2dread1(Fsou, ss, ns , 2);
	pt2dread1(Frec, rr, nr, 2);

	// set source on GPU
	sf_warning("Source location: ");
	printpt2d(*ss);

	// perform 3d linear interpolation on source
	cs = lint2d_make(ns, ss, fdm);

	sf_warning("Source interp coeffs:");
    sf_warning("00:%f | 01:%f | 10:%f | 11:%f", cs->w00[0], cs->w01[0], cs->w10[0], cs->w11[0]);

    cudaMemcpy(d_Sw00, cs->w00, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sw01, cs->w01, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sw10, cs->w10, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sw11, cs->w11, ns * sizeof(float), cudaMemcpyHostToDevice);
    sf_check_gpu_error("copy source interpolation coefficients to device");

    cudaMemcpy(d_Sjth, cs->jz, ns * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Sjra, cs->jx, ns * sizeof(int), cudaMemcpyHostToDevice);
	sf_check_gpu_error("copy source coords to device");


	// SET RECEIVERS ON THE GPU
	sf_warning("Receiver Count: %d", nr);
	cr = lint2d_make(nr, rr, fdm);

	cudaMemcpy(d_Rw00, cr->w00, nr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rw01, cr->w01, nr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rw10, cr->w10, nr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rw11, cr->w11, nr * sizeof(float), cudaMemcpyHostToDevice);
    sf_check_gpu_error("copy receiver interpolation coefficients to device");

    cudaMemcpy(d_Rjth, cr->jz, nr * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rjra, cr->jx, nr * sizeof(int), cudaMemcpyHostToDevice);
    sf_check_gpu_error("copy receiver coords to device");


	// set pressure to 0 on gpu
	cudaMemset(d_ppo, 0, nthpad*nrapad*sizeof(float));
	cudaMemset(d_po , 0, nthpad*nrapad*sizeof(float));
	cudaMemset(d_fpo, 0, nthpad*nrapad*sizeof(float));  
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
        dim3 dimGridS(MIN(ns, ceil(ns/1024.0f)), 1);
        dim3 dimBlockS(MIN(ns, 1024), 1);
        inject_sources_2D<<<dimGridS,dimBlockS>>>(d_po, d_ww, 
                       d_Sw00, d_Sw01, d_Sw10, d_Sw11,
                       d_Sjra, d_Sjth, it, ns, nrapad, nthpad);
        sf_check_gpu_error("inject sources Kernel");

	    // APPLY WAVE EQUATION
	    dim3 dimGrid2(ceil(nrapad/8.0f),ceil(nthpad/8.0f));
	    dim3 dimBlock2(8,8);
	    solve_2D<<<dimGrid2, dimBlock2>>>(d_fpo, d_po, d_ppo,
			    		  d_vel,
					  dra, dth, ora, oth, dt,
					  nrapad, nthpad);
	    sf_check_gpu_error("solve Kernel");
	    
	    // SHIFT PRESSURE FIELDS IN TIME
	    shift_2D<<<dimGrid2, dimBlock2>>>(d_fpo, d_po, d_ppo,
					   nrapad, nthpad);
	    sf_check_gpu_error("shift Kernel");

	    // ONE WAY BC
	    onewayBC_2D<<<dimGrid2, dimBlock2>>>(d_po, d_ppo,
			                      d_bthl, d_bthh, d_bral, d_brah,
					      nrapad, nthpad);
	    
	    // SPONGE
	    spongeKernel_2D<<<dimGrid2, dimBlock2>>>(d_po, nrapad, nthpad, nb);
	    sf_check_gpu_error("sponge Kernel");
	    spongeKernel_2D<<<dimGrid2, dimBlock2>>>(d_ppo, nrapad, nthpad, nb);
        sf_check_gpu_error("sponge Kernel");

	    // FREE SURFACE
        if (fsrf) {
            freeSurf_2D<<<dimGrid2, dimBlock2>>>(d_po, nrapad, nthpad, nb);
            sf_check_gpu_error("freeSurf Kernel");
        }

	    if (snap && it%jsnap==0) {

            cudaMemcpy(h_po, d_po, nrapad*nthpad*sizeof(float), cudaMemcpyDefault);

            for (int ra = 0; ra < nrapad; ra++) {
                for (int th = 0; th < nthpad; th++) {
                    po[ra][th] = h_po[th*nrapad + ra];
                }
            }	

            if (bnds) {
                sf_floatwrite(po[0], nthpad*nrapad, Fwfl);
            }
            else {
                cut2d(po, oslice, fdm, ath, ara);
                sf_floatwrite(oslice[0], sf_n(ath)*sf_n(ara), Fwfl);
            }
	    }
	    
	    // EXTRACT TO RECEIVERS
	    dim3 dimGrid3(MIN(nr, ceil(nr/1024.0f)), 1);
	    dim3 dimBlock3(MIN(nr, 1024), 1);

	    extract_2D<<<dimGrid3, dimBlock3>>>(d_dd_pp, it, nr,
			    		     nrapad, nthpad, 
					     d_po, d_Rjra, d_Rjth,
					     d_Rw00, d_Rw01, d_Rw10, d_Rw11);
	    sf_check_gpu_error("extract Kernel");

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

    // FREE ALLOCATED MEMORY
    cudaFree(d_ww);

    cudaFree(d_Sw00); cudaFree(d_Sw01); cudaFree(d_Sw10); cudaFree(d_Sw11);
    cudaFree(d_Sjra); cudaFree(d_Sjth);
    
    cudaFree(d_Rw00); cudaFree(d_Rw01); cudaFree(d_Rw10); cudaFree(d_Rw11);
    cudaFree(d_Rjra); cudaFree(d_Rjth);

    cudaFree(d_dd_pp);
    cudaFree(d_ppo); cudaFree(d_po); cudaFree(d_fpo);

}

