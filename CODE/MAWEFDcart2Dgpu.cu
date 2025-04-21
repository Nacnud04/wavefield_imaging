#include <cuda.h>
#include <cuda_runtime_api.h>
#include <time.h>

extern "C" {
    #include <rsf.h>
    #include <rsf_su.h>
}

#include "fdutil.c"

#include "cart_kernels.cu"

#define MIN(x, y) (((x) < (y)) ? (x): (y))
#define NOP 4

#ifdef _OPENMP
#include <omp.h>
#include "omputil.h"
#endif

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
    sf_axis at, ax, az, acx, acz; // time, xdius, zeta
    sf_axis as, ar;

    // define dimension sizes
    int nt, nx, nz, ns, nr, ncs, nb;
    int it;
    float dt, dx, dz;
    float ot, ox, oz;

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

    // define gpu to be used
    int gpu;
    if (! sf_getint("gpu", &gpu)) gpu = 0;
    sf_warning("Using gpu #%d", gpu);
    cudaSetDevice(gpu);

    // set up axis
    at  = sf_iaxa(Fwav,2); sf_setlabel(at ,"t" ); // time
    ax = sf_iaxa(Fvel,2); sf_setlabel(ax,"x"); // xdius
    az = sf_iaxa(Fvel,1); sf_setlabel(az,"z"); // zeta

    as  = sf_iaxa(Fsou,2); sf_setlabel(as ,"s" ); // sources
    ar  = sf_iaxa(Frec,2); sf_setlabel(ar ,"r" ); // receivers
    sf_axis ar_3, as_3;
    ar_3 = sf_iaxa(Frec, 3);
    as_3 = sf_iaxa(Fsou, 3);

    nt  = sf_n(at ); dt  = sf_d(at );
    nx = sf_n(ax); dx = sf_d(ax);
    nz = sf_n(az); dz = sf_d(az);
    
    ns  = sf_n(as_3) * sf_n(as);
    nr  = sf_n(ar_3) * sf_n(ar);

    ox = sf_o(ax); oz = sf_o(az); ot = sf_o(at);

    sf_warning("nx:%d|nz:%d|nt:%d|ns:%d|nr:%d",nx,nz,nt,ns,nr);
    sf_warning("dx:%f|dz:%f|dt:%f", dx, dz, dt);
    sf_warning("ox:%f|oz:%f|ot:%f", ox, oz, ot);
    
    // how often to extxct receiver data?
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
        acz = sf_maxa(nz,oz,dz); 
        acx = sf_maxa(nx,ox,dx); 
    
        sf_setn(at, (nt-1)/jsnap+1);
        sf_setd(at,dt*jsnap);
        
        sf_oaxa(Fwfl,acz,1);
        sf_oaxa(Fwfl,acx,2);
        sf_oaxa(Fwfl,at,3);

    }
    

    // init FDM
    fdm = fdutil_init(false, fsrf, az, ax, nb, 1);
    // origin is very slighly different under FDM due to gridsize.
    sf_warning("Adjusted Origins: oz %f, ox %f", fdm->ozpad, fdm->oxpad);
    oz = fdm->ozpad; ox = fdm->oxpad;

    // MOVE SOURCE WAVELET INTO THE GPU
    ncs = 1;
    float *ww = NULL;
    ww = sf_floatalloc(nt); // allocate var for ncs dims over nt time
    sf_floatread(ww, nt, Fwav); // read wavelet into allocated mem

    float *d_ww;
    cudaMalloc((void**)&d_ww, ncs*nt*sizeof(float));
    sf_check_gpu_error("cudaMalloc source wavelet to device");
    cudaMemcpy(d_ww, ww, ncs*nt*sizeof(float), cudaMemcpyHostToDevice);

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

    // x and z coordinates of each source
    int *d_Sjx, *d_Sjz;
    cudaMalloc((void**)&d_Sjx, ns * sizeof(int));
    cudaMalloc((void**)&d_Sjz, ns * sizeof(int));
    sf_check_gpu_error("cudaMalloc source coords to device");

    float *d_Rw00, *d_Rw01, *d_Rw10, *d_Rw11;
    cudaMalloc((void**)&d_Rw00, nr * sizeof(float));
    cudaMalloc((void**)&d_Rw01, nr * sizeof(float));
    cudaMalloc((void**)&d_Rw10, nr * sizeof(float));
    cudaMalloc((void**)&d_Rw11, nr * sizeof(float));
    sf_check_gpu_error("cudaMalloc receiver interpolation coefficients to device");

    // x and z locations of each receiver
    int *d_Rjx, *d_Rjz;
    cudaMalloc((void**)&d_Rjx, nr * sizeof(int));
    cudaMalloc((void**)&d_Rjz, nr * sizeof(int));
    sf_check_gpu_error("cudaMalloc receiver coords to device");

    // allocate memory to import velocity data
    float *tt1 = (float*)malloc(nx * nz * sizeof(float));

    // x, y, z pad to nxpad, nzpad, nphpad
    int nxpad=fdm->nxpad; int nzpad=fdm->nzpad; 
    h_vel = (float*)malloc(nxpad * nzpad * sizeof(float));

    // expand dimensions to allow for absorbing boundary conditions
    sf_warning("Expanding dimensions to allocate for bound. conditions");
    sf_warning("nxpad: %d | nzpad: %d", nxpad, nzpad);
    
    // read in velocity data & expand domain
    sf_floatread(tt1, nx*nz, Fvel);
    expand_cpu_2d(tt1, h_vel, fdm->nb, nz, nzpad, nx, nxpad);
    cudaMalloc((void **)&d_vel, nzpad*nxpad*sizeof(float));
    sf_check_gpu_error("allocated velocity to device");
    cudaMemcpy(d_vel, h_vel, nzpad*nxpad*sizeof(float), cudaMemcpyHostToDevice);
    sf_check_gpu_error("copy velocity to device");

    // CREATE DATA ARRAYS FOR RECEIVERS
    float *d_dd_pp; float *h_dd_pp;
    h_dd_pp = (float*)malloc(nsmp * nr * sizeof(float));
    cudaMalloc((void**)&d_dd_pp, nsmp*nr*sizeof(float));
    sf_check_gpu_error("allocate data arrays");

    // allocate pressure arrays for past, present and future on GPU's
    cudaMalloc((void**)&d_ppo, nzpad*nxpad*sizeof(float));
    cudaMalloc((void**)&d_po , nzpad*nxpad*sizeof(float));
    cudaMalloc((void**)&d_fpo, nzpad*nxpad*sizeof(float));
    h_po = (float*)malloc(nzpad*nxpad*sizeof(float));
    sf_check_gpu_error("allocate pressure arrays");
 
    if (snap){
        oslice = sf_floatalloc2(nz,nx);
        po = sf_floatalloc2(nzpad,nxpad);
    }

    if (bnds){
        sf_setn(acz, nzpad);
        sf_setn(acx, nxpad);
        sf_oaxa(Fwfl,acz,1);
        sf_oaxa(Fwfl,acx,2);
    }

    // SET UP ONE WAY BOUND CONDITIONS
    float *one_bzl = sf_floatalloc(nxpad);
    float *one_bzh = sf_floatalloc(nxpad);
    float *one_bxl = sf_floatalloc(nzpad);
    float *one_bxh = sf_floatalloc(nzpad);

    float d;
    for (int ix=0; ix<nxpad; ix++) {
	d = h_vel[NOP * nxpad + ix] * (dt / dz);
	one_bzl[ix] = (1-d)/(1+d);
	d = h_vel[(nzpad-NOP-1)*nxpad + ix] * (dt / dz);
	one_bzh[ix] = (1-d)/(1+d);
    }
    for (int iz=0; iz<nzpad; iz++) {
        d = h_vel[iz * nxpad + NOP] * (dt / dx);
	one_bxl[iz] = (1-d)/(1+d);
	d = h_vel[iz * nxpad + nxpad-NOP-1] * (dt / dx);
	one_bxh[iz] = (1-d)/(1+d);
    }

    float *d_bzl, *d_bzh, *d_bxl, *d_bxh;
    cudaMalloc((void**)&d_bzl, nxpad*sizeof(float));
    cudaMemcpy(d_bzl, one_bzl, nxpad*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_bzh, nxpad*sizeof(float));
    cudaMemcpy(d_bzh, one_bzh, nxpad*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_bxl, nzpad*sizeof(float));
    cudaMemcpy(d_bxl, one_bxl, nzpad*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_bxh, nzpad*sizeof(float));
    cudaMemcpy(d_bxh, one_bxh, nzpad*sizeof(float), cudaMemcpyHostToDevice);

    // ITERATE OVER SHOTS
    for (int isrc = 0; isrc < 1; isrc ++) {

	// read source and receiver coordinates
	pt2dread1(Fsou, ss, ns, 2);
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

    cudaMemcpy(d_Sjz, cs->jz, ns * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Sjx, cs->jx, ns * sizeof(int), cudaMemcpyHostToDevice);
	sf_check_gpu_error("copy source coords to device");


	// SET RECEIVERS ON THE GPU
	sf_warning("Receiver Count: %d", nr);
	cr = lint2d_make(nr, rr, fdm);

	cudaMemcpy(d_Rw00, cr->w00, nr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rw01, cr->w01, nr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rw10, cr->w10, nr * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rw11, cr->w11, nr * sizeof(float), cudaMemcpyHostToDevice);
    sf_check_gpu_error("copy receiver interpolation coefficients to device");

    cudaMemcpy(d_Rjz, cr->jz, nr * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rjx, cr->jx, nr * sizeof(int), cudaMemcpyHostToDevice);
    sf_check_gpu_error("copy receiver coords to device");


	// set pressure to 0 on gpu
	cudaMemset(d_ppo, 0, nzpad*nxpad*sizeof(float));
	cudaMemset(d_po , 0, nzpad*nxpad*sizeof(float));
	cudaMemset(d_fpo, 0, nzpad*nxpad*sizeof(float));
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
        inject_sources_2D<<<dimGridS,dimBlockS>>>(d_po, d_ww, d_vel,
                       d_Sw00, d_Sw01, d_Sw10, d_Sw11,
                       d_Sjx, d_Sjz, it, ns, fdm->nxpad, fdm->nzpad);
        sf_check_gpu_error("inject sources Kernel");

	    // APPLY WAVE EQUATION
	    dim3 dimGrid2(ceil(nxpad/8.0f),ceil(nzpad/8.0f));
	    dim3 dimBlock2(8,8);
	    solve_2D<<<dimGrid2, dimBlock2>>>(d_fpo, d_po, d_ppo,
			    		  d_vel,
					  dx, dz, dt,
					  nxpad, nzpad);
	    sf_check_gpu_error("solve Kernel");

	    // SHIFT PRESSURE FIELDS IN TIME
	    shift_2D<<<dimGrid2, dimBlock2>>>(d_fpo, d_po, d_ppo,
					   nxpad, nzpad);
	    sf_check_gpu_error("shift Kernel");

	    // ONE WAY BC
        onewayBC_2D<<<dimGrid2, dimBlock2>>>(d_po, d_ppo,
                        d_bzl, d_bzh, d_bxl, d_bxh,
                        nxpad, nzpad);

	    // SPONGE
	    spongeKernel_2D<<<dimGrid2, dimBlock2>>>(d_po, nxpad, nzpad, nb);
	    sf_check_gpu_error("sponge Kernel");
	    spongeKernel_2D<<<dimGrid2, dimBlock2>>>(d_ppo, nxpad, nzpad, nb);
        sf_check_gpu_error("sponge Kernel");

	    // FREE SURFACE
	    freeSurf_2D<<<dimGrid2, dimBlock2>>>(d_po, nxpad, nzpad, nb);
	    sf_check_gpu_error("freeSurf Kernel");

	    if (snap && it%jsnap==0) {

            cudaMemcpy(h_po, d_po, nxpad*nzpad*sizeof(float), cudaMemcpyDefault);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) private(x, z) shared(po, h_po, nxpad)
#endif

            for (int x = 0; x < nxpad; x++) {
                for (int z = 0; z < nzpad; z++) {
                    po[x][z] = h_po[z*nxpad + x];
                }
            }

            if (bnds) {
                sf_floatwrite(po[0], nzpad*nxpad, Fwfl);
            }
            else {

                cut2d(po, oslice, fdm, az, ax);

                sf_floatwrite(oslice[0], sf_n(az)*sf_n(ax), Fwfl);

            }
	    }
	    
	    // EXTRACT TO RECEIVERS
	    dim3 dimGrid3(MIN(nr, ceil(nr/1024.0f)), 1);
	    dim3 dimBlock3(MIN(nr, 1024), 1);

	    extract_2D<<<dimGrid3, dimBlock3>>>(d_dd_pp, it, nr,
			    		     nxpad, nzpad, 
					     d_po, d_Rjx, d_Rjz,
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
    cudaFree(d_Sjx); cudaFree(d_Sjz);
    
    cudaFree(d_Rw00); cudaFree(d_Rw01); cudaFree(d_Rw10); cudaFree(d_Rw11);
    cudaFree(d_Rjx); cudaFree(d_Rjz);

    cudaFree(d_dd_pp);
    cudaFree(d_ppo); cudaFree(d_po); cudaFree(d_fpo);

}

