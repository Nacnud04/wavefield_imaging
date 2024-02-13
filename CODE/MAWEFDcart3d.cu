#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C" {
    #include <rsf.h>
}

#include "fdutil_old.c"
#include "amodelca_kernels.cu"

#define MIN(x, y) (((x) < (y)) ? (x): (y))
#define NOP 4

// funct to check gpu error:
static void sf_check_gpu_error (const char *msg) {
    cudaError_t err = cudaGetLastError ();
    if (cudaSuccess != err)
        sf_error ("Cuda error: %s: %s", msg, cudaGetErrorString (err));
}

// entry
int main(int argc, char*argv[]) {
    
    // define input vars from scons
    bool verb, fsrf, snap, bnds, ssou, dabc;
    int jsnap, jdata;
    
    // define IO files
    sf_file Fwav=NULL; // wavelet
    sf_file Fsou=NULL; // sources
    sf_file Frec=NULL; // receviers
    sf_file Fvel=NULL; // velocity
    sf_file Fdat=NULL; // data
    sf_file Fwfl=NULL; // wavefield    	

    // define axis
    sf_axis at, awt, ax, ay, az, acx, acy, acz;
    sf_axis as, ar; // source, receiver, dimensions
		
    // define dimension sizes
    int nt, nz, ny, nx, ns, nr, ncs, nb;
    int it, ix, iy, iz;
    float dt, dz, dy, dx;

    // FDM structure
    fdm3d fdm=NULL;

    // device and host velocity
    float *h_vel, *d_vel;

    float *h_po, *d_fpo, *d_po, *d_ppo; // pressure
    float ***po=NULL;
    float ***oslice=NULL;

    // linear interpolation weights/indicies
    lint3d cs, cr;

    int nbell; // gaussian bell

    sf_init(argc, argv);

    // exec flags
    if(! sf_getbool("verb",&verb)) verb=false; /* verbosity flag */
    if(! sf_getbool("snap",&snap)) snap=false; /* wavefield snapshots flag */
    if(! sf_getbool("bnds",&bnds)) bnds=false;
    if(! sf_getbool("free",&fsrf)) fsrf=false; /* free surface flag */
    if(! sf_getbool("ssou",&ssou)) ssou=false; /* stress source */
    if(! sf_getbool("dabc",&dabc)) dabc=false; /* absorbing BC */
    sf_warning("verb:%b | snap:%b | free:%b | ssou:%b | dabc:%b",verb,snap,fsrf,ssou,dabc);

    // IO
    Fwav = sf_input ("in" ); /* wavelet   */
    Fvel = sf_input ("vel"); /* stiffness */
    Fsou = sf_input ("sou"); /* sources   */
    Frec = sf_input ("rec"); /* receivers */

    Fdat = sf_output("out"); // data
    Fwfl = sf_output("wfl"); // wavefield
        
    // define gpu to be used
    int gpu;
    if (! sf_getint("gpu", &gpu)) gpu = 0; //gpu id
    sf_warning("using gpu #%d", gpu);
    cudaSetDevice(gpu);

    // set up axis
    at = sf_iaxa(Fwav,1); sf_setlabel(at,"t"); if(verb) sf_raxa(at); //time
    az = sf_iaxa(Fvel,2); sf_setlabel(az,"z"); if(verb) sf_raxa(az); //depth
    ay = sf_iaxa(Fvel,3); sf_setlabel(ay,"y"); if(verb) sf_raxa(ay); //y
    ax = sf_iaxa(Fvel,1); sf_setlabel(ax,"x"); if(verb) sf_raxa(ax); //x

    as = sf_iaxa(Fsou,2); sf_setlabel(as,"s"); if(verb) sf_raxa(as); //sources    
    ar = sf_iaxa(Frec,2); sf_setlabel(ar,"r"); if(verb) sf_raxa(ar); //receivers
    sf_axis ar_3;
    ar_3 = sf_iaxa(Frec, 3);

    awt = at;

    nt = sf_n(at); dt = sf_d(at);
    nz = sf_n(az); dz = sf_d(az);
    ny = sf_n(ay); dy = sf_d(ay);
    nx = sf_n(ax); dx = sf_d(ax);
    ns = sf_n(as);
    nr = sf_n(ar_3) * sf_n(ar);

    sf_warning("nx:%d|ny:%d|nz:%d|nt:%d|ns:%d|nr:%d",nx,ny,nz,nt,ns,nr);
    sf_warning("dx:%f|dy:%f|dz:%f|dt:%f", dx, dy, dz, dt);
    
    // define bell size
    if(! sf_getint("nbell",&nbell)) nbell=5;  //bell size
    sf_warning("nbell=%d",nbell);
    
    // how often to extract receiver data?
    if(! sf_getint("jdata",&jdata)) jdata=1;
    
    if(snap) {

        if(! sf_getint("jsnap",&jsnap)) jsnap=nt;       // save wavefield every jsnap time steps 
    
        sf_warning("extracting recevier data every %d timesteps", jsnap);
        
	acz = sf_maxa(nz, sf_o(az), dz);
        acx = sf_maxa(nx, sf_o(ax), dx);
	acy = sf_maxa(ny, sf_o(ay), dy);

	int ntsnap;
        ntsnap=0;
        for(it=0; it<nt; it++) {
            if(it%jsnap==0) ntsnap++;
        }
        
	sf_warning("therefore there are %d extractions", ntsnap);
   
        sf_setn(awt,ntsnap);
        sf_setd(awt,dt*jsnap);

	sf_oaxa(Fwfl, acz, 1);
	sf_oaxa(Fwfl, acx, 2);
	sf_oaxa(Fwfl, acy, 3);

	sf_oaxa(Fwfl, awt, 4);

    }
    
    // how many time steps in each extraction?
    int nsmp = (nt/jdata);
    if(! sf_getint("jdata",&jdata)) jdata=1;    // extract receiver data every jdata time steps 

    
    // define increase in domain of model for boundary conditions
    if( !sf_getint("nb",&nb) || nb<NOP) nb=NOP;
    
    // init fdm
    fdm=fdutil3d_init(verb,fsrf,az,ax,ay,nb,1);
    sf_warning("ox %f, oy %f, oz %f", fdm->oxpad, fdm->oypad, fdm->ozpad);

    // CREATE GAUSSIAN BELL
    if (nbell * 2 + 1 > 32) {sf_error("nbell must be <= 15\n");}
    float *h_bell, *d_bell;
    h_bell = (float*)malloc((2*nbell+1)*(2*nbell+1)*(2*nbell+1)*sizeof(float));
    float s = 0.5*nbell;
    
    // iterate over bell space
    for (ix=-nbell;ix<=nbell;ix++) {
        for (iy=-nbell;iy<=nbell;iy++) {
            for (iz=-nbell;iz<=nbell;iz++) {
                h_bell[(iy + nbell) * (2*nbell+1) * (2*nbell+1) + (iz + nbell) * (2*nbell+1) + (ix + nbell)] = exp(-(iy*iy+iz*iz+ix*ix)/s);
            }
        }
    }
    
    sf_warning("gauss bell 1d size: %d with dims: x:%d, y:%d, z:%d", (nbell*2) * (2*nbell+1) * (2*nbell+1) + (nbell*2) * (2*nbell+1) + (nbell*2), nbell*2+1, nbell*2+1, nbell*2+1);
    cudaMalloc((void**)&d_bell, (2*nbell+1)*(2*nbell+1)*(2*nbell+1)*sizeof(float));
    sf_check_gpu_error("cudaMalloc d_bell");
    cudaMemcpy(d_bell, h_bell, (2*nbell+1)*(2*nbell+1)*(2*nbell+1)*sizeof(float), cudaMemcpyHostToDevice);
    sf_check_gpu_error("copy d_bell to device");

    // MOVE SOURCE WAVELET INTO GPU
    // for this we basically have to compute the weights of the wavelet to make it correct when on the grid
    ncs = 1;
    float ***ww=NULL;
    ww = sf_floatalloc3(1, ncs, nt); // allocate variable for ncs dimensions over nt time
    sf_floatread(ww[0][0],nt*ncs*1,Fwav); // read the wavelet into the allocated memory

    float *h_ww;
    h_ww = (float*)malloc(1*ncs*nt*sizeof(float));
    for (int t=0; t<nt; t++){
	for (int c=0; c<ncs; c++){
	    h_ww[t * ncs + c] = ww[t][c][0];
	}
    }
    float *d_ww;
    cudaMalloc((void**)&d_ww, 1*ncs*nt*sizeof(float));
    sf_check_gpu_error("cudaMalloc source wavelet to device");
    cudaMemcpy(d_ww, h_ww, 1*ncs*nt*sizeof(float), cudaMemcpyHostToDevice);
    sf_check_gpu_error("copy source wavelet to device");
    
    // SETUP SOURCE/RECEIVER COORDS
    pt3d *ss=NULL;
    pt3d *rr=NULL;

    ss = (pt3d*) sf_alloc(1, sizeof(*ss));
    rr = (pt3d*) sf_alloc(nr, sizeof(*rr));
    
    float *d_Sw000, *d_Sw001, *d_Sw010, *d_Sw011, *d_Sw100, *d_Sw101, *d_Sw110, *d_Sw111;
    cudaMalloc((void**)&d_Sw000, 1 * sizeof(float));
    cudaMalloc((void**)&d_Sw001, 1 * sizeof(float));
    cudaMalloc((void**)&d_Sw010, 1 * sizeof(float));
    cudaMalloc((void**)&d_Sw011, 1 * sizeof(float));
    cudaMalloc((void**)&d_Sw100, 1 * sizeof(float));
    cudaMalloc((void**)&d_Sw101, 1 * sizeof(float));
    cudaMalloc((void**)&d_Sw110, 1 * sizeof(float));
    cudaMalloc((void**)&d_Sw111, 1 * sizeof(float));
    sf_check_gpu_error("cudaMalloc source interpolation coefficients to device");

    // z and x,y coordinates of each source
    int *d_Sjz, *d_Sjx, *d_Sjy;
    cudaMalloc((void**)&d_Sjz, 1 * sizeof(int));
    cudaMalloc((void**)&d_Sjx, 1 * sizeof(int));
    cudaMalloc((void**)&d_Sjy, 1 * sizeof(int));
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

    // z and x coordinates of each receiver
    int *d_Rjz, *d_Rjx, *d_Rjy;
    cudaMalloc((void**)&d_Rjz, nr * sizeof(int));
    cudaMalloc((void**)&d_Rjx, nr * sizeof(int));
    cudaMalloc((void**)&d_Rjy, nr * sizeof(int));
    sf_check_gpu_error("cudaMalloc receiver coords to device");

    // read density and stiffness
    float *tt1 = (float*)malloc(nz * nx * ny * sizeof(float)); // array to transfer data with
    // allocate host stiffness (h_vel)
    h_vel=(float*)malloc(fdm->nzpad * fdm->nxpad * fdm->nypad * sizeof(float));

    sf_warning("Expanding dimensions for boundary conditions to:");
    sf_warning("nxpad: %d | nypad: %d | nzpad: %d", fdm->nxpad, fdm->nypad, fdm->nzpad);
    
    // read and expand velocity
    sf_floatread(tt1,nz*nx*ny,Fvel);
    expand_cpu_3d(tt1, h_vel, fdm->nb, nx, fdm->nxpad, ny, fdm->nypad, nz, fdm->nzpad);
   
    cudaMalloc((void **)&d_vel, fdm->nzpad*fdm->nxpad*fdm->nypad*sizeof(float));
    sf_check_gpu_error("allocated velocity to device");

    cudaMemcpy(d_vel, h_vel, fdm->nzpad * fdm->nxpad * fdm->nypad * sizeof(float), cudaMemcpyHostToDevice);
    sf_check_gpu_error("copy velocity to device");
       
    // CREATE DATA ARRAYS
    float *d_dd_pp; float *h_dd_pp; 

    h_dd_pp = (float*)malloc(nsmp * nr * sizeof(float));
    cudaMalloc((void**)&d_dd_pp, nsmp*nr*sizeof(float));
    
    sf_check_gpu_error("allocate data arrays");

    // allocate grid arrays on GPUs
    cudaMalloc((void **)&d_ppo, fdm->nzpad * fdm->nxpad * fdm->nypad * sizeof(float));
    cudaMalloc((void **)&d_po, fdm->nzpad * fdm->nxpad * fdm->nypad * sizeof(float));
    cudaMalloc((void **)&d_fpo, fdm->nzpad * fdm->nxpad * fdm->nypad * sizeof(float));
    h_po=(float*)malloc(fdm->nzpad * fdm->nxpad * fdm->nypad * sizeof(float));
    sf_check_gpu_error("allocate grid arrays");

    // create array for wavefield
    if (snap) {
        oslice = sf_floatalloc3(sf_n(az), sf_n(ax), sf_n(ay));
        po = sf_floatalloc3(fdm->nzpad, fdm->nxpad, fdm->nypad);
    }

    if (bnds) {
	sf_setn(acz, fdm->nzpad);
	sf_setn(acx, fdm->nxpad);
	sf_setn(acy, fdm->nypad);

	sf_oaxa(Fwfl, acz, 1);
        sf_oaxa(Fwfl, acx, 2);
        sf_oaxa(Fwfl, acy, 3);
    }

    // ITERATE OVER SHOTS
    for (int isrc = 0; isrc < ns; isrc++){

        sf_warning("Modeling shot %d", isrc+1);

	pt3dread1(Fsou, ss, 1, 3); // read source coords
	pt3dread1(Frec, rr, nr, 3); // read receiver coords
	
	// SET SOURCE ON GPU
	sf_warning("Source location: ");
	printpt3d(*ss);

        // do 3d linear interpolation to find source location
	cs = lint3d_make(1, ss, fdm);	

	sf_warning("Source interp coeffs:");
	sf_warning("000:%f | 001:%f | 010:%f | 011:%f | 100:%f | 101:%f | 110:%f | 111:%f", cs->w000[0], cs->w001[0], cs->w010[0], cs->w011[0], cs->w100[0], cs->w101[0], cs->w101[0], cs->w111[0]); 

	cudaMemcpy(d_Sw000, cs->w000, 1 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Sw001, cs->w001, 1 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Sw010, cs->w010, 1 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Sw011, cs->w011, 1 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Sw100, cs->w100, 1 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Sw101, cs->w101, 1 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Sw110, cs->w110, 1 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Sw111, cs->w111, 1 * sizeof(float), cudaMemcpyHostToDevice);
        sf_check_gpu_error("copy source interpolation coefficients to device");

        cudaMemcpy(d_Sjz, cs->jz, 1 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Sjx, cs->jx, 1 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Sjy, cs->jy, 1 * sizeof(int), cudaMemcpyHostToDevice);
        sf_check_gpu_error("copy source coords to device");

	
	// SET RECEIVERS ON GPU
	sf_warning("Receiver Count: %d", nr);
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

        cudaMemcpy(d_Rjz, cr->jz, nr * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Rjx, cr->jx, nr * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Rjy, cr->jy, nr * sizeof(int), cudaMemcpyHostToDevice);
        sf_check_gpu_error("copy receiver coords to device");
	
	
	// set pressure to 0 on gpu
	cudaMemset(d_ppo, 0, fdm->nzpad * fdm->nxpad * fdm->nypad * sizeof(float));
        cudaMemset(d_po, 0, fdm->nzpad * fdm->nxpad * fdm->nypad * sizeof(float));
	cudaMemset(d_fpo, 0, fdm->nzpad * fdm->nxpad * fdm->nypad * sizeof(float));
	sf_check_gpu_error("initialize grid arrays");

	// set data to zero
        cudaMemset(d_dd_pp, 0, nsmp * nr * sizeof(float));
	
        for (int i = 0; i < nsmp * nr; i++){
            h_dd_pp[i] = 0.f;
        }
	sf_warning("nx:%d|ny:%d|nz:%d",fdm->nxpad,fdm->nypad,fdm->nzpad);
	// -= TIME LOOP =-
	if(verb) fprintf(stderr,"\n");
        sf_warning("total number of time steps: %d", nt);
	int itr = 0; int wfnum = 0;
	for (it=0; it<nt; it++) {
	    
	    fprintf(stderr, "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\btime step: %d", it+1);
	    
	    // INJECT STRESS SOURCE
	    if(ssou) {
                
		dim3 dimGrid7(ns, 1, 1);
                dim3 dimBlock7(2 * nbell + 1, 2 * nbell + 1, 1);
                lint3d_bell_gpu<<<dimGrid7, dimBlock7>>>(d_po, d_ww, d_Sw000, d_Sw001, d_Sw010, d_Sw011, d_Sw100, d_Sw101, d_Sw110, d_Sw111, d_bell, d_Sjz, d_Sjy, d_Sjx, it, ncs, 1, 0, nbell, fdm->nxpad, fdm->nzpad);
                sf_check_gpu_error("lint2d_bell_gpu Kernel");

            }
	    dim3 dimGrid4(ceil(fdm->nxpad/8.0f),ceil(fdm->nypad/8.0f),ceil(fdm->nzpad/8.0f));
            dim3 dimBlock4(8,8,8);
	    solve<<<dimGrid4, dimBlock4>>>(d_fpo, d_po, d_ppo,
			    		   d_vel,
					   dx, dy, dz, dt,
					   fdm->nxpad, fdm->nypad, fdm->nzpad);
	    sf_check_gpu_error("solve Kernel");

	    shift<<<dimGrid4, dimBlock4>>>(d_fpo, d_po, d_ppo,
			    		   fdm->nxpad, fdm->nypad, fdm->nzpad);
	    sf_check_gpu_error("shift Kernel");

	    // APPLY FREE SURFACE BOUNDARY CONDITION   
	    dim3 dimGrid3(ceil(fdm->nxpad/8.0f), ceil(fdm->nypad/8.0f), ceil(fdm->nzpad/8.0f));
	    dim3 dimBlock3(8,8,8);
	    freeSurf<<<dimGrid3,dimBlock3>>>(d_po, fdm->nxpad, fdm->nypad, fdm->nzpad, fdm->nb);
	    sf_check_gpu_error("freeSurf Kernel");

	    // APPLY SPONGE BOUNDARY CONDITION
	    spongeKernel<<<dimGrid3, dimBlock3>>>(d_po, fdm->nxpad, fdm->nypad, fdm->nzpad, fdm->nb);
	    sf_check_gpu_error("sponge Kernel");
	    
	    // MOVE DATA TO GPU
	    if (it % jdata == 0) {
		itr += 1;
		dim3 dimGrid_extract(MIN(nr, ceil(nr/1024.0f)), 1, 1);
		dim3 dimBlock_extract(MIN(nr, 1024), 1, 1);
		lint3d_extract_gpu<<<dimGrid_extract, dimBlock_extract>>>(d_dd_pp, 
				                                          itr, nr, fdm->nxpad, fdm->nypad, fdm->nzpad,
									  d_po, d_Rjx, d_Rjy, d_Rjz,
									  d_Rw000, d_Rw001, d_Rw010, d_Rw011,
									  d_Rw100, d_Rw101, d_Rw110, d_Rw111);
		sf_check_gpu_error("lint3d_extract_gpu Kernel");
	    }

	    // EXTRACT WAVEFIELD EVERY JSNAP STEPS
	    if (snap && it % jsnap == 0) {
		
		cudaMemcpy(h_po, d_po, fdm->nxpad*fdm->nypad*fdm->nzpad*sizeof(float), cudaMemcpyDefault);
        	
		for (int x = 0; x < fdm->nxpad; x++){
                    for (int z = 0; z < fdm->nzpad; z++){
			for (int y = 0; y < fdm->nypad; y++) { 
                            po[y][x][z] = h_po[y*fdm->nzpad*fdm->nxpad + z * fdm->nxpad + x];
                        }
		    }
                }
	
		if (bnds){
		    sf_floatwrite(po[0][0], fdm->nzpad*fdm->nxpad*fdm->nypad, Fwfl);
	        } else {	    
		    cut3d(po, oslice, fdm, az, ax, ay);
		    sf_floatwrite(oslice[0][0], sf_n(az)*sf_n(ax)*sf_n(ay), Fwfl);
	        }
	    }
  
	}
/*    
	cudaMemcpy(h_vel, d_po, fdm->nxpad*fdm->nypad*fdm->nzpad*sizeof(float), cudaMemcpyDefault);

        sf_setn(ax, fdm->nxpad);
        sf_setn(ay, fdm->nypad);
        sf_setn(az, fdm->nzpad);
        sf_oaxa(Fdat, ax, 1);
        sf_oaxa(Fdat, ay, 3);
        sf_oaxa(Fdat, az, 2);

        sf_floatwrite(h_vel, fdm->nxpad*fdm->nypad*fdm->nzpad*sizeof(float), Fdat);	
*/

	cudaMemcpy(h_dd_pp, d_dd_pp, nsmp*nr*sizeof(float), cudaMemcpyDefault);
	
	sf_setn(ar, nr);
	sf_setn(at, nsmp);
	sf_setd(at, dt*jdata);

	sf_oaxa(Fdat, at, 2);
	sf_oaxa(Fdat, ar, 1);

	sf_floatwrite(h_dd_pp, nsmp*nr*sizeof(float), Fdat);
        
    }
}
