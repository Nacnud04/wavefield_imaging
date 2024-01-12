#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C" {
    #include <rsf.h>
}

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
    bool verb, fsrf, snap, ssour, dabc;
    int jsnap, ntsnap, jdata;
    
    // define IO files
    sf_file Fwav=NULL; // wavelet
    sf_file Fsou=NULL; // sources
    sf_file Frec=NULL; // receviers
    sf_file Fvel=NULL; // velocity
    sf_file Fden=NULL; // density
    sf_file Ftmp=NULL; // temp
		      
    // define axis
    sf_axis at, ax, ay, az;
    sf_axis as, ar, ac; // source, receiver, dimensions
		
    // define dimension sizes
    int nt, nz, ny, nx, ns, nr, nc, ncs, nb;
    int it, iz, iy, iz;
    int dt, dz, dy, dx, idz, idy, idx;

    // FDM structure
    fdm3d fdm=NULL;

    // device and host velocity
    float *h_vel, *d_vel;

    // device and host density
    float *h_ro, *d_ro;

    // device particle velocity
    float *d_uoz, *d_uoy, *d_uox;
    float *d_po; // pressure

    // linear interpolation weights/indicies
    lint3d cs, cr;

    int nbell; // gaussian bell

    sf_init(argc, argv);
