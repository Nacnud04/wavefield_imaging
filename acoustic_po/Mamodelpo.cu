#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C" {
    #include <rsf.h>
}

#include "fdutil.c"
#include "emodel3d_kernels.cu"

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
    bool verb, frsf, snap, ssou, dabc;
    int jsnap, jdata;

    // define IO files
    sf_file Fwav=NULL; //wavelet
    sf_file Fsou=NULL; //source
    sf_file Frec=NULL; //receivers
    sf_file Fvel=NULL; //velocity
    sf_file Fden=NULL; //density
    sf_file Fdat=NULL; //data

}

