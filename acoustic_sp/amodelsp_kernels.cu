#include <stdio.h>
#include <math.h>

// macro for 1d index to simulate a 3d matrix
#define INDEX3D(ix, iy, iz, nx, nz) ((ix)+(iz)*(nx)+(iy)*(nz)*(nx))

void expand_cpu_3d(float *a, float *b, int nb, int x_a, int x_b, int y_a, int y_b, int z_a, int z_b){

        // copy into other array
	for (int ix = 0; ix < x_a; ix++) {
                for (int iy = 0; iy < y_a; iy++) {
                        for (int iz = 0; iz < z_a; iz++) {
                                b[INDEX3D(ix+nb,iy+nb,iz+nb,z_b,x_b)] = a[INDEX3D(ix,iy,iz,x_a,z_a)];
                        }
                }
        }
        // expand z direction
        
	for (int ix = 0; ix < x_b; ix++) {
                for (int iy = 0; iy < y_b; iy++) {
                        for (int iz = 0; iz < nb; iz++) {
                                b[INDEX3D(ix,iy,iz,x_b,z_b)] = b[INDEX3D(ix,iy,nb,x_b,z_b)];
                                b[INDEX3D(ix,iy,z_b-iz-1,x_b,z_b)] = b[INDEX3D(ix,iy,z_b-nb-1,x_b,z_b)];
                        }
                }
        }
	
        // expand y direction
        
	for (int ix = 0; ix < x_b; ix++) {
                for (int iy = 0; iy < nb; iy++) {
                        for (int iz = 0; iz < z_b; iz++) {
                                b[INDEX3D(ix,iy,iz,x_b,z_b)] = b[INDEX3D(ix,nb,iz,x_b,z_b)];
                                b[INDEX3D(ix,y_b-iy-1,iz,x_b,z_b)] = b[INDEX3D(ix,y_b-nb-1,iz,x_b,z_b)];
                        }
                }
        }
	
        //expand x direction
        
	for (int ix = 0; ix < nb; ix++) {
                for (int iy = 0; iy < y_b; iy++) {
                        for (int iz = 0; iz < z_b; iz++) {
                                b[INDEX3D(ix,iy,iz,x_b,z_b)] = b[INDEX3D(nb,iy,iz,x_b,z_b)];
                                b[INDEX3D(x_b-ix-1,iy,iz,x_b,z_b)] = b[INDEX3D(x_b-nb-1,iy,iz,x_b,z_b)];
                        }
                }
        }
	
}

__global__ void lint3d_bell_gpu(float *d_uu, float *d_ww, float *d_Sw000, float *d_Sw001, float *d_Sw010, float *d_Sw011, float *d_Sw100, float *d_Sw101, float *d_Sw110, float *d_Sw111, float *d_bell, int *d_jx, int *d_jy, int *d_jz, int it, int nc, int ns, int c, int nbell, int nxpad, int nzpad) {

        int ix = threadIdx.x;
        int iy = threadIdx.y;
        int iz = threadIdx.z;
        int ia = blockIdx.x;

        float wa = d_ww[it * nc * ns + c * ns + ia] * d_bell[(iy * (2*nbell+1)*(2*nbell+1)) + (iz * (2*nbell+1)) + ix];

        int y_comp = (d_jy[ia] - nbell) + iy;
	int z_comp = (d_jz[ia] - nbell) + iz;
	int x_comp = (d_jx[ia] - nbell) + ix;
	int xz = nxpad * nzpad;

        atomicAdd(&d_uu[(y_comp)     * xz + (z_comp)     * nxpad + (x_comp    )], ((wa * d_Sw000[ia])));
        atomicAdd(&d_uu[(y_comp)     * xz + (z_comp + 1) * nxpad + (x_comp    )], ((wa * d_Sw001[ia])));
        atomicAdd(&d_uu[(y_comp)     * xz + (z_comp)     * nxpad + (x_comp + 1)], ((wa * d_Sw010[ia])));
        atomicAdd(&d_uu[(y_comp)     * xz + (z_comp + 1) * nxpad + (x_comp + 1)], ((wa * d_Sw011[ia])));
        atomicAdd(&d_uu[(y_comp + 1) * xz + (z_comp)     * nxpad + (x_comp    )], ((wa * d_Sw100[ia])));
        atomicAdd(&d_uu[(y_comp + 1) * xz + (z_comp + 1) * nxpad + (x_comp    )], ((wa * d_Sw101[ia])));
        atomicAdd(&d_uu[(y_comp + 1) * xz + (z_comp)     * nxpad + (x_comp + 1)], ((wa * d_Sw110[ia])));
        atomicAdd(&d_uu[(y_comp + 1) * xz + (z_comp + 1) * nxpad + (x_comp + 1)], ((wa * d_Sw111[ia])));

}


// divergence 3d for cpml
#define NOP 4 // half of the order in space

__global__ void solve(float *d_fpo, float *d_po, float *d_ppo, float *d_vel,
		      float dra, float dph, float dth, float ora, float oph, float oth, 
		      float dt,
		      int nrapad, int nthpad, int nphpad) {

	int ira = threadIdx.x + blockIdx.x * blockDim.x;
	int iph = threadIdx.y + blockIdx.y * blockDim.y;
	int ith = threadIdx.z + blockIdx.z * blockDim.z;

	if (ira < nrapad && ith < nthpad && iph < nphpad){
		
		int globalAddr = iph * nthpad * nrapad + ith * nrapad + ira;			  float laplace;

		// extract true location from deltas and indicies
		float ra; float ph; float th;
		ra = dra * ira + ora;
		ph = dph * iph + oph;
		th = dth * ith + oth;
		
		// extract true velocity
		float v;
		v  = d_vel[globalAddr];

		// perform only in boundaries:
		if (ira >= NOP && ira < nrapad-NOP && iph >= NOP && iph < nphpad-NOP && ith >= NOP && ith < nthpad - NOP) {

			// CALCULATE ALL SPATIAL DERIVS IN LAPLACIAN
			float pra; 
			pra =  d_po[INDEX3D(ira+1, iph, ith, nrapad, nthpad)] \
			      -d_po[INDEX3D(ira-1, iph, ith, nrapad, nthpad)];
			pra = pra / (2 * dra);

			float ppra;
			ppra =    d_po[INDEX3D(ira+1, iph, ith,nrapad,nthpad)] \
                               -2*d_po[INDEX3D(ira  , iph, ith,nrapad,nthpad)] \
                                 +d_po[INDEX3D(ira-1, iph, ith,nrapad,nthpad)];
			ppra = ppra / (dra * dra);

			float pth;
			pth =  d_po[INDEX3D(ira, iph, ith+1, nrapad, nthpad)] \
                              -d_po[INDEX3D(ira, iph, ith-1, nrapad, nthpad)];
                        pth = pth / (2 * dth);

			float ppth;
			ppth =    d_po[INDEX3D(ira, iph, ith+1,nrapad,nthpad)] \
                               -2*d_po[INDEX3D(ira, iph, ith  ,nrapad,nthpad)] \
                                 +d_po[INDEX3D(ira, iph, ith-1,nrapad,nthpad)];
                        ppth = ppth / (dth * dth);

			float ppph;
                        ppph =    d_po[INDEX3D(ira, iph+1, ith,nrapad,nthpad)] \
                               -2*d_po[INDEX3D(ira, iph  , ith,nrapad,nthpad)] \
                                 +d_po[INDEX3D(ira, iph-1, ith,nrapad,nthpad)];
                        ppph = ppph / (dph * dph);

			// COMBINE SPATIAL DERIVS TO CREATE LAPLACIAN
			laplace =  (2/ra)*pra + ppra \              // ra component
				    +(cos(th)/(ra*ra*sin(th)))*pth \  // th component 1
				    +(1/(ra*ra))*ppth \               // th component 2
				    +(1/(ra*ra*sin(th)*sin(th)))*ppph;// ph component

		} else {
			laplace = 0.;
		}

		// compute pressure at next time step
		d_fpo[globalAddr] = (dt*dt) * (v*v) * laplace + 2*d_po[globalAddr] - d_ppo[globalAddr];


	}

}


__global__ void shift(float *d_fpo, float *d_po, float *d_ppo,
		      int nrapad, int nphpad, int nthpad) {
	
	int ira = threadIdx.x + blockIdx.x * blockDim.x;
        int iph = threadIdx.y + blockIdx.y * blockDim.y;
        int ith = threadIdx.z + blockIdx.z * blockDim.z;

	if (ira < nrapad && iph < nphpad && ith < nthpad){

		int globalAddr = iph * nthpad * nrapad + ith * nrapad + ira;
		
		// replace ppo with po and fpo with po
		d_ppo[globalAddr] = d_po[globalAddr];
		d_po[globalAddr] = d_fpo[globalAddr];

	}
}



__global__ void lint3d_extract_gpu(float *d_dd_pp, 
				   int it, int nr,
				   int nxpad, int nypad, int nzpad,
				   float *d_po, int *d_Rjx, int *d_Rjy, int *d_Rjz,
				   float *d_Rw000, float *d_Rw001, float *d_Rw010, float *d_Rw011, 
				   float *d_Rw100, float *d_Rw101, float *d_Rw110, float *d_Rw111) {

	int rr = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = it * nr;

	if (rr < nr){
		int y_comp = d_Rjy[rr] * nxpad * nzpad;
		int y_comp_1 = (d_Rjy[rr]+1) * nxpad * nzpad;
		int z_comp = d_Rjz[rr] * nxpad;
		int z_comp_1 = (d_Rjz[rr]+1) * nxpad;
		d_dd_pp[offset + rr] = d_po[y_comp   + z_comp   + (d_Rjx[rr])]   * d_Rw000[rr] +
                                       d_po[y_comp   + z_comp_1 + d_Rjx[rr]]     * d_Rw001[rr] +
                                       d_po[y_comp   + z_comp   + (d_Rjx[rr]+1)] * d_Rw010[rr] +
                                       d_po[y_comp   + z_comp_1 + (d_Rjx[rr]+1)] * d_Rw011[rr] +
                                       d_po[y_comp_1 + z_comp   + (d_Rjx[rr])]   * d_Rw100[rr] +
                                       d_po[y_comp_1 + z_comp_1 + d_Rjx[rr]]     * d_Rw101[rr] +
                                       d_po[y_comp_1 + z_comp   + (d_Rjx[rr]+1)] * d_Rw110[rr] +
                                       d_po[y_comp_1 + z_comp_1 + (d_Rjx[rr]+1)] * d_Rw111[rr];

	}

}


__global__ void freeSurf(float *d_po, int nrapad, int nphpad, int nthpad, int nb) {

        int ira = threadIdx.x + blockIdx.x * blockDim.x;
        int iph = threadIdx.y + blockIdx.y * blockDim.y;
        int ith = threadIdx.z + blockIdx.z * blockDim.z;

	if (iph < nphpad && ith < nthpad && ira > nrapad - nb) {
		
		int addr = iph * nthpad * nrapad + ith * nrapad + ira;

		d_po[addr] = 0;

	}
}


__global__ void spongeKernel(float *d_po, int nrapad, int nphpad, int nthpad, int nb){

	int ira = threadIdx.x + blockIdx.x * blockDim.x;
	int iph = threadIdx.y + blockIdx.y * blockDim.y;
	int ith = threadIdx.z + blockIdx.z * blockDim.z;

	float alpha = 0.90;

	// apply sponge
	if (ira < nrapad && iph < nphpad && ith < nthpad) {
        
		int addr = iph * nthpad * nrapad + ith * nrapad + ira;
		int i;

		// apply to low values
		if (ira < nb || iph < nb || ith < nb){

			if      (ira < nb) {i = nb - ira;}
			else if (iph < nb) {i = nb - iph;}
			else if (ith < nb) {i = nb - ith;}
			
			// dampining funct 1
			//double damp = exp(-1.0*fabs(((i-1.0)*log(alpha))/nb)); 
			
			// dampining funct 2
			double damp = exp(-1.0*fabs((pow((i-1.0),2)*log(alpha))/(pow(nb,2))));

			d_po[addr] *= damp;
		
		} 
		// apply to high values
		// do not apply to high values of ra where free surface lies
		// this wouldn't make a difference if it was applied, just
		// makes the code shorter
		if (iph > nphpad - nb || ith > nthpad - nb) {
			
			if      (iph > nphpad - nb) {i = iph - (nphpad - nb);}
                        else                        {i = ith - (nthpad - nb);}

			// dampining funct 1
			//double damp = exp(-1.0*fabs(((i-1.0)*log(alpha))/nb));

                        // dampining funct 2
                        double damp = exp(-1.0*fabs((pow((i-1.0),2)*log(alpha))/(pow(nb,2))));

			d_po[addr] *= damp;

		}
	}
}
