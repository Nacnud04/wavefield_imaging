#include <stdio.h>
#include <math.h>

// macro for 1d index to simulate a 3d matrix
#define INDEX3D(ix, iy, iz, nx, nz) ((iz)+(ix)*(nz)+(iy)*(nz)*(nx))

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

__global__ void lint3d_bell_gpu(float *d_uu, float *d_ww, float *d_Sw000, float *d_Sw001, float *d_Sw010, float *d_Sw011, float *d_Sw100, float *d_Sw101, float *d_Sw110, float *d_Sw111, float *d_bell, int *d_jz, int *d_jy, int *d_jx, int it, int nc, int ns, int c, int nbell, int nxpad, int nzpad) {

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
// finite difference stencil coefficients
__device__ __constant__ float C[4] = {1.2139592f, -0.0911060f, 0.0139508f, -0.0014958f};
//__device__ __constant__ float C[4] = {1.0f, 0.0f, 0.0f, 0.0f};

__global__ void grad3d_cpml(float *d_po,
                            float *d_ux, float *d_uy, float *d_uz,
                            float *d_vel, float *d_ro,
                            float idx, float idy, float idz,
                            float dt, int nxpad, int nypad, int nzpad,
                            int npml
                            ){

	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int iz = threadIdx.z + blockIdx.z * blockDim.z;
	
	float c;
	float ro;

	ro = d_ro[iy * nzpad * nxpad + iz * nxpad + ix];
	c = dt / (ro * (1/idx));

	if (ix < nxpad && iy < nypad && iz < nzpad) {
		
		int globalAddr = iy * nxpad * nzpad + iz * nxpad + ix;
		float pox, poy, poz;
		int i;

		if (ix >= NOP && ix < nxpad-NOP && iy >= NOP && iy < nxpad-NOP && iz >= NOP && iz < nzpad-NOP) {
			
			pox = 0; poy = 0; poz = 0;
			
			// iterate through finite difference coeffs to approximate a derivative
			for (i = 4; i > 0; i--) {
				pox += C[i-1]*(d_po[iy * nxpad * nzpad + iz * nxpad + (ix+i)] - d_po[iy * nxpad * nzpad + iz * nxpad + (ix-i+1)]);
				poy += C[i-1]*(d_po[(iy+i) * nxpad * nzpad + iz * nxpad + ix] - d_po[(iy-i+1)* nxpad * nzpad + iz * nxpad + ix]);
				poz += C[i-1]*(d_po[iy * nxpad * nzpad + (iz+i) * nxpad + ix] - d_po[iy * nxpad * nzpad + (iz-i+1)* nxpad + ix]);
			}
			
			pox *= idx;
			poy *= idy;
			poz *= idz;
			
		} else {
			pox = 0.; poy = 0.; poz = 0.;
		}
		
		d_ux[globalAddr] += c*pox;
		d_uy[globalAddr] += c*poy;
		d_uz[globalAddr] += c*poz;
	

	}

}


__global__ void div3d_cpml(float *d_ux, float *d_uy, float *d_uz,
		           float *d_po, float *d_vel, float *d_ro,
			   float idx, float idy, float idz, float dt,
			   int nxpad, int nypad, int nzpad,
			   int npml
			   ){

	int ix = threadIdx.x + blockIdx.x * blockDim.x;
        int iy = threadIdx.y + blockIdx.y * blockDim.y;
        int iz = threadIdx.z + blockIdx.z * blockDim.z;
        
	if (ix < nxpad && iy < nypad && iz < nzpad){
		
		int globalAddr = iy * nxpad * nzpad + iz * nxpad + ix;
		float kappa = d_ro[globalAddr]*d_vel[globalAddr]*d_vel[globalAddr];
		float uxx, uyy, uzz;
		int i;

		// iterate over are excluding boundaries of edge length = NOP
		if (ix >= NOP && ix < nxpad-NOP && iy >= NOP && iy < nxpad-NOP && iz >= NOP && iz < nxpad - NOP) {
			
			uxx = 0.; uyy = 0.; uzz = 0.;

			// calculate derivative via finite differences
			for (i = 4; i > 0; i--) {
				
				uxx += C[i-1] * (d_ux[iy*nxpad*nzpad + iz*nxpad + (ix+i-1)] - d_ux[iy*nxpad*nzpad + iz*nxpad + (ix-i)]);
				uyy += C[i-1] * (d_uy[(iy+i-1)*nxpad*nzpad + iz*nxpad + ix] - d_uy[(iy-i)*nxpad*nzpad + iz*nxpad + ix]);
				uzz += C[i-1] * (d_uz[iy*nxpad*nzpad + (iz+i-1)*nxpad + ix] - d_uz[iy*nxpad*nzpad + (iz-i)*nxpad + ix]);

			}

			//uxx *= idx; uyy *= idy; uzz *= idz;
				
		}

		// do someting different if we are computing for near the boundaries
		else {

			uxx = 0.; uyy = 0.; uzz = 0.;

		}

		d_po[globalAddr] += dt * kappa * (uxx + uyy + uzz);
		
	}
}


__global__ void lint3d_extract_gpu(float *d_dd_pp, float *d_dd_ux, float *d_dd_uy, float *d_dd_uz, 
				   int it, int nr,
				   int nxpad, int nypad, int nzpad,
				   float *d_uox, float *d_uoy, float *d_uoz,
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

                

                /* uz */
                d_dd_uz[offset + rr] = d_uoz[y_comp   + z_comp   + (d_Rjx[rr])]   * d_Rw000[rr] +
                                       d_uoz[y_comp   + z_comp_1 + d_Rjx[rr]]     * d_Rw001[rr] +
                                       d_uoz[y_comp   + z_comp   + (d_Rjx[rr]+1)] * d_Rw010[rr] +
                                       d_uoz[y_comp   + z_comp_1 + (d_Rjx[rr]+1)] * d_Rw011[rr] +
                                       d_uoz[y_comp_1 + z_comp   + (d_Rjx[rr])]   * d_Rw100[rr] +
                                       d_uoz[y_comp_1 + z_comp_1 + d_Rjx[rr]]     * d_Rw101[rr] +
                                       d_uoz[y_comp_1 + z_comp   + (d_Rjx[rr]+1)] * d_Rw110[rr] +
                                       d_uoz[y_comp_1 + z_comp_1 + (d_Rjx[rr]+1)] * d_Rw111[rr];

                d_dd_ux[offset + rr] = d_uox[y_comp   + z_comp   + (d_Rjx[rr])]   * d_Rw000[rr] +
                                       d_uox[y_comp   + z_comp_1 + d_Rjx[rr]]     * d_Rw001[rr] +
                                       d_uox[y_comp   + z_comp   + (d_Rjx[rr]+1)] * d_Rw010[rr] +
                                       d_uox[y_comp   + z_comp_1 + (d_Rjx[rr]+1)] * d_Rw011[rr] +
                                       d_uox[y_comp_1 + z_comp   + (d_Rjx[rr])]   * d_Rw100[rr] +
                                       d_uox[y_comp_1 + z_comp_1 + d_Rjx[rr]]     * d_Rw101[rr] +
                                       d_uox[y_comp_1 + z_comp   + (d_Rjx[rr]+1)] * d_Rw110[rr] +
                                       d_uox[y_comp_1 + z_comp_1 + (d_Rjx[rr]+1)] * d_Rw111[rr];

		d_dd_uy[offset + rr] = d_uoy[y_comp   + z_comp   + (d_Rjx[rr])]   * d_Rw000[rr] +
                                       d_uoy[y_comp   + z_comp_1 + d_Rjx[rr]]     * d_Rw001[rr] +
                                       d_uoy[y_comp   + z_comp   + (d_Rjx[rr]+1)] * d_Rw010[rr] +
                                       d_uoy[y_comp   + z_comp_1 + (d_Rjx[rr]+1)] * d_Rw011[rr] +
                                       d_uoy[y_comp_1 + z_comp   + (d_Rjx[rr])]   * d_Rw100[rr] +
                                       d_uoy[y_comp_1 + z_comp_1 + d_Rjx[rr]]     * d_Rw101[rr] +
                                       d_uoy[y_comp_1 + z_comp   + (d_Rjx[rr]+1)] * d_Rw110[rr] +
                                       d_uoy[y_comp_1 + z_comp_1 + (d_Rjx[rr]+1)] * d_Rw111[rr];

	}

}


__global__ void freeSurf(float *d_po, int nxpad, int nypad, int nzpad, int nb) {

        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x < nxpad && y < nypad && z < nb) {
		
		int addr = y * nxpad * nzpad + z * nxpad + x;

		d_po[addr] = 0;

	}
}


__global__ void spongeKernel(float *d_po, int nxpad, int nypad, int nzpad, int nb){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	float alpha = 0.90;

	// apply sponge
	if (x < nxpad && y < nypad && z < nzpad) {
        	
		// apply to low values
		if (x < nb || y < nb){

			int addr = y * nxpad * nzpad + z * nxpad + x;

			int i = nb - x;
			// dampining funct 1
			//double damp = exp(-1.0*fabs(((i-1.0)*log(alpha))/nb)); 
			
			// dampining funct 2
			double damp = exp(-1.0*fabs((pow((i-1.0),2)*log(alpha))/(pow(nb,2))));

			d_po[addr] *= damp;
		
		}
		// apply to high values
		if (x > nxpad - nb || y > nypad - nb || z > nzpad - nb) {
			
			int addr = y * nxpad * nzpad + z * nxpad + x;
			
			int i = x - (nxpad - nb);
			// dampining funct 1
			//double damp = exp(-1.0*fabs(((i-1.0)*log(alpha))/nb));

                        // dampining funct 2
                        double damp = exp(-1.0*fabs((pow((i-1.0),2)*log(alpha))/(pow(nb,2))));

			d_po[addr] *= damp;

		}

	}

}
