#include <stdio.h>
#include <math.h>

// macro for 1d index to simulate a 3d matrix
#define INDEX2D(ix, iz, nx) ((ix)+(iz)*(nx))
// PLACEHOLDER
#define INDEX3D(ix, iy, iz, nx, nz) ((ix)+(iz)*(nx)+(iy)*(nx)*(nz))

void expand_cpu_2d(float *a, float *b, int nb, int x_a, int x_b, int z_a, int z_b){

        // copy into other array
	for (int ix = 0; ix < x_a; ix++) {
                for (int iz = 0; iz < z_a; iz++) {
                        b[INDEX2D(ix+nb,iz+nb,z_b)] = a[INDEX2D(ix,iz,x_a)];
		}
        }
	
        // expand z direction
	for (int ix = 0; ix < x_b; ix++) {
                for (int iz = 0; iz < nb; iz++) {
                        b[INDEX2D(ix,iz,x_b)] = b[INDEX2D(ix,nb,x_b)];
                        b[INDEX2D(ix,z_b-iz-1,x_b)] = b[INDEX2D(ix,z_b-nb-1,x_b)];
                }
        }
	
        //expand x direction 
	for (int ix = 0; ix < nb; ix++) {
                for (int iz = 0; iz < z_b; iz++) {
                        b[INDEX2D(ix,iz,x_b)] = b[INDEX2D(nb,iz,x_b)];
                        b[INDEX2D(x_b-ix-1,iz,x_b)] = b[INDEX2D(x_b-nb-1,iz,x_b)];
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

        atomicAdd(&d_uu[(y_comp)     * xz + (z_comp)     * nxpad + (x_comp)], ((wa * d_Sw000[ia])));
        atomicAdd(&d_uu[(y_comp)     * xz + (z_comp + 1) * nxpad + (x_comp)], ((wa * d_Sw001[ia])));
        atomicAdd(&d_uu[(y_comp)     * xz + (z_comp)     * nxpad + (x_comp)], ((wa * d_Sw010[ia])));
        atomicAdd(&d_uu[(y_comp)     * xz + (z_comp + 1) * nxpad + (x_comp)], ((wa * d_Sw011[ia])));
        atomicAdd(&d_uu[(y_comp + 1) * xz + (z_comp)     * nxpad + (x_comp)], ((wa * d_Sw100[ia])));
        atomicAdd(&d_uu[(y_comp + 1) * xz + (z_comp + 1) * nxpad + (x_comp)], ((wa * d_Sw101[ia])));
        atomicAdd(&d_uu[(y_comp + 1) * xz + (z_comp)     * nxpad + (x_comp)], ((wa * d_Sw110[ia])));
        atomicAdd(&d_uu[(y_comp + 1) * xz + (z_comp + 1) * nxpad + (x_comp)], ((wa * d_Sw111[ia])));

}


// divergence 3d for cpml
#define NOP 4 // half of the order in space

__global__ void solve(float *d_fpo, float *d_po, float *d_ppo, float *d_vel,
		      float dra, float dph, float dth, float dt,
		      int nrapad, int nthpad, int nphpad) {

	int ira = threadIdx.x + blockIdx.x * blockDim.x;
	int iph = threadIdx.y + blockIdx.y * blockDim.y;
	int ith = threadIdx.z + blockIdx.z * blockDim.z;

	if (ira < nrapad && ith < nthpad && iph < nphpad){
		
		int globalAddr = iph * nthpad * nrapad + ith * nrapad + ira;			  float laplace;

		// extract true location from deltas and indicies
		float ra; float ph; float th;
		ra = dra * ira;
		ph = dph * iph;
		th = dth * ith;
		
		// extract true velocity
		float v;
		v  = d_vel[globalAddr];

		// perform only in boundaries:
		if (ira >= NOP && ira < nrapad-NOP && iph >= NOP && iph < nphpad-NOP && ith >= NOP && ith < nthpad - NOP) {

			
			// compute d/dra*(ra^2*dp/dra)
			// to do this we need to compute dp/dra one step forward
			// and one back to produce d/dra
			
			float pra_p; float pra_n; // + and - derivative along ra
			pra_p =  d_po[INDEX3D(ira+1+1, iph, ith, nrapad, nthpad)]\
			        -d_po[INDEX3D(ira-1+1, iph, ith, nrapad, nthpad)];
			pra_p = pra_p / (2 * dra);
			pra_n =  d_po[INDEX3D(ira+1-1, iph, ith, nrapad, nthpad)]\
				-d_po[INDEX3D(ira-1-1, iph, ith, nrapad, nthpad)];
			pra_n = pra_n / (2 * dra);
			
			// multiply by r^2
			pra_p = pra_p * ra * ra;
			pra_n = pra_n * ra * ra;

			// compute FD using pra_p and pra_n
			float ppra;
			ppra = pra_p - pra_n;
			ppra = ppra / (2 * dra);

			
			// compute d/dth*(Sin(th)*dp/dth)
			// to do this we need to compute dp/dth one step forward
			// and one back to prudce d/dth
			
			float pth_p; float pth_n; // + and - derivative along th
			pth_p =  d_po[INDEX3D(ira, iph, ith+1+1, nrapad, nthpad)]\
				-d_po[INDEX3D(ira, iph, ith-1+1, nrapad, nthpad)];
			pth_p = pth_p / (2 * dth);
			pth_n =  d_po[INDEX3D(ira, iph, ith+1-1, nrapad, nthpad)]\
				-d_po[INDEX3D(ira, iph, ith-1-1, nrapad, nthpad)];
			pth_n = pth_n / (2 * dth);

			// multiply by sin(theta)
			pth_p = sin(th) * pth_p;
			pth_n = sin(th) * pth_n;

			// compute FD using pth_p and pth_n
			float ppth;
			ppth = pth_p - pth_n;
			ppth = ppth / (2 * dth);


			// compute pphph (d^2p/dph^2)
			float pphph;
			pphph =    d_po[INDEX3D(ira, iph, ith+1,nrapad,nthpad)] \
                                -2*d_po[INDEX3D(ira, iph, ith  ,nrapad,nthpad)] \
                                  +d_po[INDEX3D(ira, iph, ith-1,nrapad,nthpad)];
                        pphph = pphph / (dph * dph);


			// multiply r^-2 with ppra
			ppra = ppra / (ra * ra);
			// multiply 1/(r^2*Sin(th))
			ppth = ppth / (ra * ra * sin(th));
			// multiply 1/(r^2*Sin(th))
			pphph = pphph / (ra * ra * sin(th) * sin(th));

			// combine into laplacian
			laplace = ppra + ppth + pphph;

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
			double damp = exp(-1.0*fabs(((i-1.0)*log(alpha))/nb)); 
			
			// dampining funct 2
			//double damp = exp(-1.0*fabs((pow((i-1.0),2)*log(alpha))/(pow(nb,2))));

			d_po[addr] *= damp;
		
		}
		// apply to high values
		if (x > nxpad - nb || y > nypad - nb || z > nzpad - nb) {
			
			int addr = y * nxpad * nzpad + z * nxpad + x;
			
			int i = x - (nxpad - nb);
			// dampining funct 1
			double damp = exp(-1.0*fabs(((i-1.0)*log(alpha))/nb));

                        // dampining funct 2
                        //double damp = exp(-1.0*fabs((pow((i-1.0),2)*log(alpha))/(pow(nb,2))));

			d_po[addr] *= damp;

		}

	}

}
