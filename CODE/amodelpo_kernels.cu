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
                        b[INDEX2D(ix+nb,iz+nb,x_b)] = a[INDEX2D(ix,iz,x_a)];
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

__global__ void lint2d_bell_gpu(float *d_uu, float *d_ww, float *d_Sw00, float *d_Sw01, float *d_Sw10, float *d_Sw11, float *d_bell, int *d_jx, int *d_jz, int it, int nc, int ns, int c, int nbell, int nxpad) {

        int ix = threadIdx.x;
        int iz = threadIdx.y;
        int ia = blockIdx.x;

        float wa = d_ww[it * nc * ns + c * ns + ia] * d_bell[(iz * (2*nbell+1)) + ix];

	int z_comp = (d_jz[ia] - nbell) + iz;
	int x_comp = (d_jx[ia] - nbell) + ix;

        atomicAdd(&d_uu[(z_comp)     * nxpad + (x_comp    )], ((wa * d_Sw00[ia])));
        atomicAdd(&d_uu[(z_comp + 1) * nxpad + (x_comp    )], ((wa * d_Sw01[ia])));
        atomicAdd(&d_uu[(z_comp)     * nxpad + (x_comp + 1)], ((wa * d_Sw10[ia])));
        atomicAdd(&d_uu[(z_comp + 1) * nxpad + (x_comp + 1)], ((wa * d_Sw11[ia])));

}


// divergence 3d for cpml
#define NOP 4 // half of the order in space

__global__ void solve(float *d_fpo, float *d_po, float *d_ppo, float *d_vel,
		      float dra, float dth, float ora, float oth, float dt,
		      int nrapad, int nthpad) {

	int ira = threadIdx.x + blockIdx.x * blockDim.x;
	int ith = threadIdx.y + blockIdx.y * blockDim.y;

	if (ira < nrapad && ith < nthpad){
		
		int globalAddr = ith * nrapad + ira;			  
		float laplace;
		float compra, compth;

		// extract true location from deltas and indicies
		float ra; float th;
		ra = dra * ira + ora;
		th = dth * ith + oth;
		
		// extract true velocity
		float v;
		v  = d_vel[globalAddr];

		// perform only in boundaries:
		if (ira >= NOP && ira < nrapad-NOP && ith >= NOP && ith < nthpad - NOP) {

			// CALC LAPLACE VIA STENCIL
			
			// START BY LOOKING AT VALS ALONG -R then R then +R
			compra = ((1/(dra*dra))+(1/(2*ra*dra))) * d_po[INDEX2D(ira-1,ith,nrapad)] + 
				 (-2/(dra*dra))                 * d_po[globalAddr] +
				 ((1/(dra*dra))-(1/(2*ra*dra))) * d_po[INDEX2D(ira+1,ith,nrapad)];
			
			// NOW COMPUTE COMPONENTS DEPENDENT ON THETA
			compth = ((1/(ra*ra*dth*dth))) * d_po[INDEX2D(ira,ith-1,nrapad)] + 
				 (-2/(ra*ra*dth*dth))  * d_po[globalAddr] + 
				 ((1/(ra*ra*dth*dth))) * d_po[INDEX2D(ira,ith+1,nrapad)];

			// SUM TO GET LAPLACIAN
			laplace = compra + compth;

		} else {
			laplace = 0.;
		}

		// compute pressure at next time step
		d_fpo[globalAddr] = (dt*dt) * (v*v) * laplace + 2*d_po[globalAddr] - d_ppo[globalAddr];


	}

}


__global__ void shift(float *d_fpo, float *d_po, float *d_ppo,
		      int nrapad, int nthpad) {
	
	int ira = threadIdx.x + blockIdx.x * blockDim.x;
        int ith = threadIdx.y + blockIdx.y * blockDim.y;

	if (ira < nrapad && ith < nthpad){

		int globalAddr = ith * nrapad + ira;
		
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


__global__ void freeSurf(float *d_po, int nrapad, int nthpad, int nb) {

        int ira = threadIdx.x + blockIdx.x * blockDim.x;
        int ith = threadIdx.y + blockIdx.y * blockDim.y;

	// apply freesurface on the extent of the planet
	// AKA where radius is greatest
	if (ith < nthpad && ira > nrapad - nb) {
		int addr = ith * nrapad + ira;
		d_po[addr] = 0;
	}
}


__global__ void spongeKernel(float *d_po, int nxpad, int nzpad, int nb){

        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int z = threadIdx.y + blockIdx.y * blockDim.y;

        float alpha = 0.90;
        double damp;
        int i = 1;

        // apply sponge
        if (x < nxpad && z < nzpad) {

                int addr = z * nxpad + x;

                // apply to low values
                if (x < nb || z < nb){

                        if (x < nb) { i = nb - x; }
                        else { i = nb - z; }

                        float fb = i / (sqrt(2.0)*(4.0*nb));
                        damp = exp(-fb * fb);
                        damp = exp(-1.0*fabs((pow((i-1.0),2)*log(alpha))/(pow(nb,2))));
                        d_po[addr] *= damp;

                }
                // apply to high values
                // NOTE: even though this is applied to all surfaces it only influences
                //       high th due to high ra being a free surface
                else if (x > nxpad - nb || z > nzpad - nb) {

                        if (x > nxpad - nb) { i = x - (nxpad - nb); }
                        else { i = z - (nzpad - nb); }

                        float fb = i / (sqrt(2.0)*(4.0*nb));
                        damp = exp(-fb * fb);
                        damp = exp(-1.0*fabs((pow((i-1.0),2)*log(alpha))/(pow(nb,2))));
                        d_po[addr] *= damp;

                }

        }

}


__global__ void spongeKernelOLD(float *d_po, int nrapad, int nthpad, int nb){

	int ra = threadIdx.x + blockIdx.x * blockDim.x;
	int th = threadIdx.y + blockIdx.y * blockDim.y;

	float alpha = 0.95;
	double damp;
	int i = 1;

	// apply sponge
	if (ra < nrapad && th < nthpad) {
        
		int addr = th * nrapad + ra;

		// apply to low values
		if (ra < nb || th < nb){
			
			if (ra < nb) { i = nb - ra; }
			else { i = nb - th; }

			// dampining funct 1
			//double damp = exp(-1.0*fabs(((i-1.0)*log(alpha))/nb)); 
			
			// dampining funct 2
			//double damp = exp(-1.0*fabs((pow((i-1.0),2)*log(alpha))/(pow(nb,2))));

			//double damp = 0.5 + 0.5*cos(((1.0*i)/nb)*(3.14159));

			float fb = i / (sqrt(2.0)*(4.0*nb));
			damp = exp(-fb * fb);	
			//d_po[addr] = damp;
			d_po[addr] *= damp;
		
		}
		// apply to high values
		// NOTE: even though this is applied to all surfaces it only influences
		//       high th due to high ra being a free surface
		else if (ra > nrapad - nb || th > nthpad - nb) {
				
			if (ra > nrapad - nb) { i = ra - (nrapad - nb); }
			else { i = th - (nthpad - nb); }
			
			// dampining funct 1
			//double damp = exp(-1.0*fabs(((i-1.0)*log(alpha))/nb));

                        // dampining funct 2
                        double damp = exp(-1.0*fabs((pow((i-1.0),2)*log(alpha))/(pow(nb,2))));
			
			//double damp = 0.5 + 0.5*cos(((1.0*i)/nb)*(3.14159));
			
			float fb = i / (sqrt(2.0)*(4.0*nb));
			damp = exp(-fb * fb);
			//d_po[addr] = damp;
			d_po[addr] *= damp;

		}

		//else { d_po[addr] = 1; }

	}

}


__global__ void onewayBC(float *uo, float *um,
		         float *d_bthl, float *d_bthh, float *d_bral, float *d_brah,
			 int nrapad, int nthpad) {

	int ira = threadIdx.x + blockIdx.x * blockDim.x;
	int ith = threadIdx.y + blockIdx.y * blockDim.y;
	int iop;

	int addr = ith * nrapad + ira;

	if (ira < nrapad && ith < nthpad) {

		for (ira=0; ira<nrapad; ira++) {
			for (iop=0; iop<NOP; iop++) {
				
				// top bc
				if (ith == NOP-iop) {
					uo[addr] =  um[(ith+1)*nrapad+ira] +
						   (um[addr] - uo[(ith+1)*nrapad+ira]) * d_bthl[ira];
				}
				// bottom bc
				if (ith == nthpad-NOP+iop-1) {
                                        uo[addr] =  um[(ith-1)*nrapad+ira] +
                                                   (um[addr] - uo[(ith-1)*nrapad+ira]) * d_bthh[ira];
                                }

			}
		}

		for (ith=0; ith<nthpad; ith++) {
			for (iop=0; iop<NOP; iop++) {
				
				// left bc
				if (ira == NOP-iop) {
					uo[addr] =  um[ith*nrapad+ira+1] +
                                                   (um[addr] - uo[ith*nrapad+ira+1]) * d_bral[ith];
				}
				// bottom bc
				if (ira == nrapad-NOP+iop-1) {
					uo[addr] =  um[ith*nrapad+ira-1] +
                                                   (um[addr] - uo[ith*nrapad+ira-1]) * d_brah[ith];
				}

			}
		}		

	}

}


__global__ void extract(float *d_dd_pp, 
			int it, int nr,
			int nrapad, int nthpad,
			float *d_po, int *d_Rjra, int *d_Rjth,
			float *d_Rw00, float *d_Rw01, float *d_Rw10, float *d_Rw11) {

	// receiver number
	int rr = threadIdx.x + blockIdx.x * blockDim.x;
	// time offset
	// avoids rewriting over previously received data
	int offset = it * nr;

	// only perform if the receiver number represents an actual existing receiver
	if (rr < nr){

		int th_comp   = (d_Rjth[rr]) * nrapad;
		int th_comp_1 = (d_Rjth[rr]+1) * nrapad;

		// set recived pressure vals
		
		d_dd_pp[offset + rr] = d_po[th_comp   + (d_Rjra[rr])]   * d_Rw00[rr] +
                                       d_po[th_comp_1 + (d_Rjra[rr])]   * d_Rw01[rr] +
                                       d_po[th_comp   + (d_Rjra[rr]+1)] * d_Rw10[rr] +
                                       d_po[th_comp_1 + (d_Rjra[rr]+1)] * d_Rw11[rr];


	}

}
