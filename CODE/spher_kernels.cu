#include <stdio.h>
#include <math.h>
#define PI 3.141592654

#define INDEX2D(ix, iz, nx) ((ix)+(iz)*(nx))
// macro for 1d index to simulate a 3d matrix
#define INDEX3D(ix, iy, iz, nx, nz) ((size_t)(ix)+(size_t)(iz)*(nx)+(size_t)(iy)*(nz)*(nx))

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

void expand_cpu_3D(float *a, float *b, int nb, int x_a, int x_b, int y_a, int y_b, int z_a, int z_b){

        // copy into other array
	for (int ix = 0; ix < x_a; ix++) {
                for (int iy = 0; iy < y_a; iy++) {
                        for (int iz = 0; iz < z_a; iz++) {
                                b[INDEX3D(ix+nb,iy+nb,iz+nb,x_b,z_b)] = a[INDEX3D(ix,iy,iz,x_a,z_a)];
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

__global__ void lint3D_bell_gpu(float *d_uu, float *d_ww, float *d_Sw000, float *d_Sw001, float *d_Sw010, float *d_Sw011, float *d_Sw100, float *d_Sw101, float *d_Sw110, float *d_Sw111, float *d_bell, int *d_jz, int *d_jy, int *d_jx, int it, int nc, int ns, int c, int nbell, int nxpad, int nzpad) {

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

__global__ void inject_single_source_3D(float *d_uu, float *d_ww, 
		float *d_Sw000, float *d_Sw001, float *d_Sw010, float *d_Sw011, 
		float *d_Sw100, float *d_Sw101, float *d_Sw110, float *d_Sw111, 
		int *d_jx, int *d_jy, int *d_jz, 
		int it, 
		int nxpad, int nypad, int nzpad) {

        int ix = threadIdx.x + blockIdx.x * blockDim.x;
        int iy = threadIdx.y + blockIdx.y * blockDim.y;
        int iz = threadIdx.z + blockIdx.z * blockDim.z;

        int s_x = d_jx[0];
        int s_y = d_jy[0];
        int s_z = d_jz[0];

        float wa = d_ww[it];

        int addr = iy * nxpad * nzpad + iz * nxpad + ix;

        if (ix == s_x && iy == s_y && iz == s_z){
                d_uu[addr] = wa * d_Sw000[0];
        }
        if (ix == s_x && iy == s_y && iz == s_z + 1){
                d_uu[addr] = wa * d_Sw001[0];
        }
        if (ix == s_x + 1 && iy == s_y && iz == s_z){
                d_uu[addr] = wa * d_Sw010[0];
        }
        if (ix == s_x + 1 && iy == s_y && iz == s_z + 1){
                d_uu[addr] = wa * d_Sw011[0];
        }
        if (ix == s_x && iy == s_y + 1 && iz == s_z){
                d_uu[addr] = wa * d_Sw100[0];
        }
        if (ix == s_x && iy == s_y + 1 && iz == s_z + 1){
                d_uu[addr] = wa * d_Sw101[0];
        }
        if (ix == s_x + 1 && iy == s_y + 1 && iz == s_z){
                d_uu[addr] = wa * d_Sw110[0];
        }
        if (ix == s_x + 1 && iy == s_y + 1 && iz == s_z + 1){
                d_uu[addr] = wa * d_Sw111[0];
        }

}

__global__ void inject_sources_2D(float *d_po, float *d_ww,
							   float *d_Sw00, float *d_Sw01, float *d_Sw10, float *d_Sw11,
							   int *d_Sjx, int *d_Sjz,
							   int it, int ns,
							   int nxpad, int nzpad) {

	int ss = threadIdx.x + blockIdx.x * blockDim.x;

	float wa = d_ww[it];

	if (ss < ns) {

		int s_x = d_Sjx[ss];
		int s_z = d_Sjz[ss];

		d_po[ s_z      * nxpad + s_x    ] += wa * d_Sw00[ss];
		d_po[(s_z + 1) * nxpad + s_x    ] += wa * d_Sw01[ss];
		d_po[ s_z      * nxpad + s_x + 1] += wa * d_Sw10[ss];
		d_po[(s_z + 1) * nxpad + s_x + 1] += wa * d_Sw11[ss];

	}

}

__global__ void inject_sources_3D(float *d_po, float *d_ww, 
		float *d_Sw000, float *d_Sw001, float *d_Sw010, float *d_Sw011, 
		float *d_Sw100, float *d_Sw101, float *d_Sw110, float *d_Sw111, 
		int *d_Sjx, int *d_Sjy, int *d_Sjz, 
		int it, int ns,
		int nxpad, int nypad, int nzpad) {

        int ss = threadIdx.x + blockIdx.x * blockDim.x;

        float wa = d_ww[it];

        if (ss < ns) {

                int s_x = d_Sjx[ss];
                int s_y = d_Sjy[ss];
                int s_z = d_Sjz[ss];

                size_t xz = nxpad * nzpad;

                d_po[xz*s_y     + s_z*nxpad     + s_x  ] += wa * d_Sw000[ss];
                d_po[xz*s_y     + (s_z+1)*nxpad + s_x  ] += wa * d_Sw001[ss];
                d_po[xz*s_y     + s_z*nxpad     + s_x+1] += wa * d_Sw010[ss];
                d_po[xz*s_y     + (s_z+1)*nxpad + s_x+1] += wa * d_Sw011[ss];
                d_po[xz*(s_y+1) + s_z*nxpad     + s_x  ] += wa * d_Sw100[ss];
                d_po[xz*(s_y+1) + (s_z+1)*nxpad + s_x  ] += wa * d_Sw101[ss];
                d_po[xz*(s_y+1) + s_z*nxpad     + s_x+1] += wa * d_Sw110[ss];
                d_po[xz*(s_y+1) + (s_z+1)*nxpad + s_x+1] += wa * d_Sw111[ss];

        }
}

__global__ void inject_sources_3D_adj(float *d_po, float *d_ww, 
        float *d_Sw000, float *d_Sw001, float *d_Sw010, float *d_Sw011, 
        float *d_Sw100, float *d_Sw101, float *d_Sw110, float *d_Sw111, 
        int *d_Sjx, int *d_Sjy, int *d_Sjz, 
        int it, int ns,
        int nxpad, int nypad, int nzpad) {

        int ss = threadIdx.x + blockIdx.x * blockDim.x;

        size_t index = (size_t)it * (size_t)ns + ss;

        float wa = d_ww[index];

        if (ss < ns) {

                int s_x = d_Sjx[ss];
                int s_y = d_Sjy[ss];
                int s_z = d_Sjz[ss];

                size_t xz = nxpad * nzpad;

                d_po[xz*s_y     + s_z*nxpad     + s_x  ] += wa * d_Sw000[ss];
                d_po[xz*s_y     + (s_z+1)*nxpad + s_x  ] += wa * d_Sw001[ss];
                d_po[xz*s_y     + s_z*nxpad     + s_x+1] += wa * d_Sw010[ss];
                d_po[xz*s_y     + (s_z+1)*nxpad + s_x+1] += wa * d_Sw011[ss];
                d_po[xz*(s_y+1) + s_z*nxpad     + s_x  ] += wa * d_Sw100[ss];
                d_po[xz*(s_y+1) + (s_z+1)*nxpad + s_x  ] += wa * d_Sw101[ss];
                d_po[xz*(s_y+1) + s_z*nxpad     + s_x+1] += wa * d_Sw110[ss];
                d_po[xz*(s_y+1) + (s_z+1)*nxpad + s_x+1] += wa * d_Sw111[ss];

}
}


// divergence 3d for cpml
#define NOP 4 // half of the order in space

__global__ void solve_2D(float *d_fpo, float *d_po, float *d_ppo, float *d_vel,
		      float dra, float dth, float ora, float oth, float dt,
		      int nrapad, int nthpad) {

	int ira = threadIdx.x + blockIdx.x * blockDim.x;
	int ith = threadIdx.y + blockIdx.y * blockDim.y;

	if (ira < nrapad && ith < nthpad){
		
		int globalAddr = ith * nrapad + ira;			  
		float laplace;
		float compra, compth;

		// extract true location from deltas and indicies
		float ra;
		ra = dra * ira + ora;
		
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

__global__ void solve_3D(float *d_fpo, float *d_po, float *d_ppo, float *d_vel,
		      float dra, float dph, float dth, float ora, float oph, float oth, 
		      float dt,
		      int nrapad, int nphpad, int nthpad) {

	int ira = threadIdx.x + blockIdx.x * blockDim.x;
	int iph = threadIdx.y + blockIdx.y * blockDim.y;
	int ith = threadIdx.z + blockIdx.z * blockDim.z;

	if (ira < nrapad && ith < nthpad && iph < nphpad){
		
		size_t globalAddr = (size_t)iph * nthpad * nrapad + (size_t)ith * nrapad + (size_t)ira;			  
                float laplace;

		// extract true location from deltas and indicies
		float ra; float th;
		ra = dra * ira + ora;
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

__global__ void shift_2D(float *d_fpo, float *d_po, float *d_ppo,
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

__global__ void shift_3D(float *d_fpo, float *d_po, float *d_ppo,
		      int nrapad, int nphpad, int nthpad) {
	
	int ira = threadIdx.x + blockIdx.x * blockDim.x;
        int iph = threadIdx.y + blockIdx.y * blockDim.y;
        int ith = threadIdx.z + blockIdx.z * blockDim.z;

	if (ira < nrapad && iph < nphpad && ith < nthpad){

		size_t globalAddr = (size_t)iph * nthpad * nrapad + (size_t)ith * nrapad + (size_t)ira;
		
		// replace ppo with po and fpo with po
		d_ppo[globalAddr] = d_po[globalAddr];
		d_po[globalAddr] = d_fpo[globalAddr];

	}
}


__global__ void wavefield_extract_3D(float ***d_po3d, float *d_po, 
				  int nrapad, int nphpad, int nthpad) {

	int ira = threadIdx.x + blockIdx.x * blockDim.x;
        int iph = threadIdx.y + blockIdx.y * blockDim.y;
        int ith = threadIdx.z + blockIdx.z * blockDim.z;

        if (ira < nrapad && iph < nphpad && ith < nthpad){

		int globalAddr = iph * nthpad * nrapad + ith * nrapad + ira;
		d_po3d[iph][ira][ith] = d_po[globalAddr];

	}

}

__global__ void extract_2D(float *d_dd_pp, 
			int it, int nr,
			int nrapad, int nthpad,
			float *d_po, int *d_Rjra, int *d_Rjth,
			float *d_Rw00, float *d_Rw01, float *d_Rw10, float *d_Rw11) {

	// receiver number
	int rr = threadIdx.x + blockIdx.x * blockDim.x;
	// time offset
	// avoids rewriting over previously received data
	size_t offset = (size_t)it * (size_t)nr;

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

__global__ void extract_3D(float *d_dd_pp, 
				   int it, int nr,
				   int nxpad, int nypad, int nzpad,
				   float *d_po, int *d_Rjx, int *d_Rjy, int *d_Rjz,
				   float *d_Rw000, float *d_Rw001, float *d_Rw010, float *d_Rw011, 
				   float *d_Rw100, float *d_Rw101, float *d_Rw110, float *d_Rw111) {

	int rr = threadIdx.x + blockIdx.x * blockDim.x;

	if (rr < nr){

		int y_comp = d_Rjy[rr] * nxpad * nzpad;
		int y_comp_1 = (d_Rjy[rr]+1) * nxpad * nzpad;
		int z_comp = d_Rjz[rr] * nxpad;
		int z_comp_1 = (d_Rjz[rr]+1) * nxpad;
		d_dd_pp[rr] = d_po[y_comp   + z_comp   + (d_Rjx[rr])]   * d_Rw000[rr] +
                                       d_po[y_comp   + z_comp_1 + d_Rjx[rr]]     * d_Rw001[rr] +
                                       d_po[y_comp   + z_comp   + (d_Rjx[rr]+1)] * d_Rw010[rr] +
                                       d_po[y_comp   + z_comp_1 + (d_Rjx[rr]+1)] * d_Rw011[rr] +
                                       d_po[y_comp_1 + z_comp   + (d_Rjx[rr])]   * d_Rw100[rr] +
                                       d_po[y_comp_1 + z_comp_1 + d_Rjx[rr]]     * d_Rw101[rr] +
                                       d_po[y_comp_1 + z_comp   + (d_Rjx[rr]+1)] * d_Rw110[rr] +
                                       d_po[y_comp_1 + z_comp_1 + (d_Rjx[rr]+1)] * d_Rw111[rr];

	}

}


__global__ void freeSurf_2D(float *d_po, int nrapad, int nthpad, int nb) {

        int ira = threadIdx.x + blockIdx.x * blockDim.x;
        int ith = threadIdx.y + blockIdx.y * blockDim.y;

	// apply freesurface on the extent of the planet
	// AKA where radius is greatest
	if (ith < nthpad && ira > nrapad - nb) {
		int addr = ith * nrapad + ira;
		d_po[addr] = 0;
	}
}

__global__ void freeSurf_3D(float *d_po, int nrapad, int nphpad, int nthpad, int nb) {

        int ira = threadIdx.x + blockIdx.x * blockDim.x;
        int iph = threadIdx.y + blockIdx.y * blockDim.y;
        int ith = threadIdx.z + blockIdx.z * blockDim.z;

	if (iph < nphpad && ith < nthpad && ira > nrapad - nb) {
		
		int addr = iph * nthpad * nrapad + ith * nrapad + ira;

		d_po[addr] = 0;

	}
}

__global__ void onewayBC_2D(float *uo, float *um,
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

__global__ void onewayBC_3D(float *uo, float *um,
                         float *d_bzl, float *d_bzh,
                         float *d_bxl, float *d_bxh,
                         float *d_byl, float *d_byh,
                         int nxpad, int nypad, int nzpad) {

        int ix = threadIdx.x + blockIdx.x * blockDim.x;
        int iy = threadIdx.y + blockIdx.y * blockDim.y;
        int iz = threadIdx.z + blockIdx.z * blockDim.z;

        size_t addr = (size_t)iy * nxpad * nzpad + (size_t)iz * nxpad + (size_t)ix;

        if (ix < nxpad && iy < nypad && iz < nzpad) {

                int iop;

                if (ix < NOP) {iop = ix;}
                else if (ix > nxpad - NOP) {iop = nxpad - ix;}
                else if (iz < NOP) {iop = iz;}
                else if (iz > nzpad - NOP) {iop = nzpad - iz;}
                else if (iy < NOP) {iop = iy;}
                else if (iy > nypad - NOP) {iop = nypad - iy;}
                else {iop = 0;}

                // top bc
                if (iz <= NOP - iop) {
                        size_t taddr = (size_t)iy*nxpad*nzpad + (iz+1)*nxpad + ix;
                        uo[addr] = um[taddr] + (um[addr] - uo[taddr]) *
                                   d_bzl[iy*nxpad + ix];
                }
                // bottom bc
                if (iz >= nzpad-NOP-1+iop) {
                        size_t taddr = (size_t)iy*nxpad*nzpad + (iz-1)*nxpad + ix;
                        uo[addr] = um[taddr] + (um[addr] - uo[taddr]) *
                                   d_bzh[iy*nxpad + ix];
                }
                
                if (ix <= NOP-iop) {
                        size_t taddr = (size_t)iy*nxpad*nzpad + iz*nxpad + ix + 1;
                        uo[addr] = um[taddr] + (um[addr] - uo[taddr]) *
                                   d_bxl[iy*nzpad + iz];
                }
                if (ix >= nxpad-NOP-1+iop) {
                        size_t taddr = (size_t)iy*nxpad*nzpad + iz*nxpad + ix - 1;
                        uo[addr] = um[taddr] + (um[addr] - uo[taddr]) *
                                   d_bzh[iy*nzpad + iz];
                }

                if (iy <= NOP-iop) {
                        size_t taddr = (size_t)(iy+1)*nxpad*nzpad + iz*nxpad + ix;
                        uo[addr] = um[taddr] + (um[addr] - uo[taddr]) *
                                   d_byl[iz*nxpad + ix];
                }
                if (iy >= nypad-NOP-1+iop) {
                        size_t taddr = (size_t)(iy-1)*nxpad*nzpad + iz*nxpad + ix;
                        uo[addr] = um[taddr] + (um[addr] - uo[taddr]) *
                                   d_byh[iz*nxpad + ix];
                }
                
        }

}

__global__ void spongeKernel_2D(float *d_po, int nxpad, int nzpad, int nb){

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

__global__ void spongeKernel_3D(float *d_po, int nxpad, int nypad, int nzpad, int nb){

        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int z = threadIdx.z + blockIdx.z * blockDim.z;

        float alpha = 0.90;
        double damp;
        int i = 1;

        // apply sponge
        if (x < nxpad && y < nypad && z < nzpad) {

                size_t addr = (size_t)y * nxpad * nzpad + (size_t)z * nxpad + x;

                // apply to low values
                if (x < nb || y < nb || z < nb){

                        if (x < nb) { i = nb - x; }
                        else if (y < nb) { i = nb - y; }
                        else { i = nb - z; }

                        float fb = i / (sqrt(2.0)*(4.0*nb));
                        damp = exp(-fb * fb);
                        damp = exp(-1.0*fabs((pow((i-1.0),2)*log(alpha))/(pow(nb,2))));
                        d_po[addr] *= damp;

                }
                // apply to high values
                // NOTE: even though this is applied to all surfaces it only influences
                //       high th due to high ra being a free surface
                else if (x > nxpad - nb || y > nypad - nb || z > nzpad - nb) {

                        if (x > nxpad - nb) { i = x - (nxpad - nb); }
                        else if (y > nypad - nb) { i = y - (nypad - nb);}
                        else { i = z - (nzpad - nb); }

                        float fb = i / (sqrt(2.0)*(4.0*nb));
                        damp = exp(-fb * fb);
                        damp = exp(-1.0*fabs((pow((i-1.0),2)*log(alpha))/(pow(nb,2))));
                        d_po[addr] *= damp;

                }

        }

}


__global__ void spongeKernelOLD_3D(float *d_po, int nrapad, int nphpad, int nthpad, int nb){

	int ira = threadIdx.x + blockIdx.x * blockDim.x;
	int iph = threadIdx.y + blockIdx.y * blockDim.y;
	int ith = threadIdx.z + blockIdx.z * blockDim.z;

	float alpha = 0.90;

	// apply sponge
	if (ira < nrapad && iph < nphpad && ith < nthpad) {
        
		int addr = iph * nthpad * nrapad + ith * nrapad + ira;
		int i;

		// apply to low values
		// this is temporarily avoiding iph due to artifacts
		if (ira < nb || iph < nb || ith < nb){

			if      (ira < nb) {i = nb - ira;}
			else if (iph < nb) {i = nb - iph;}
			else if (ith < nb) {i = nb - ith;}
			
			// dampining funct 1
			//double damp = exp(-1.0*fabs(((i-1.0)*log(alpha))/nb)); 
			
			// dampining funct 2
			double damp = exp(-1.0*fabs((pow((i-1.0),2)*log(alpha))/(pow(nb,2))));

			// damping funct 3
			//double damp = ((1-alpha)/2)*cos(i*(PI/(0.5*nb)))+0.5+alpha/2;

			d_po[addr] *= damp;
		
		} 
		// apply to high values
		// do not apply to high values of ra where free surface lies
		// this wouldn't make a difference if it was applied, just
		// makes the code shorter

		// this is also temporarily ignoring the iph > nphpad - nb condition
		// due to artifacts
		if (ith > nthpad - nb || iph > nphpad - nb) {
			
			if      (iph > nphpad - nb) {i = iph - (nphpad - nb);}
                        else                        {i = ith - (nthpad - nb);}

			// dampining funct 1
			//double damp = exp(-1.0*fabs(((i-1.0)*log(alpha))/nb));

                        // dampining funct 2
                        double damp = exp(-1.0*fabs((pow((i-1.0),2)*log(alpha))/(pow(nb,2))));

			//double damp = ((1-alpha)/2)*cos(i*(PI/(0.5*nb)))+0.5+alpha/2;

			d_po[addr] *= damp;

		}
	}
}
