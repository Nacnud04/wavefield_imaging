#include <math.h>

// definitions for multithreading
#ifdef _OPENMP
#include <omp.h>
#include "omputil.h"
#endif

// macro for 1d index to simulate a 3d matrix
#define INDEX3D(ix, iy, iz, nx, nz) ((ix)+(iz)*(nx)+(iy)*(nz)*(nx))

// expand domain
void expand_cpu_3d(float *a, float *b, int nb, int x_a, int x_b, int y_a, int y_b, int z_a, int z_b){

    int ix, iy, iz;

    // copy into other array
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
        private(ix, iy, iz) \
        shared(a, b, nb, x_a, x_b, y_a, y_b, z_a, z_b)
#endif
	for (ix = 0; ix < x_a; ix++) {
        for (iy = 0; iy < y_a; iy++) {
            for (iz = 0; iz < z_a; iz++) {
                b[INDEX3D(ix+nb,iy+nb,iz+nb,x_b,z_b)] = a[INDEX3D(ix,iy,iz,x_a,z_a)];
            }
        }
    }

    // expand z direction
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
        private(ix, iy, iz) \
        shared(a, b, nb, x_a, x_b, y_a, y_b, z_a, z_b)
#endif
	for (ix = 0; ix < x_b; ix++) {
        for (iy = 0; iy < y_b; iy++) {
            for (iz = 0; iz < nb; iz++) {
                b[INDEX3D(ix,iy,iz,x_b,z_b)] = b[INDEX3D(ix,iy,nb,x_b,z_b)];
                b[INDEX3D(ix,iy,z_b-iz-1,x_b,z_b)] = b[INDEX3D(ix,iy,z_b-nb-1,x_b,z_b)];
            }
        }
    }
	
    // expand y direction
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
        private(ix, iy, iz) \
        shared(a, b, nb, x_a, x_b, y_a, y_b, z_a, z_b)
#endif
	for (ix = 0; ix < x_b; ix++) {
        for (iy = 0; iy < nb; iy++) {
            for (iz = 0; iz < z_b; iz++) {
                b[INDEX3D(ix,iy,iz,x_b,z_b)] = b[INDEX3D(ix,nb,iz,x_b,z_b)];
                b[INDEX3D(ix,y_b-iy-1,iz,x_b,z_b)] = b[INDEX3D(ix,y_b-nb-1,iz,x_b,z_b)];
            }
        }
    }
	
    //expand x direction
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
        private(ix, iy, iz) \
        shared(a, b, nb, x_a, x_b, y_a, y_b, z_a, z_b)
#endif
	for (ix = 0; ix < nb; ix++) {
        for (iy = 0; iy < y_b; iy++) {
            for (iz = 0; iz < z_b; iz++) {
                b[INDEX3D(ix,iy,iz,x_b,z_b)] = b[INDEX3D(nb,iy,iz,x_b,z_b)];
                b[INDEX3D(x_b-ix-1,iy,iz,x_b,z_b)] = b[INDEX3D(x_b-nb-1,iy,iz,x_b,z_b)];
            }
        }
    }
	
}


// function to inject many sources
void inject_sources(float *po, float *ww, 
                float *Sw000, float *Sw001, float *Sw010, float *Sw011, 
                float *Sw100, float *Sw101, float *Sw110, float *Sw111, 
                int *Sjx, int *Sjy, int *Sjz, 
                int it, int ns,
                int nxpad, int nypad, int nzpad) {

    int ss;

    float wa = ww[it];

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
        private(ss) \
        shared(wa, po, Sw000, Sw001, Sw010, Sw011, Sw100, Sw101, Sw110, Sw111, Sjx, Sjy, Sjz, ns, nxpad, nypad, nzpad)
#endif

    for (ss = 0; ss < ns; ss++) {

        int s_x = Sjx[ss];
        int s_y = Sjy[ss];
        int s_z = Sjz[ss];

        int xz = nxpad * nzpad;

        po[s_y*xz + s_z*nxpad         + s_x  ] += wa * Sw000[ss];
        po[s_y*xz + (s_z+1)*nxpad     + s_x  ] += wa * Sw001[ss];
        po[s_y*xz + s_z*nxpad         + s_x+1] += wa * Sw010[ss];
        po[s_y*xz + (s_z+1)*nxpad     + s_x+1] += wa * Sw011[ss];
        po[(s_y+1)*xz + s_z*nxpad     + s_x  ] += wa * Sw100[ss];
        po[(s_y+1)*xz + (s_z+1)*nxpad + s_x  ] += wa * Sw101[ss];
        po[(s_y+1)*xz + s_z*nxpad     + s_x+1] += wa * Sw110[ss];
        po[(s_y+1)*xz + (s_z+1)*nxpad + s_x+1] += wa * Sw111[ss];

    }
}


#define NOP 4
// function to solve for next timestep
void solve(float *fpo, float *po, float *ppo, float *vel,
		      float dra, float dph, float dth, float ora, float oph, float oth, 
		      float dt,
		      int nrapad, int nphpad, int nthpad) {

	int ira, iph, ith, addr;
    float laplace, v;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
        private(ira, iph, ith, addr, laplace, v) \
        shared(fpo, po, ppo, vel, dra, dph, dth, ora, oph, oth, dt, nrapad, nphpad, nthpad)
#endif

    for (ira = 0; ira < nrapad; ira++) {
        for (ith = 0; ith < nthpad; ith++ ) {
            for (iph = 0; iph < nphpad; iph++) {
		
                addr = iph * nthpad * nrapad + ith * nrapad + ira;			  

                // extract true location from deltas and indicies
                float ra; float ph; float th;
                ra = dra * ira + ora;
                ph = dph * iph + oph;
                th = dth * ith + oth;
                
                // extract true velocity
                v  = vel[addr];

                // perform only in boundaries:
                if (ira >= NOP && ira < nrapad-NOP && iph >= NOP && iph < nphpad-NOP && ith >= NOP && ith < nthpad - NOP) {

                    // CALCULATE ALL SPATIAL DERIVS IN LAPLACIAN
                    float pra; 
                    pra =  po[INDEX3D(ira+1, iph, ith, nrapad, nthpad)] \
                        -po[INDEX3D(ira-1, iph, ith, nrapad, nthpad)];
                    pra = pra / (2 * dra);

                    float ppra;
                    ppra =    po[INDEX3D(ira+1, iph, ith,nrapad,nthpad)] \
                                    -2*po[INDEX3D(ira  , iph, ith,nrapad,nthpad)] \
                                        +po[INDEX3D(ira-1, iph, ith,nrapad,nthpad)];
                    ppra = ppra / (dra * dra);

                    float pth;
                    pth =  po[INDEX3D(ira, iph, ith+1, nrapad, nthpad)] \
                                    -po[INDEX3D(ira, iph, ith-1, nrapad, nthpad)];
                                pth = pth / (2 * dth);

                    float ppth;
                    ppth =    po[INDEX3D(ira, iph, ith+1,nrapad,nthpad)] \
                                    -2*po[INDEX3D(ira, iph, ith  ,nrapad,nthpad)] \
                                        +po[INDEX3D(ira, iph, ith-1,nrapad,nthpad)];
                                ppth = ppth / (dth * dth);

                    float ppph;
                                ppph =    po[INDEX3D(ira, iph+1, ith,nrapad,nthpad)] \
                                    -2*po[INDEX3D(ira, iph  , ith,nrapad,nthpad)] \
                                        +po[INDEX3D(ira, iph-1, ith,nrapad,nthpad)];
                                ppph = ppph / (dph * dph);

                    // COMBINE SPATIAL DERIVS TO CREATE LAPLACIAN
                    laplace =  (2/ra)*pra + ppra              // ra component
                            +(cos(th)/(ra*ra*sin(th)))*pth    // th component 1
                            +(1/(ra*ra))*ppth                 // th component 2
                            +(1/(ra*ra*sin(th)*sin(th)))*ppph;// ph component

                } else {
                    laplace = 0.;
                }

                // compute pressure at next time step
                fpo[addr] = (dt*dt) * (v*v) * laplace + 2*po[addr] - ppo[addr];

            }
        }
	}

}


// func to shift pressure fields
void shift(float *fpo, float *po, float *ppo,
		      int nrapad, int nphpad, int nthpad) {
	
	int ira, ith, iph, addr;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
        private(ira, iph, ith, addr) \
        shared(fpo, po, ppo, nrapad, nphpad, nthpad)
#endif
	for (ira = 0; ira < nrapad; ira++) {
        for (ith = 0; ith < nthpad; ith++ ) {
            for (iph = 0; iph < nphpad; iph++) {

                addr = iph * nthpad * nrapad + ith * nrapad + ira;
                
                // replace ppo with po and fpo with po
                ppo[addr] = po[addr];
                po[addr] = fpo[addr];

            }
        }
	}
}


// implementation of one way boundary conditions
void onewayBC(float *uo, float *um,
                float *bzl, float *bzh,
                float *bxl, float *bxh,
                float *byl, float *byh,
                int nxpad, int nypad, int nzpad) {

    int ix, iy, iz;
    int addr;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
        private(ix, iy, iz, addr) \
        shared(uo, um, bzl, bzh, bxl, bxh, byl, byh, nxpad, nypad, nzpad)
#endif

    for (ix = 0; ix < nxpad; ix++) {
        for (iz = 0; iz < nzpad; iz++) {
            for (iy = 0; iy < nypad; iy++) {

                addr = iy * nxpad * nzpad + iz * nxpad + ix;

                int iop;

                if (ix < NOP) {iop = ix;}
                if (ix > nxpad - NOP) {iop = nxpad - ix;}

                if (iz < NOP) {iop = iz;}
                if (iz > nzpad - NOP) {iop = nzpad - iz;}

                if (iy < NOP) {iop = iy;}
                if (iy > nypad - NOP) {iop = nypad - iy;}

                // top bc
                if (iz <= NOP - iop) {
                    int taddr = iy*nxpad*nzpad + (iz+1)*nxpad + ix;
                    uo[addr] = um[taddr] + (um[addr] - uo[taddr]) *
                                bzl[iy*nxpad + ix];
                }
                // bottom bc
                if (iz >= nzpad-NOP+iop-1) {
                    int taddr = iy*nxpad*nzpad + (iz-1)*nxpad + ix;
                    uo[addr] = um[taddr] + (um[addr] - uo[taddr]) *
                                bzh[iy*nxpad + ix];
                }

                if (ix <= NOP - iop) {
                    int taddr = iy*nxpad*nzpad + iz*nxpad + ix + 1;
                    uo[addr] = um[taddr] + (um[addr] - uo[taddr]) *                                                                  bxl[iy*nzpad + iz];
                }
                if (ix >= nxpad-NOP+iop-1) {
                    int taddr = iy*nxpad*nzpad + iz*nxpad + ix - 1;
                    uo[addr] = um[taddr] + (um[addr] - uo[taddr]) *
                                bzh[iy*nzpad + iz];
                }

                if (iy <= NOP - iop) {
                    int taddr = (iy+1)*nxpad*nzpad + iz*nxpad + ix;
                    uo[addr] = um[taddr] + (um[addr] - uo[taddr]) *
                                byl[iz*nxpad + ix];
                }
                if (iy >= nypad-NOP+iop-1) {
                    int taddr = (iy-1)*nxpad*nzpad + iz*nxpad + ix;
                    uo[addr] = um[taddr] + (um[addr] - uo[taddr]) *
                                byh[iz*nxpad + ix];
                }

            }
        }
    }

}


// function for sponge boundary conditions
void spongeBC(float *po, int nxpad, int nypad, int nzpad, int nb){

    int x, y, z;
    float alpha = 0.90;
    int i = 1;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
        private(x, y, z) \
        shared(po, alpha, i, nxpad, nypad, nzpad, nb)
#endif

    // apply sponge
    for (x = 0; x < nxpad; x++) {
        for (z = 0; z < nzpad; z++) {
            for (y = 0; y < nypad; y++) {

                int addr = y * nxpad * nzpad + z * nxpad + x;
                double damp;

                // apply to low values
                if (x < nb || y < nb || z < nb){

                        if (x < nb) { i = nb - x; }
                        else if (y < nb) { i = nb - y; }
                        else { i = nb - z; }

                        float fb = i / (sqrt(2.0)*(4.0*nb));
                        damp = exp(-fb * fb);
                        damp = exp(-1.0*fabs((pow((i-1.0),2)*log(alpha))/(pow(nb,2))));
                        po[addr] *= damp;

                }

                // apply to high values
                else if (x > nxpad - nb || y > nypad - nb || z > nzpad - nb) {

                        if (x > nxpad - nb) { i = x - (nxpad - nb); }
                        else if (y > nypad - nb) { i = y - (nypad - nb);}
                        else { i = z - (nzpad - nb); }

                        float fb = i / (sqrt(2.0)*(4.0*nb));
                        damp = exp(-fb * fb);
                        damp = exp(-1.0*fabs((pow((i-1.0),2)*log(alpha))/(pow(nb,2))));
                        po[addr] *= damp;

                }

            }
        }
    }

}



// free surface BC
void freeBC(float *po, int nrapad, int nphpad, int nthpad, int nb) {

    int ira, iph, ith;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
    private(ira, iph, ith) \
    shared(po, nrapad, nphpad, nthpad)
#endif

    for (ira=0; ira < nrapad; ira++) {
        for (ith=0; ith < nthpad; ith++) {
            for (iph=0; iph < nphpad; iph++) {

                int addr = iph * nthpad * nrapad + ith * nrapad + ira;
                po[addr] = 0;

            }
        }
    }
}


// extract receiver data
void extract(float *dd_pp, 
				   int it, int nr,
				   int nxpad, int nypad, int nzpad,
				   float *po, int *Rjx, int *Rjy, int *Rjz,
				   float *Rw000, float *Rw001, float *Rw010, float *Rw011, 
				   float *Rw100, float *Rw101, float *Rw110, float *Rw111) {

	int rr;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
    private(rr) \
    shared(dd_pp, po, Rjx, Rjy, Rjz, Rw000, Rw001, Rw010, Rw011, Rw100, Rw101, Rw110, Rw111)
#endif
	
	for (rr=0; rr<nr; rr++) {

        int offset = it * nr;

		int y_comp = Rjy[rr] * nxpad * nzpad;
		int y_comp_1 = (Rjy[rr]+1) * nxpad * nzpad;
		int z_comp = Rjz[rr] * nxpad;
		int z_comp_1 = (Rjz[rr]+1) * nxpad;
		dd_pp[offset + rr] = po[y_comp   + z_comp   + (Rjx[rr])]   * Rw000[rr] +
                                po[y_comp   + z_comp_1 + Rjx[rr]]     * Rw001[rr] +
                                po[y_comp   + z_comp   + (Rjx[rr]+1)] * Rw010[rr] +
                                po[y_comp   + z_comp_1 + (Rjx[rr]+1)] * Rw011[rr] +
                                po[y_comp_1 + z_comp   + (Rjx[rr])]   * Rw100[rr] +
                                po[y_comp_1 + z_comp_1 + Rjx[rr]]     * Rw101[rr] +
                                po[y_comp_1 + z_comp   + (Rjx[rr]+1)] * Rw110[rr] +
                                po[y_comp_1 + z_comp_1 + (Rjx[rr]+1)] * Rw111[rr];

	}

}