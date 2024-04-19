#include <math.h>
#include <rsf.h>

// macro for 1d index to simulate a 3d matrix
#define INDEX2D(ix, iz, nx) ((ix)+(iz)*(nx))

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

// function to inject pressure from sources into pressure array
void inject_sources(float *h_po, float *h_ww,
                    float *Sw00, float *Sw01, float *Sw10, float *Sw11,
                    int *Sjx, int *Sjz,
                    int it, int ns,
                    int nxpad, int nzpad) {

        float wa = h_ww[it];
        int s_x, s_z; 
        int ss;

        // iterate over sources
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) private(ss, s_x, s_z) shared(it, h_po, wa, nxpad, nzpad, Sjx, Sjz, Sw00, Sw01, Sw10, Sw11)
#endif
        for (ss = 0; ss < ns; ss++) {

                s_x = Sjx[ss];
                s_z = Sjz[ss];

                h_po[ s_z      * nxpad + s_x    ] += wa * Sw00[ss];
		h_po[(s_z + 1) * nxpad + s_x    ] += wa * Sw01[ss];
		h_po[ s_z      * nxpad + s_x + 1] += wa * Sw10[ss];
		h_po[(s_z + 1) * nxpad + s_x + 1] += wa * Sw11[ss];

        }

}


#define NOP 4 // buffer region to not do operations
// solve function which updates the wavefield
void solve(float *fpo, float *po, float *ppo, float *vel,
           float dra, float dth, float ora, float oth, float dt, 
           int nrapad, int nthpad) {

        int ira, ith, addr;
        float laplace, compra, compth, ra, th;
        float v;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
        private(ira, ith, ra, th, laplace, compra, compth, v, addr) \
        shared(fpo, po, ppo, vel, dra, dth, ora, oth, dt, nrapad, nthpad)
#endif

        for (ira = 0; ira < nrapad; ira++) {
                for (ith = 0; ith < nthpad; ith++) {
                        
                        addr = ith * nrapad + ira; // compute 1d index address of location

                        // find true location using deltas and indicies
                        ra = dra * ira + ora;
                        th = dth * ith + oth;

                        // extract true velocity
                        v = vel[addr];

                        if (ira >= NOP && ira < nrapad-NOP && ith >= NOP && ith < nthpad - NOP) {

                                // calculate polar laplacian
                                
                                // calc derivates with respect to ra
                                compra = ((1/(dra*dra))+(1/(2*ra*dra))) * po[INDEX2D(ira-1,ith,nrapad)] + 
                                        (-2/(dra*dra))                 * po[addr] +
                                        ((1/(dra*dra))-(1/(2*ra*dra))) * po[INDEX2D(ira+1,ith,nrapad)];
                                
                                // calc derivatives with respect to th
                                compth = ((1/(ra*ra*dth*dth))) * po[INDEX2D(ira,ith-1,nrapad)] + 
                                        (-2/(ra*ra*dth*dth))  * po[addr] + 
                                        ((1/(ra*ra*dth*dth))) * po[INDEX2D(ira,ith+1,nrapad)];

                                // SUM TO GET LAPLACIAN
                                laplace = compra + compth;

                        } else {
                                laplace = 0.;
                        }

                        // use laplacian to calculate next time step of pressure
                        fpo[addr] = (dt*dt) * (v*v) * laplace + 2*po[addr] - ppo[addr];

                }
        }

}


// function to shift pressure fields in time
void shift(float *fpo, float *po, float *ppo, int nrapad, int nthpad) {

        int ira, ith;
        int addr;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
        private(ira, ith, addr) \
        shared(fpo, po, ppo, nrapad, nthpad)
#endif

        for (ira = 0; ira < nrapad; ira++) {
                for (ith = 0; ith < nthpad; ith++) {

                        addr = ith * nrapad + ira;

                        // move arrays
                        ppo[addr] = po[addr];
                        po[addr] = fpo[addr];

                }
        }
}


// function which does one way boundary conditions
void onewayBC(float *uo, float *um,
              float *bthl, float *bthh, float *bral, float *brah,
              int nrapad, int nthpad) {

	int ira, ith, iop, addr;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
        private(ira, ith, addr, iop) \
        shared(uo, um, bthl, bthh, bral, brah, nrapad, nthpad)
#endif

	for (ira = 0; ira < nrapad; ira++) {
                for (ith = 0; ith < nthpad; ith++) {

                        addr = ith * nrapad + ira;

                        if (ira < NOP) {iop = ira;}
                        if (ira > nrapad - NOP) {iop = nrapad - ira;}

                        if (ith < NOP) {iop = ith;}
                        if (ith > nthpad - NOP) {iop = nthpad - ith;}
                        
                        // top bc
                        if (ith == NOP-iop) {
                                uo[addr] =  um[(ith+1)*nrapad+ira] +
                                                (um[addr] - uo[(ith+1)*nrapad+ira]) * bthl[ira];
                        }
                        // bottom bc
                        if (ith == nthpad-NOP+iop-1) {
                                uo[addr] =  um[(ith-1)*nrapad+ira] +
                                                (um[addr] - uo[(ith-1)*nrapad+ira]) * bthh[ira];
                        }
                        
                        // left bc
                        if (ira == NOP-iop) {
                                uo[addr] =  um[ith*nrapad+ira+1] +
                                                (um[addr] - uo[ith*nrapad+ira+1]) * bral[ith];
                        }
                        // bottom bc
                        if (ira == nrapad-NOP+iop-1) {
                                uo[addr] =  um[ith*nrapad+ira-1] +
                                                (um[addr] - uo[ith*nrapad+ira-1]) * brah[ith];
                        }		

	        }
        }

}


// function for basic sponge boundary conditions
void spongeBC(float *po, int nxpad, int nzpad, int nb){

        int x, z, i, addr;

        float alpha = 0.90;
        double damp;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
        private(x, z, i, addr, damp) \
        shared(alpha, po, nxpad, nzpad, nb)
#endif

        for (x = 0; x < nxpad; x++) {
                for (z = 0; z < nzpad; z++) {

                        addr = z * nxpad + x;

                        // apply to low values
                        if (x < nb || z < nb){

                                if (x < nb) { i = nb - x; }
                                else { i = nb - z; }

                                float fb = i / (sqrt(2.0)*(4.0*nb));
                                damp = exp(-fb * fb);
                                damp = exp(-1.0*fabs((pow((i-1.0),2)*log(alpha))/(pow(nb,2))));
                                po[addr] *= damp;

                        }

                        // apply to high values
                        else if (x > nxpad - nb || z > nzpad - nb) {

                                if (x > nxpad - nb) { i = x - (nxpad - nb); }
                                else { i = z - (nzpad - nb); }

                                float fb = i / (sqrt(2.0)*(4.0*nb));
                                damp = exp(-fb * fb);
                                damp = exp(-1.0*fabs((pow((i-1.0),2)*log(alpha))/(pow(nb,2))));
                                po[addr] *= damp;

                        }
                }
        }
}


// function to define free surface
void freeSurf(float *po, int nrapad, int nthpad, int nb) {

        int ira, ith, addr;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
        private(ira, ith) \
        shared(po, nrapad, nthpad, nb)
#endif

        for (ira = 0; ira < nrapad; ira++) {
                for (ith = 0; ith < nthpad; ith++) {
                        if (ith < nthpad && ira > nrapad - nb) {
                                addr = ith * nrapad + ira;
                                po[addr] = 0;
                        }
                }
        }
}


// kernel to extract data to receivers
void extract(float *data, 
                int it, int nr,
                int nrapad, int nthpad,
                float *po, int *Rjra, int *Rjth,
                float *Rw00, float *Rw01, float *Rw10, float *Rw11) {

	// receiver number
	int rr;
	// time offset
	int offset = it * nr;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
        private(rr) \
        shared(nr, offset, nrapad, po, Rjra, Rjth, Rw00, Rw01, Rw10, Rw11)
#endif

	for (rr = 0; rr < nr; rr++){

		int th_comp   = (Rjth[rr]) * nrapad;
		int th_comp_1 = (Rjth[rr]+1) * nrapad;
		
		data[offset + rr] = po[th_comp   + (Rjra[rr])]   * Rw00[rr] +
                                    po[th_comp_1 + (Rjra[rr])]   * Rw01[rr] +
                                    po[th_comp   + (Rjra[rr]+1)] * Rw10[rr] +
                                    po[th_comp_1 + (Rjra[rr]+1)] * Rw11[rr];


	}

}