#include <math.h>

// definitions for multithreading
#ifdef _OPENMP
#include <omp.h>
#include "omputil.h"
#endif

// macro for 1d index to simulate a 2d/3d matrix
#define INDEX2D(ix, iz, nx) ((ix)+(iz)*(nx))
#define INDEX3D(ix, iy, iz, nx, nz) ((ix)+(iz)*(nx)+(iy)*(nz)*(nx))

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

// function to inject pressure from sources into pressure array
void inject_sources_2D_const(float *h_po, float *h_ww,
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

void inject_sources_2D(float *h_po, float *h_ww, float *h_vel,
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

                float r;

                // compute the local reflectivity
                // first find a velocity nearby which does not equal the injection spot velocity
                if (h_vel[ s_z      * nxpad + s_x    ] != h_vel[(s_z + 1) * nxpad + s_x    ]) {
                r = fabs((h_vel[ s_z      * nxpad + s_x    ] - h_vel[(s_z + 1) * nxpad + s_x    ]) / (h_vel[ s_z      * nxpad + s_x    ] + h_vel[(s_z + 1) * nxpad + s_x    ]));
                } else if (h_vel[ s_z      * nxpad + s_x    ] != h_vel[ s_z      * nxpad + s_x + 1]) {
                r = fabs((h_vel[ s_z      * nxpad + s_x    ] - h_vel[ s_z      * nxpad + s_x + 1]) / (h_vel[ s_z      * nxpad + s_x    ] + h_vel[ s_z      * nxpad + s_x + 1]));
                } else if (h_vel[ s_z      * nxpad + s_x    ] != h_vel[(s_z + 1) * nxpad + s_x + 1]) {
                r = fabs((h_vel[ s_z      * nxpad + s_x    ] - h_vel[(s_z + 1) * nxpad + s_x + 1]) / (h_vel[ s_z      * nxpad + s_x    ] + h_vel[(s_z + 1) * nxpad + s_x + 1]));
                } else if (h_vel[ s_z      * nxpad + s_x    ] != h_vel[ s_z      * nxpad + s_x - 1]) {
                r = fabs((h_vel[ s_z      * nxpad + s_x    ] - h_vel[ s_z      * nxpad + s_x - 1]) / (h_vel[ s_z      * nxpad + s_x    ] + h_vel[ s_z      * nxpad + s_x - 1]));
                } else if (h_vel[ s_z      * nxpad + s_x    ] != h_vel[(s_z - 1) * nxpad + s_x - 1]) {
                r = fabs((h_vel[ s_z      * nxpad + s_x    ] - h_vel[(s_z - 1) * nxpad + s_x - 1]) / (h_vel[ s_z      * nxpad + s_x    ] + h_vel[(s_z - 1) * nxpad + s_x - 1]));
                } else if (h_vel[ s_z      * nxpad + s_x    ] != h_vel[(s_z + 1) * nxpad + s_x - 1]) {
                r = fabs((h_vel[ s_z      * nxpad + s_x    ] - h_vel[(s_z + 1) * nxpad + s_x - 1]) / (h_vel[ s_z      * nxpad + s_x    ] + h_vel[(s_z + 1) * nxpad + s_x - 1]));
                } else if (h_vel[ s_z      * nxpad + s_x    ] != h_vel[(s_z - 1) * nxpad + s_x + 1]) {
                r = fabs((h_vel[ s_z      * nxpad + s_x    ] - h_vel[(s_z - 1) * nxpad + s_x + 1]) / (h_vel[ s_z      * nxpad + s_x    ] + h_vel[(s_z - 1) * nxpad + s_x + 1]));
                } else {
                r = 0;
                }

                h_po[ s_z      * nxpad + s_x    ] += wa * Sw00[ss] * r;
                h_po[(s_z + 1) * nxpad + s_x    ] += wa * Sw01[ss] * r;
                h_po[ s_z      * nxpad + s_x + 1] += wa * Sw10[ss] * r;
                h_po[(s_z + 1) * nxpad + s_x + 1] += wa * Sw11[ss] * r;

        }

}

// function to inject many sources
void inject_sources_3D_const(float *po, float *ww, 
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

void inject_sources_3D(float *po, float *ww, float *h_vel,
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

                float r;

                // cardinal directions
                if        (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[s_y*xz + (s_z+1)*nxpad     + s_x  ]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[s_y*xz + (s_z+1)*nxpad     + s_x  ]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[s_y*xz + (s_z+1)*nxpad     + s_x  ]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[(s_y+1)*xz + s_z*nxpad     + s_x  ]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[(s_y+1)*xz + s_z*nxpad     + s_x  ]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[(s_y+1)*xz + s_z*nxpad     + s_x  ]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[s_y*xz + s_z*nxpad         + s_x+1]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[s_y*xz + s_z*nxpad         + s_x+1]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[s_y*xz + s_z*nxpad         + s_x+1]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[s_y*xz + (s_z-1)*nxpad     + s_x  ]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[s_y*xz + (s_z-1)*nxpad     + s_x  ]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[s_y*xz + (s_z-1)*nxpad     + s_x  ]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[(s_y-1)*xz + s_z*nxpad     + s_x  ]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[(s_y-1)*xz + s_z*nxpad     + s_x  ]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[(s_y-1)*xz + s_z*nxpad     + s_x  ]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[s_y*xz + s_z*nxpad         + s_x-1]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[s_y*xz + s_z*nxpad         + s_x-1]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[s_y*xz + s_z*nxpad         + s_x-1]));
                }
                // diagonals
                else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[s_y*xz + (s_z+1)*nxpad     + s_x+1]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[s_y*xz + (s_z+1)*nxpad     + s_x+1]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[s_y*xz + (s_z+1)*nxpad     + s_x+1]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[s_y*xz + (s_z-1)*nxpad     + s_x-1]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[s_y*xz + (s_z-1)*nxpad     + s_x-1]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[s_y*xz + (s_z-1)*nxpad     + s_x-1]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[s_y*xz + (s_z+1)*nxpad     + s_x-1]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[s_y*xz + (s_z+1)*nxpad     + s_x-1]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[s_y*xz + (s_z+1)*nxpad     + s_x-1]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[s_y*xz + (s_z-1)*nxpad     + s_x+1]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[s_y*xz + (s_z-1)*nxpad     + s_x+1]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[s_y*xz + (s_z-1)*nxpad     + s_x+1]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[(s_y+1)*xz + (s_z+1)*nxpad + s_x  ]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[(s_y+1)*xz + (s_z+1)*nxpad + s_x  ]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[(s_y+1)*xz + (s_z+1)*nxpad + s_x  ]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[(s_y-1)*xz + (s_z-1)*nxpad + s_x  ]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[(s_y-1)*xz + (s_z-1)*nxpad + s_x  ]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[(s_y-1)*xz + (s_z-1)*nxpad + s_x  ]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[(s_y+1)*xz + (s_z-1)*nxpad + s_x  ]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[(s_y+1)*xz + (s_z-1)*nxpad + s_x  ]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[(s_y+1)*xz + (s_z-1)*nxpad + s_x  ]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[(s_y-1)*xz + (s_z+1)*nxpad + s_x  ]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[(s_y-1)*xz + (s_z+1)*nxpad + s_x  ]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[(s_y-1)*xz + (s_z+1)*nxpad + s_x  ]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[(s_y+1)*xz + s_z*nxpad     + s_x+1]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[(s_y+1)*xz + s_z*nxpad     + s_x+1]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[(s_y+1)*xz + s_z*nxpad     + s_x+1]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[(s_y-1)*xz + s_z*nxpad     + s_x-1]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[(s_y-1)*xz + s_z*nxpad     + s_x-1]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[(s_y-1)*xz + s_z*nxpad     + s_x-1]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[(s_y+1)*xz + s_z*nxpad     + s_x-1]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[(s_y+1)*xz + s_z*nxpad     + s_x-1]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[(s_y+1)*xz + s_z*nxpad     + s_x-1]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[(s_y-1)*xz + s_z*nxpad     + s_x+1]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[(s_y-1)*xz + s_z*nxpad     + s_x+1]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[(s_y-1)*xz + s_z*nxpad     + s_x+1]));
                } 
                // corners
                else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[(s_y+1)*xz + s_z*nxpad     + s_x-1]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[(s_y+1)*xz + s_z*nxpad     + s_x-1]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[(s_y+1)*xz + s_z*nxpad     + s_x-1]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[(s_y+1)*xz + (s_z+1)*nxpad  + s_x+1]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[(s_y+1)*xz + (s_z+1)*nxpad + s_x+1]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[(s_y+1)*xz + (s_z+1)*nxpad + s_x+1]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[(s_y+1)*xz + (s_z+1)*nxpad  + s_x-1]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[(s_y+1)*xz + (s_z+1)*nxpad + s_x-1]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[(s_y+1)*xz + (s_z+1)*nxpad + s_x-1]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[(s_y+1)*xz + (s_z-1)*nxpad  + s_x+1]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[(s_y+1)*xz + (s_z-1)*nxpad + s_x+1]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[(s_y+1)*xz + (s_z-1)*nxpad + s_x+1]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[(s_y+1)*xz + (s_z-1)*nxpad  + s_x-1]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[(s_y+1)*xz + (s_z-1)*nxpad + s_x-1]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[(s_y+1)*xz + (s_z-1)*nxpad + s_x-1]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[(s_y-1)*xz + (s_z+1)*nxpad  + s_x+1]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[(s_y-1)*xz + (s_z+1)*nxpad + s_x+1]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[(s_y-1)*xz + (s_z+1)*nxpad + s_x+1]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[(s_y-1)*xz + (s_z+1)*nxpad  + s_x-1]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[(s_y-1)*xz + (s_z+1)*nxpad + s_x-1]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[(s_y-1)*xz + (s_z+1)*nxpad + s_x-1]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[(s_y-1)*xz + (s_z-1)*nxpad  + s_x+1]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[(s_y-1)*xz + (s_z-1)*nxpad + s_x+1]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[(s_y-1)*xz + (s_z-1)*nxpad + s_x+1]));
                } else if (h_vel[s_y*xz + s_z*nxpad         + s_x  ] != h_vel[(s_y-1)*xz + (s_z-1)*nxpad  + s_x-1]) {
                        r = fabs((h_vel[s_y*xz + s_z*nxpad         + s_x  ] - h_vel[(s_y-1)*xz + (s_z-1)*nxpad + s_x-1]) / (h_vel[s_y*xz + s_z*nxpad         + s_x  ] + h_vel[(s_y-1)*xz + (s_z-1)*nxpad + s_x-1]));
                } else {
                        r = 0;
                }

                po[s_y*xz + s_z*nxpad         + s_x  ] += wa * Sw000[ss] * r;
                po[s_y*xz + (s_z+1)*nxpad     + s_x  ] += wa * Sw001[ss] * r;
                po[s_y*xz + s_z*nxpad         + s_x+1] += wa * Sw010[ss] * r;
                po[s_y*xz + (s_z+1)*nxpad     + s_x+1] += wa * Sw011[ss] * r;
                po[(s_y+1)*xz + s_z*nxpad     + s_x  ] += wa * Sw100[ss] * r;
                po[(s_y+1)*xz + (s_z+1)*nxpad + s_x  ] += wa * Sw101[ss] * r;
                po[(s_y+1)*xz + s_z*nxpad     + s_x+1] += wa * Sw110[ss] * r;
                po[(s_y+1)*xz + (s_z+1)*nxpad + s_x+1] += wa * Sw111[ss] * r;

        }
}



#define NOP 4
// function to solve for next timestep

void solve_2D(float *fpo, float *po, float *ppo, float *vel,
           float dra, float dth, float ora, float oth, float dt, 
           int nrapad, int nthpad) {

        int ira, ith, addr;
        float laplace, compra, compth, ra;
        float v;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
        private(ira, ith, ra, laplace, compra, compth, v, addr) \
        shared(fpo, po, ppo, vel, dra, dth, ora, oth, dt, nrapad, nthpad)
#endif

        for (ira = 0; ira < nrapad; ira++) {
                for (ith = 0; ith < nthpad; ith++) {
                        
                        addr = ith * nrapad + ira; // compute 1d index address of location

                        // find true location using deltas and indicies
                        ra = dra * ira + ora;

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

void solve_3D(float *fpo, float *po, float *ppo, float *vel,
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
                float ra; float th;
                ra = dra * ira + ora;
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


// function to shift pressure fields in time
void shift_2D(float *fpo, float *po, float *ppo, int nrapad, int nthpad) {

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


// func to shift pressure fields
void shift_3D(float *fpo, float *po, float *ppo,
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

void onewayBC_2D(float *uo, float *um,
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
                        else if (ira > nrapad - NOP) {iop = nrapad - ira;}
                        else if (ith < NOP) {iop = ith;}
                        else if (ith > nthpad - NOP) {iop = nthpad - ith;}
                        else {iop = 0;}

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

void onewayBC_3D(float *uo, float *um,
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
                else if (ix > nxpad - NOP) {iop = nxpad - ix;}
                else if (iz < NOP) {iop = iz;}
                else if (iz > nzpad - NOP) {iop = nzpad - iz;}
                else if (iy < NOP) {iop = iy;}
                else if (iy > nypad - NOP) {iop = nypad - iy;}
                else {iop = 0;}

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
void spongeBC_2D(float *po, int nxpad, int nzpad, int nb){

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

void spongeBC_3D(float *po, int nxpad, int nypad, int nzpad, int nb){

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
void freeSurf_2D(float *po, int nrapad, int nthpad, int nb) {

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

void freeBC_3D(float *po, int nrapad, int nphpad, int nthpad, int nb) {

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


// kernel to extract data to receivers
void extract_2D(float *data, 
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


void extract_3D(float *dd_pp, 
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