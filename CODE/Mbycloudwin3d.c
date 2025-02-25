/*
3D CLoud WINdowing
Paul Sava
Copyright (C) 2022 Colorado School of Mines
*/
#include <rsf.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

#include <complex.h>

#define NCO 9
#define NCG 3

/*------------------------------------------------------------*/
bool isInCone(pt3d *oco, pt3d *gco, vc3d *gno, float cosapt)
{
  float Rx,Ry,Rz,R, cosobl;

  // cone axis vector
  Rx = oco->x - gco->x;
  Ry = oco->y - gco->y;
  Rz = oco->z - gco->z;
  R = sqrtf( Rx * Rx + Ry * Ry + Rz * Rz );
  Rx /= R;
  Ry /= R;
  Rz /= R;

  cosobl = SF_ABS( Rx * gno->dx + Ry * gno->dy + Rz * gno->dz );
  if( cosobl > cosapt) return true;
  else                 return false;
}

/*------------------------------------------------------------*/
int main(int argc, char* argv[])
{
  bool verb, fast;
  float apt, cosapt;         /* aperture */

  /* I/O files */
  sf_file Fin = NULL;        /* input   cloud */
  sf_file Fgou = NULL;        /* output  cloud */
  sf_file Fdat = NULL;       /* modeled data */
  sf_file Fdou = NULL;       /* output for modeled data */
  sf_file Fo  = NULL;        /* orbit   */

  sf_axis ao,ag,aw,af;          /* cube axes */
  size_t  no,ng,nw,nf;
  size_t  io,ig,iw;
  size_t ncount = 0;
  int ico, iff;

  pt3d       * oco   = NULL; /* orbit coordinates  */
  pt3d         gco;          /* ground coordinate  */
  vc3d         gno;          /* ground normal */
  float      * jnk   = NULL;
  float      * dgin   = NULL;
  sf_complex * ddin   = NULL; 
  float      * dgou   = NULL;
  sf_complex * ddou   = NULL;
  bool       * fin   = NULL;
  off_t      * gwmap = NULL;

  jnk = sf_floatalloc( NCO );

  /*------------------------------------------------------------*/
  /* init RSF */
  sf_init(argc,argv);

  /* OMP init */
  #ifdef _OPENMP
    omp_init();
  #endif

  /* default behavior */
  if (!sf_getbool ("verb",&verb)) verb=true; /* verbosity  */
  if (!sf_getbool ("fast",&fast)) fast=true;  /* in-core windowing  */
  if (!sf_getfloat("apt", &apt))   apt=15.0;  /* aperture (deg) */
  cosapt = cos(apt*SF_PI/180);                /* cos(aperture) */
  
  verb = true;
  if(verb) {
    sf_warning("apt=%6.2f",apt);
  }

  /* setup i/o and auxiliary file */
  Fin = sf_input ( "in");
  Fo  = sf_input ( "oo");
  Fdat = sf_input("dat");
  Fgou = sf_output("out");
  Fdou = sf_output("dout");

  // set the data output as complex type
  sf_settype(Fdou, SF_COMPLEX);

  /* coordinate axes */
  ao = sf_iaxa(Fo ,2); sf_setlabel(ao,"o"); /* orbit */
  af = sf_iaxa(Fdat,2); sf_setlabel(af,"t"); /* time axis*/
  ag = sf_iaxa(Fin,2); sf_setlabel(ag,"g"); /* ground input */
  if(verb) sf_raxa(ao);
  if(verb) sf_raxa(ag);

  no = sf_n(ao);
  ng = sf_n(ag);
  nf = sf_n(af);

  /*------------------------------------------------------------*/
  /* orbit coordinates */
  oco = (pt3d*) sf_alloc(no,sizeof(*oco));

  for( io = 0; io < no; io++) {
    sf_floatread( jnk,NCO,Fo);
    oco[io].x = jnk[0];
    oco[io].y = jnk[1];
    oco[io].z = jnk[2];
  }

  /* ground coordinates */
  if(fast) {
    /*------------------------------------------------------------*/
    /* FAST */
    /*------------------------------------------------------------*/

    // read input
    if(verb) sf_warning("read ground points");
    // first read in receiver coords/ground points
    dgin = sf_floatalloc( (size_t)ng * NCO );
    sf_floatread(dgin, (size_t)ng * NCO, Fin);
    if(verb) sf_warning("read model data");
    // now read in modeled data
    // this requires A LOT of memory
    sf_warning("ng: %d", ng);
    sf_warning("nf: %d", nf);
    ddin = sf_complexalloc( (size_t)ng * nf);
    sf_complexread(ddin, (size_t)ng * nf, Fdat);

    // flag window points
    if(verb) sf_warning("flag window points");
    fin = sf_boolalloc ( ng );
#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic) \
  private( ig, io, gco, gno ) \
  shared(  ng, no, dgin, fin, oco, cosapt )
#endif
    for( ig = 0; ig < ng; ig++) {
      fin[ig] = 0;

      gco.x  = dgin[ig * NCO + 0];
      gco.y  = dgin[ig * NCO + 1];
      gco.z  = dgin[ig * NCO + 2];

      gno.dx = dgin[ig * NCO + 3];
      gno.dy = dgin[ig * NCO + 4];
      gno.dz = dgin[ig * NCO + 5];

      for( io = 0; io < no; io++ ) {
        if( isInCone( &oco[io], &gco, &gno, cosapt) ) {
          fin[ig] = 1;
          break;
        }
      }
    }

    // count window points
    if(verb) sf_warning("count window points");
    nw = 0;
#ifdef _OPENMP
  #pragma omp parallel for reduction(+ : nw)
#endif
    for( ig = 0; ig < ng; ig++) nw += fin[ig];

    // write window header for Fgou
    if(verb) sf_warning("write window header");
    aw = sf_maxa(nw,0,1);
    sf_oaxa(Fgou,aw,2);
    if(verb) sf_raxa(aw);

    // write header for Fdou
    sf_oaxa(Fdou, af, 2);
    sf_oaxa(Fdou, aw, 1);

    // avoid empty window
    nw = SF_MAX(nw,1);

    // index window points
    if(verb) sf_warning("index window points");
    dgou = sf_floatalloc( nw * NCO );
    ddou = sf_complexalloc( nw * nf);
    for(int i = 0; i < nw * NCO; i++) dgou[i] = 0.0;

    // keep indices in all cloud
    gwmap = sf_largeintalloc( nw );
    iw = 0;
    for( ig = 0; ig < ng; ig++) {
      if( fin[ig] == 1 ) {
        gwmap[ iw ] = ig;
        iw++;
      }
    }

    sf_warning("nw: %d", nw);

    // move window points for ground data
    if(verb) sf_warning("move window points");
#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic) \
  private( iw, ig, ico) \
  shared(  nw, dgou, dgin, gwmap)
#endif
    for( iw = 0; iw < nw; iw++ ) {
      ig = gwmap[ iw ];
      for(ico = 0; ico < NCO; ico++) {
        dgou[iw * NCO + ico] = dgin[ig * NCO + ico];
      }
    }

    // now do the same for the modeled data
    if(verb) sf_warning("move window data");
#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic) \
  private( iw, ig, iff) \
  shared(  nw, nf, dgou, dgin, gwmap)
#endif
    for( iw = 0; iw < nw; iw++ ) {
      ig = gwmap[ iw ];
      for(iff = 0; iff < nf; iff++) {
        ddou[iw * nf + iff] = ddin[ig * nf + iff];
      }
    }    

    // write window points
    if(verb) sf_warning("write window points");
    sf_floatwrite(dgou, nw * NCO, Fgou);
    sf_complexwrite(ddou, nw * nf,  Fdou);

    // deallocate arrays
    free(gwmap);
    free(dgin);
    free(dgou);
    free(ddin);
    free(ddou);

  } else {
    /*------------------------------------------------------------*/
    /* SLOW */
    /*------------------------------------------------------------*/

    for( int ipass = 0; ipass < 2; ipass++) {
      sf_seek(Fin,0,SEEK_SET); // seek to the start of the input file

      for( ig = 0; ig < sf_n(ag); ig++) {

        sf_floatread(jnk, NCO, Fin);
        gco.x  = jnk[0];
        gco.y  = jnk[1];
        gco.z  = jnk[2];

        gno.dx = jnk[3];
        gno.dy = jnk[4];
        gno.dz = jnk[5];

        for( io = 0; io < sf_n(ao); io++ ) {
          if( isInCone( &oco[io], &gco, &gno, cosapt) ) {
            if( ipass == 0) ncount++;          // count points
            else sf_floatwrite(jnk, NCO, Fgou); // write points
            break;
          }
        }

      }

      if(ipass == 0) { // make output axis after first pass
        aw = sf_maxa( SF_MAX(ncount,1),0,1);
        sf_oaxa(Fgou,aw,2);
        if(verb) sf_raxa(aw);

        //if(nw == 0) exit(0);
      }

    } // ipass

  }

  /*------------------------------------------------------------*/
  /* deallocate arrays */
  free(jnk);
  free(oco);

  exit (0);
}
