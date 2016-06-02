/*********************************************************************
run-cg.cu

Hauptprogramm. Testet Reduktion und ruft cg auf.

**********************************************************************/
#define MAIN_PROGRAM

#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "global.h"
#include "geometry.h"
#include "linalg.h"
#include "cg.h"

int main(int argc, char **argv)
{
   printf("%s Starting...\n", argv[0]);

   int nBytes, status, N;
   double *w, *v, *x, *s, *vo, *rnormalt, *svo, *r, *ro, *rnorm, *sv, *vro, *vr, *vso, *vs;
   double iStart, iElaps;

   N=32;
   int dimx = 256;
   int dimy = 1;
   if (argc>1)
   {
      N=atoi(argv[1]);
   }
   if (argc>3)
   {
      dimx=atoi(argv[2]);
      dimy=atoi(argv[3]);
   }

   // set up device
   int dev = 0;
   cudaDeviceProp deviceProp;
   CHECK(cudaGetDeviceProperties(&deviceProp, dev));
   printf("Using Device %d: %s\n", dev, deviceProp.name);
   CHECK(cudaSetDevice(dev));

   // Globale Variablen setzen:
   // Anzahl der Inneren Punkte in x- und y-Richtung
   Nx=N;
   Ny=N;
   // Gesamtanzahl der Gitterpunkte
   npts=(Nx+2)*(Ny+2);
   // Aktive Punkte - Array
   active_pts();

   // Speicherbedarf pro Vektor in Byte
   nBytes=npts*sizeof(double);

   // Speicher für Vektoren allozieren + GPU ERWEITERUNGEN
   w=(double*)malloc(nBytes);
   v=(double*)malloc(nBytes);
   
   x=(double*)malloc(nBytes);
   s=(double*)malloc(nBytes);
   vo=(double*)malloc(nBytes);
   rnormalt=(double*)malloc(nBytes);
   svo=(double*)malloc(nBytes);
   r=(double*)malloc(nBytes);
   ro=(double*)malloc(nBytes);
   rnorm=(double*)malloc(nBytes);
   sv=(double*)malloc(nBytes);
   vro=(double*)malloc(nBytes);
   vr=(double*)malloc(nBytes);
   vs=(double*)malloc(nBytes);
   vso=(double*)malloc(nBytes);
   

   // auf Null setzen + GPU ERWEITERUNGEN
   memset(w, 0, nBytes);
   memset(v, 0, nBytes);
   
   memset(x, 0, nBytes);
   memset(s, 0, nBytes);
   memset(vo, 0, nBytes);
   memset(rnormalt, 0,nBytes);
   memset(svo, 0, nBytes);
   memset(r, 0, nBytes);
   memset(ro, 0, nBytes);
   memset(rnorm, 0, nBytes);
   memset(sv, 0, nBytes);
   memset(vro, 0, nBytes);
   memset(vr, 0, nBytes);
   memset(vso, 0, nBytes);
   memset(vs, 0, nBytes);

   // Aktive Punkte ausgeben
   if ((Nx<=16)&&(Ny<=16))
      print_active();

   random_vector(w);
   random_vector(v);
   double *d_v, *d_w, *d_x, *d_s, *d_vo, *d_rnorm_alt, *d_svo, *d_r, *d_ro, *d_rnorm, *d_sv, *d_vro, *d_vr, *d_vso, *d_vs;
   CHECK(cudaMalloc((void **)&d_v, nBytes));
   CHECK(cudaMalloc((void **)&d_w, nBytes));
   
   CHECK(cudaMalloc((void **)&d_x, nBytes)); // GPU ERWEITERUNGEN
   CHECK(cudaMalloc((void **)&d_s, nBytes));
   CHECK(cudaMalloc((void **)&d_vo, nBytes));
   CHECK(cudaMalloc((void **)&d_rnorm_alt, nBytes));
   CHECK(cudaMalloc((void **)&d_svo, nBytes));
   CHECK(cudaMalloc((void **)&d_r, nBytes));
   CHECK(cudaMalloc((void **)&d_ro, nBytes));
   CHECK(cudaMalloc((void **)&d_rnorm, nBytes));
   CHECK(cudaMalloc((void **)&d_sv, nBytes));
   CHECK(cudaMalloc((void **)&d_vro, nBytes));
   CHECK(cudaMalloc((void **)&d_vr, nBytes));
   CHECK(cudaMalloc((void **)&d_vso, nBytes));
   CHECK(cudaMalloc((void **)&d_vs, nBytes));
   
   // transfer data from host to device
   CHECK(cudaMemcpy(d_v, v, nBytes, cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(d_w, w, nBytes, cudaMemcpyHostToDevice));
   
   CHECK(cudaMemcpy(d_x, x, nBytes, cudaMemcpyHostToDevice)); // GPU ERWEITERUNGEN
   CHECK(cudaMemcpy(d_s, s, nBytes, cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(d_vo, vo, nBytes, cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(d_rnorm_alt, rnormalt, nBytes, cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(d_svo, svo, nBytes, cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(d_r, r, nBytes, cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(d_ro, ro, nBytes, cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(d_rnorm, rnorm, nBytes, cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(d_sv, sv, nBytes, cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(d_vro, vro, nBytes, cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(d_vr, vr, nBytes, cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(d_vso, vso, nBytes, cudaMemcpyHostToDevice));
   CHECK(cudaMemcpy(d_vs, vs, nBytes, cudaMemcpyHostToDevice));
   
   // invoke kernel at host side
   block.x=dimx;
   block.y=dimy;
   block.z=1;
   grid.x=(Nx + block.x - 1) / block.x;
   grid.y=(Ny + block.y - 1) / block.y;
   grid.z=1;

   // Test reduction
   /*int Nunroll=8;
   if (npts>256 && Nunroll>1)
   {
      double cpu_sum=0.0;
      iStart = seconds();
      for (int i = 0; i < npts; i++) cpu_sum += v[i];
      iElaps = seconds() - iStart;
      printf("cpu reduce      elapsed %f sec cpu_sum: %f\n", iElaps, cpu_sum);

      dim3 block2 (256,1);
      int nblk = (npts + (block2.x*Nunroll) - 1)/(block2.x*Nunroll);
      dim3 grid2 (nblk,1);
      CHECK(cudaMalloc((void **)&d_x, nblk*sizeof(double)));
      CHECK(cudaMemset(d_x,0,nblk*sizeof(double)));
      x=(double*)malloc(nblk*sizeof(double));
      CHECK(cudaDeviceSynchronize());
      iStart = seconds();
      reduceUnrolling<<<grid2, block2>>>(d_v, d_x, npts);
      CHECK(cudaDeviceSynchronize());
      iElaps = seconds() - iStart;
      CHECK(cudaMemcpy(x, d_x, nblk * sizeof(double),cudaMemcpyDeviceToHost));

      double gpu_sum = 0.0;
      for (int i = 0; i < grid2.x; i++) gpu_sum += x[i];

      printf("gpu Unrolling  elapsed %f sec gpu_sum: %f <<<grid %d block "
             "%d>>>\n", iElaps, gpu_sum, grid2.x, block2.x);

      assert(abs((gpu_sum-cpu_sum)/cpu_sum)<sqrt(npts)*DBL_EPSILON);
   }

   // Einheitsvektor
   memset(v, 0, nBytes);
   v[coord2index(Nx/2,Nx/2)]=1.0; // v=0, ausser am Gitterpunkt (Nx/2+1,Ny/2+1)
   print_vector("v",v,1);*/
   
   // cg auf gpu
   
    // Toleranz, Arraysize & Iterationsgrenze festlegen
   double tol = 1e-6;
   unsigned int kmax = 1e3;
   unsigned int k = 0;
   
   double size=Nx*Ny;
   
   
   // block dim grid dim
     dim3 block(dimx,dimy);
     dim3 grid(((Nx+1+block.x)/block.x), ((Ny+1+block.y)/block.y));
   
   // 0. Iteration
   
   laplace_2d_gpu<<<grid,block>>>(d_s,d_v,Nx,Ny);
   prod_gpu<<<grid,block>>>(d_vo,d_v,d_v,Nx,Ny);
   reduceUnrolling<<<grid,block>>>(d_vo,d_rnorm_alt,size);
   prod_gpu<<<grid,block>>>(d_svo,d_s,d_v,Nx,Ny);
   reduceUnrolling<<<grid,block>>>(d_svo,d_sv,size);
   
   double d_alpha= *d_rnorm_alt/ *d_sv;
   
   mul_add_gpu<<<grid,block>>>(d_x,d_alpha,d_v,Nx,Ny);
   vec_add_gpu<<<grid,block>>>(d_r,d_v,(-d_alpha),d_s,Nx,Ny);
   
   prod_gpu<<<grid,block>>>(d_ro,d_r,d_r,Nx,Ny);
   reduceUnrolling<<<grid,block>>>(d_ro,d_rnorm,size);
   
   // Iteration
   
   while (k<kmax && *rnorm>tol)
    {
      double d_beta = *d_rnorm/ *d_rnorm_alt;			// beta= rnorm/rnormalt
      update_p_gpu<<<grid,block>>>(d_r,d_beta,d_v,Nx,Ny);	// v = r+beta*v
      assign_v2v_gpu<<<grid,block>>>(d_rnorm_alt,d_rnorm,Nx,Ny);// rnormalt = rnorm
      laplace_2d_gpu<<<grid,block>>>(d_s,d_v,Nx,Ny);		// laplace (s,v)
      
      prod_gpu<<<grid,block>>>(d_vro,d_v,d_r,Nx,Ny);			// skalarprod v*r
      reduceUnrolling<<<grid,block>>>(d_vro,d_vr,size);		// skalarprod v*r
      prod_gpu<<<grid,block>>>(d_vso,d_v,d_s,Nx,Ny);			// skalarprod v*s
      reduceUnrolling<<<grid,block>>>(d_vso,d_vs,size);		// skalarprod v*s
      
      d_alpha = *d_vr/ *d_vs;					// alpha = (vr)/(vs)
      mul_add_gpu<<<grid,block>>>(d_x,d_alpha,d_v,Nx,Ny);	// x = x+alpha*v
      mul_add_gpu<<<grid,block>>>(d_r,(-d_alpha),d_s,Nx,Ny);	// r = r-alpha*s
      
      prod_gpu<<<grid,block>>>(d_ro,d_r,d_r,Nx,Ny);		// rnorm
      reduceUnrolling<<<grid,block>>>(d_ro,d_rnorm,size);	// rnorm
      
      k++;
    }
      
    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaMemcpy(x, d_x, nBytes, cudaMemcpyDeviceToHost));
   
   printf("Anzahl Iterationen: %d \n",k);
   print_vector("x_Ergebnis",x,1);
   
   CHECK(cudaFree(d_x));
   CHECK(cudaFree(d_s));
   CHECK(cudaFree(d_vo));
   CHECK(cudaFree(d_rnorm_alt));
   CHECK(cudaFree(d_svo));
   CHECK(cudaFree(d_r));
   CHECK(cudaFree(d_ro));
   CHECK(cudaFree(d_rnorm));

   
   free(active);
   free(w);
   free(v);
   free(x);
   free(s);
   free(vo);
   free(rnormalt);
   free(svo);
   free(r);
   free(ro);
   free(rnorm);
   

   return (0);
}
