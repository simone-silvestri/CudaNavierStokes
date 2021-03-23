# CUDA-DNS

# cuda-solver for 1D advection equation

to benchmark the speed on CPU:

- choose parameters in globals.h (nsteps and grid)
- compile with "make clean; make" (code will run on CPU)
- output file is in final.txt (1D slice of 3D solution)
- if the code is run with input "./ns JP KP" a graphic will show the evolution of the function in x direction
  at position j=JP and k=KP (will make the code considerably slower)


to benchmark the speed on GPU:

- choose parameters in globals.h (nsteps and grid)
- choose precision and stencil size in cuda_globals.h
- make sure that cuda location is specified in the Makefile
- make sure the major and minor (gpu-architecture in Makefile) match GPU architecture 
- compile with "make clean; make ARCH='GPU'" (code will run on GPU)
- output file is in final.txt (same 1D slice of 3D solution)

