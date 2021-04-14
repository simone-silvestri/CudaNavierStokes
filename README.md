# CUDA-DNS

# cuda-solver for 3D COMPRESSIBLE NAVIER-STOKES EQUATIONS
 
<!-- The initial function to advance in time can be chosen in initProfile (in main.cu) --> 
<!--  -->
<!-- to benchmark the speed on CPU: -->
<!--  -->
<!-- - choose parameters in globals.h (nsteps and grid) -->
<!-- - compile with "make clean; make" (code will run on CPU) -->
<!-- - output file is in final.txt (1D slice of 3D solution) -->
<!-- - if the code is run with input "./ns JP KP" a graphic will show the evolution of the function in x direction -->
<!--   at position j=JP and k=KP (will make the code considerably slower) -->
<!--  -->
<!--  -->
<!-- to benchmark the speed on GPU: -->
<!--  -->
<!-- - choose parameters in globals.h (nsteps and grid) -->
<!-- - choose precision and stencil size in cuda_globals.h -->
<!-- - make sure that cuda location is specified in the Makefile -->
<!-- - make sure the major and minor (gpu-architecture in Makefile) match GPU architecture  -->
<!-- - compile with "make clean; make ARCH='GPU'" (code will run on GPU) -->
<!-- - output file is in final.txt (same 1D slice of 3D solution) -->

features:
- rk3 or rk4 for time stepping
- possible to choose between 2nd, 4th, 6th or 8th order spatial discretization 
- shared memory usage for fast derivation
- cuda dynamic parallelism to ensure correct memory access in x- y- and z- derivatives
- flux formulation to solve the conservative full split form of the advective terms


validation case

in the validation folder exchange the chit.h file to param.h in the src to validate with
J.R. DeBonis, Solutions of the Taylor-Green Vortex Problem Using High-Resolution Explicit Finite Difference Methods, 
AIAA 2013-0382

