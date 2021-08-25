# Use Prefix to define the home folder of the source code.
# It can be different from the folder in which you want to compile and run the mesh generator. 
# In the current directory ./ you only need to have the main.cu and this Makefile

GEN = /usr

#MPI = /usr/local/openmpi-4.1.1

CUDA = /usr/local/cuda-11.2

GPU_ARCHITECTURE = 70

ifeq ($(ARCH),CPU)
gpu_usage=0
endif
ifeq ($(ARCH),)
ARCH=GPU
gpu_usage=1
endif
ifeq ($(ARCH),GPU)
gpu_usage=1
endif

SRC=./src/
OBJ=./obj/

#mx_tot=-Dmx_tot=128 
#my_tot=-Dmy_tot=128 
#mz_tot=-Dmz_tot=128 
#Lx= -DLx=6.2832
#Ly= -DLy=6.2832
#Lz= -DLz=6.2832
#Re= -DRe=1600.0
#Pr= -DPr=0.7
#Ma= -DMa=1.5
#visc= -Dviscexp=0.7
#cfl= -DCFL=0.7

#nonUniformX   = -DnonUniformX=false
#periodicX     = -DperiodicX=true
#forcing       = -Dforcing=false
#useStreams    = -DuseStreams=false
#boundaryLayer = -DboundaryLayer=false

#passing default parameters to code 
GRID   = $(mx_tot) $(my_tot) $(mz_tot) $(Lx) $(Ly) $(Lz) $(nonUniformX) $(periodicX) 
WORKERS= $(pRow) $(pCol)
STENCIL= $(stencilSize) $(stencilVisc)
PARAM  = $(Re) $(Pr) $(Ma) $(gam) $(visc) $(cfl) $(forcing) $(boundaryLayer) $(perturbed)
RESTART= $(restart)
SIM = $(nsteps) $(nfiles) $(useStreams) $(checkCFLcondition) $(checkBulk)


CPPFLAGS= $(PARAM) $(WORKERS) $(STENCIL) $(RESTART) $(GRID) $(SIM)


# Define compiler and optimizer's flags

FLAG_GPU  = -Darch=$(gpu_usage) 
FLAG_ARCH = -Dcapability=$(GPU_ARCHITECTURE)

LIBS = -lm 

FLAG2 = --use_fast_math
MAT = -ftz=true -prec-div=false
FLAG1 = -arch 'compute_$(GPU_ARCHITECTURE)' -code 'sm_$(GPU_ARCHITECTURE)'
INC = -I$(CUDA)/include -I$(GEN)/include
#INC += -I$(MPI)/include
LIB = -L$(CUDA)/lib64 -L$(GEN)/lib -lc -lstdc++ -lcuda -lcudart -lcudadevrt 
#LIB += -L$(MPI)/lib
NVCC = nvcc $(DBG) $(CPPFLAGS) -lineinfo -rdc=true # 
MAXREG = # --maxrregcount=82 --ptxas-options=-v
ifeq ($(DBG),)
NVCC += -O5  
endif

CC = $(NVCC)
CFLAGS = $(INC) $(LIB)

TARGET = ns

MPICC = mpic++ -mcmodel=large $(CPPFLAGS) 
ifeq ($(DBG),)
MPICC += -O5 
endif

# List of objects
OBJ_SRC = $(OBJ)main.o $(OBJ)comm.o $(OBJ)init.o 
OBJ_CUDA= $(OBJ)cuda_utils.o $(OBJ)cuda_math.o $(OBJ)cuda_main.o $(OBJ)cuda_derivs.o $(OBJ)cuda_rhs.o $(OBJ)calc_stress.o $(OBJ)sponge.o
OBJ_LINK= $(OBJ)cuda_link.o 
OBJECTS =  $(OBJ_SRC) $(OBJ_CUDA) $(OBJ_LINK)

all: $(TARGET)

$(TARGET): $(OBJECTS) 
	$(MPICC) $(CFLAGS) $(FLAG_GPU) $(FLAG_ARCH) -o $(TARGET) $(OBJECTS) $(LIBS) -lcudart  -lcudadevrt 

#MPIC++ compilation src files

$(OBJ)main.o: $(SRC)main.cpp
	$(MPICC) $(FLAG_GPU) -std=c++11 $(FLAG_ARCH) -c $(SRC)main.cpp $(CFLAGS) -o $(OBJ)main.o 
	
$(OBJ)comm.o: $(SRC)comm.cpp
	$(MPICC) $(FLAG_GPU) -std=c++11 $(FLAG_ARCH) -c $(SRC)comm.cpp $(CFLAGS) -o $(OBJ)comm.o 
	
$(OBJ)init.o: $(SRC)init.cpp
	$(MPICC) $(FLAG_GPU) -std=c++11 $(FLAG_ARCH) -c $(SRC)init.cpp $(CFLAGS) -o $(OBJ)init.o

#NVCC compilation src files (with link)

#compiling step (OBJ_CUDA)	
$(OBJ)cuda_main.o: $(SRC)cuda_main.cu
	$(NVCC) -c $(FLAG1) $(FLAG_ARCH) $(CFLAGS) $(SRC)cuda_main.cu $(FLAG2) -o $(OBJ)cuda_main.o

$(OBJ)calc_stress.o: $(SRC)calc_stress.cu
	$(NVCC) -c $(FLAG1) $(FLAG_ARCH) $(CFLAGS) $(SRC)calc_stress.cu $(FLAG2) -o $(OBJ)calc_stress.o

$(OBJ)cuda_rhs.o: $(SRC)cuda_rhs.cu
	$(NVCC) -c $(MAXREG) $(FLAG1) $(FLAG_ARCH) $(CFLAGS) $(SRC)cuda_rhs.cu $(FLAG2) -o $(OBJ)cuda_rhs.o

$(OBJ)cuda_math.o: $(SRC)cuda_math.cu
	$(NVCC) -c $(FLAG1) $(FLAG_ARCH) $(CFLAGS) $(SRC)cuda_math.cu $(FLAG2) -o $(OBJ)cuda_math.o
	
$(OBJ)cuda_utils.o: $(SRC)cuda_utils.cu
	$(NVCC) -c $(FLAG1) $(FLAG_ARCH) $(CFLAGS) $(SRC)cuda_utils.cu $(FLAG2) -o $(OBJ)cuda_utils.o
	
$(OBJ)cuda_derivs.o: $(SRC)cuda_derivs.cu
	$(NVCC) -c $(FLAG1) $(FLAG_ARCH) $(CFLAGS) $(SRC)cuda_derivs.cu $(FLAG2) -o $(OBJ)cuda_derivs.o

$(OBJ)sponge.o: $(SRC)sponge.cu
	$(NVCC) -c $(FLAG1) $(FLAG_ARCH) $(CFLAGS) $(SRC)sponge.cu $(FLAG2) -o $(OBJ)sponge.o

#linking step (OBJ_LINK)	
$(OBJ)cuda_link.o: $(OBJ_CUDA)
	$(NVCC) -dlink $(FLAG1) $(FLAG_ARCH) $(CFLAGS) $(OBJ_CUDA) $(FLAG2) -o $(OBJ)cuda_link.o
	

clean:
		rm -rf $(TARGET) $(OBJ)*.o
