# Use Prefix to define the home folder of the source code.
# It can be different from the folder in which you want to compile and run the mesh generator. 
# In the current directory ./ you only need to have the main.cu and this Makefile

INC_GL = /usr/include/
LIB_GL = /usr/lib/

CUDA = /usr/local/cuda-10.1

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

# Define compiler and optimizer's flags

FLAG_GPU  = -Darch=$(gpu_usage) 
FLAG_ARCH = -Dcapability=$(GPU_ARCHITECTURE)

LIBS = -lm 


ifeq ($(ARCH),GPU)
FLAG2 = --use_fast_math
MAT = -ftz=true -prec-div=false
FLAG1 = -arch 'compute_$(GPU_ARCHITECTURE)' -code 'sm_$(GPU_ARCHITECTURE)'
INC = -I$(CUDA)/include
LIB = -L$(CUDA)/lib64 -lc -lstdc++ -lcuda ## -lcudart 
NVCC = nvcc $(DBG) -O3 -lineinfo -rdc=true #--ptxas-options=-v   
endif

CC = $(NVCC)

CFLAGS = -I$(INC_GL) -L$(LIB_GL) -I$(INC) -L$(LIB) 

TARGET = ns



ifeq ($(ARCH),GPU)
OBJ_CUDA = $(OBJ)cuda_main_3streams.o $(OBJ)cuda_utils.o $(OBJ)cuda_derivs.o $(OBJ)cuda_rhs.o $(OBJ)calc_stress.o $(OBJ)cuda_math.o
endif

# List of objects
OBJ_SRC = $(OBJ)main.o

OBJECTS = $(OBJ_CUDA) $(OBJ_SRC)

all: $(TARGET)

$(TARGET): $(OBJECTS) 
	$(CC) $(CFLAGS) $(FLAG_GPU) $(FLAG_ARCH) -o $(TARGET) $(OBJECTS) -lm $(LIBS) $(FLAG1)

$(OBJ)main.o: $(SRC)main.cu
	$(CC) $(FLAG_GPU) $(FLAG_ARCH) -c $(SRC)main.cu $(CFLAGS) -o $(OBJ)main.o

ifeq ($(ARCH),GPU)
$(OBJ)cuda_main_3streams.o: $(SRC)cuda_main_3streams.cu
	$(NVCC) -c $(FLAG1) $(FLAG_ARCH) $(CFLAGS) $(SRC)cuda_main_3streams.cu $(FLAG2) -o $(OBJ)cuda_main_3streams.o

$(OBJ)calc_stress.o: $(SRC)calc_stress.cu
	$(NVCC) -c $(FLAG1) $(FLAG_ARCH) $(CFLAGS) $(SRC)calc_stress.cu $(FLAG2) -o $(OBJ)calc_stress.o

$(OBJ)cuda_rhs.o: $(SRC)cuda_rhs.cu
	$(NVCC) -c $(FLAG1) $(FLAG_ARCH) $(CFLAGS) $(SRC)cuda_rhs.cu $(FLAG2) -o $(OBJ)cuda_rhs.o

$(OBJ)cuda_math.o: $(SRC)cuda_math.cu
	$(NVCC) -c $(FLAG1) $(FLAG_ARCH) $(CFLAGS) $(SRC)cuda_math.cu $(FLAG2) -o $(OBJ)cuda_math.o

$(OBJ)cuda_utils.o: $(SRC)cuda_utils.cu
	$(NVCC) -c $(FLAG1) $(FLAG_ARCH) $(CFLAGS) $(SRC)cuda_utils.cu $(FLAG2) -o $(OBJ)cuda_utils.o

$(OBJ)cuda_derivs.o: $(SRC)cuda_derivs.cu
	$(NVCC) -c $(FLAG1) $(FLAG_ARCH) $(CFLAGS) $(SRC)cuda_derivs.cu $(FLAG2) -o $(OBJ)cuda_derivs.o
endif

clean:
		rm -rf $(TARGET) $(OBJ)*.o
