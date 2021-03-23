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
ARCH=CPU
gpu_usage=0
endif
ifeq ($(ARCH),GPU)
gpu_usage=1
endif

SRC=./src/
OBJ=./obj/

# Define compiler and optimizer's flags

FLAG_GPU = -Darch=$(gpu_usage) 
FLAG_PREC = -Dprec=float

LIBS = -lGLU -lGL -lglut -lm 

CC = nvcc -O3 $(DBG)  #-Xpreprocessor -fopenmp -O3 -Wno-c++11-extensions $(DBG)  

ifeq ($(ARCH),GPU)
FLAG2 = --use_fast_math
MAT = -ftz=true -prec-div=false
FLAG1 = -arch 'compute_$(GPU_ARCHITECTURE)' -code 'sm_$(GPU_ARCHITECTURE)'
INC = -I$(CUDA)/include
LIB = -L$(CUDA)/lib64 -lc -lstdc++ -lcuda #-lcudart 
NVCC = nvcc $(DBG)
endif

CFLAGS = -I$(INC_GL) -L$(LIB_GL) -I$(INC) -L$(LIB) 

TARGET = ns 



ifeq ($(ARCH),GPU)
OBJ_CUDA = $(OBJ)cuda_solver.o
endif

# List of objects
OBJ_SRC = $(OBJ)main.o

OBJECTS = $(OBJ_CUDA) $(OBJ_SRC)

all: $(TARGET)

$(TARGET): $(OBJECTS) 
	$(CC) $(CFLAGS) $(FLAG_GPU) -o $(TARGET) $(OBJECTS) -lm $(LIBS) 

$(OBJ)main.o: $(SRC)main.cu  
	$(CC) $(FLAG_GPU) -c $(SRC)main.cu $(CFLAGS) -o $(OBJ)main.o

ifeq ($(ARCH),GPU)
$(OBJ)cuda_solver.o: $(SRC)cuda_solver.cu
	$(NVCC) -c $(FLAG1) $(CFLAGS) $(SRC)cuda_solver.cu $(FLAG2) -o $(OBJ)cuda_solver.o
endif

clean:
		rm -rf $(TARGET) $(OBJ)*.o
