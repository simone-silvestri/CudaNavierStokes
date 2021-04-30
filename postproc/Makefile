# Use Prefix to define the home folder of the source code.
# It can be different from the folder in which you want to compile and run the mesh generator. 
# In the current directory ./ you only need to have the main.cu and this Makefile

gpu_usage=1

SRC=./
OBJ=./

# Define compiler and optimizer's flags

FLAG_GPU  = -Darch=$(gpu_usage) 
FLAG_ARCH = -Dcapability=100

LIBS = -lm 


FLAG2 = --use_fast_math
MAT = -ftz=true -prec-div=false
FLAG1 = 
INC = -I$(CUDA)/include
LIB = -L$(CUDA)/lib64 -lc -lstdc++ -lcuda ## -lcudart 
NVCC = nvcc $(DBG) -lineinfo -rdc=true #--ptxas-options=-v  
ifeq ($(DBG),)
NVCC += -O5  
endif 

CC = $(NVCC)

CFLAGS = $(INC) $(LIB) 

TARGET = post


OBJ_POST = $(OBJ)post.o $(OBJ)init.o

OBJECTS = $(OBJ_POST)

all: $(TARGET)

$(TARGET): $(OBJECTS) 
	$(CC) $(CFLAGS) $(FLAG_GPU) $(FLAG_ARCH) -o $(TARGET) $(OBJECTS) -lm $(LIBS) $(FLAG1)

$(OBJ)post.o: $(SRC)post.cu
	$(CC) $(FLAG_GPU) -std=c++11 $(FLAG_ARCH) -c $(SRC)post.cu $(CFLAGS) -o $(OBJ)post.o
	
$(OBJ)init.o: $(SRC)init.cu
	$(CC) $(FLAG_GPU) -std=c++11 $(FLAG_ARCH) -c $(SRC)init.cu $(CFLAGS) -o $(OBJ)init.o
	

clean:
		rm -rf $(TARGET) $(OBJ)*.o
