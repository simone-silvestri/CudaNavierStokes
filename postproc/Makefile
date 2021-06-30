# Use Prefix to define the home folder of the source code.
# It can be different from the folder in which you want to compile and run the mesh generator. 
# In the current directory ./ you only need to have the main.cu and this Makefile


SRC=./
OBJ=./

# Define compiler and optimizer's flags

LIBS = -lm 

INC = -I/usr/include
LIB = -L/usrlib -lc -lstdc++ 

CFLAGS = $(INC) $(LIB)

TARGET = post

MPICC = mpic++ -mcmodel=large 
ifeq ($(DBG),)
MPICC += -O5 
endif

# List of objects
OBJ_SRC = $(OBJ)post.o $(OBJ)comm.o $(OBJ)init.o 
OBJECTS =  $(OBJ_SRC) 

all: $(TARGET)

$(TARGET): $(OBJECTS) 
	$(MPICC) $(CFLAGS) -o $(TARGET) $(OBJECTS) $(LIBS) 

#MPIC++ compilation src files

$(OBJ)post.o: $(SRC)post.cpp
	$(MPICC) $(FLAG_GPU) -std=c++11 $(FLAG_ARCH) -c $(SRC)post.cpp $(CFLAGS) -o $(OBJ)post.o 
	
$(OBJ)comm.o: $(SRC)comm.cpp
	$(MPICC) $(FLAG_GPU) -std=c++11 $(FLAG_ARCH) -c $(SRC)comm.cpp $(CFLAGS) -o $(OBJ)comm.o 
	
$(OBJ)init.o: $(SRC)init.cpp
	$(MPICC) $(FLAG_GPU) -std=c++11 $(FLAG_ARCH) -c $(SRC)init.cpp $(CFLAGS) -o $(OBJ)init.o
	

clean:
		rm -rf $(TARGET) $(OBJ)*.o
