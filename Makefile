# Use Prefix to define the home folder of the source code.
# It can be different from the folder in which you want to compile and run the mesh generator. 
# In the current directory ./ you only need to have the main.cpp and this Makefile

INC_GL = /usr/include/
LIB_GL = /usr/lib/


# Define compiler and optimizer's flags
CC = g++ -Xpreprocessor -fopenmp -O3 -Wno-c++11-extensions $(DBG) 

CFLAGS = -I$(INC_GL) -L$(LIB_GL) -I$(INC_OMP) -L$(LIB_OMP)

LIBS = -lGLU -lGL -lglut $(OMPLIB) -lm 

TARGET = ns 

# List of objects
OBJ_SRC = main.o

OBJ = $(OBJ_SRC)

all: $(TARGET)

$(TARGET): $(OBJ) 
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) -lm $(LIBS) 

main.o: main.cpp  globals.h
	$(CC) -c main.cpp 

clean:
		rm -rf $(TARGET) $(OBJ)
