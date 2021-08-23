################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/calc_stress.cu \
../src/cuda_derivs.cu \
../src/cuda_main.cu \
../src/cuda_math.cu \
../src/cuda_rhs.cu \
../src/cuda_utils.cu 

CPP_SRCS += \
../src/comm.cpp \
../src/init.cpp \
../src/main.cpp 

OBJS += \
./src/calc_stress.o \
./src/comm.o \
./src/cuda_derivs.o \
./src/cuda_main.o \
./src/cuda_math.o \
./src/cuda_rhs.o \
./src/cuda_utils.o \
./src/init.o \
./src/main.o 

CU_DEPS += \
./src/calc_stress.d \
./src/cuda_derivs.d \
./src/cuda_main.d \
./src/cuda_math.d \
./src/cuda_rhs.d \
./src/cuda_utils.d 

CPP_DEPS += \
./src/comm.d \
./src/init.d \
./src/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-11.2/bin/nvcc -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-11.2/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-11.2/bin/nvcc -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-11.2/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


