################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/cuda_derivs.cu \
../src/cuda_main_dynamic.cu \
../src/cuda_rhs.cu \
../src/cuda_utils.cu \
../src/main.cu 

OBJS += \
./src/cuda_derivs.o \
./src/cuda_main_dynamic.o \
./src/cuda_rhs.o \
./src/cuda_utils.o \
./src/main.o 

CU_DEPS += \
./src/cuda_derivs.d \
./src/cuda_main_dynamic.d \
./src/cuda_rhs.d \
./src/cuda_utils.d \
./src/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-11.2/bin/nvcc -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-11.2/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


