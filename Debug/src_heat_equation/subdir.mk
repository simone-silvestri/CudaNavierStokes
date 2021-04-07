################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src_heat_equation/cuda_derivs.cu \
../src_heat_equation/cuda_main_1kernel.cu \
../src_heat_equation/cuda_main_dynamic.cu \
../src_heat_equation/cuda_main_dynamic_nostreams.cu \
../src_heat_equation/cuda_main_dynamic_staticmem.cu \
../src_heat_equation/cuda_rhs.cu \
../src_heat_equation/cuda_utils.cu \
../src_heat_equation/main.cu 

OBJS += \
./src_heat_equation/cuda_derivs.o \
./src_heat_equation/cuda_main_1kernel.o \
./src_heat_equation/cuda_main_dynamic.o \
./src_heat_equation/cuda_main_dynamic_nostreams.o \
./src_heat_equation/cuda_main_dynamic_staticmem.o \
./src_heat_equation/cuda_rhs.o \
./src_heat_equation/cuda_utils.o \
./src_heat_equation/main.o 

CU_DEPS += \
./src_heat_equation/cuda_derivs.d \
./src_heat_equation/cuda_main_1kernel.d \
./src_heat_equation/cuda_main_dynamic.d \
./src_heat_equation/cuda_main_dynamic_nostreams.d \
./src_heat_equation/cuda_main_dynamic_staticmem.d \
./src_heat_equation/cuda_rhs.d \
./src_heat_equation/cuda_utils.d \
./src_heat_equation/main.d 


# Each subdirectory must supply rules for building sources it contributes
src_heat_equation/%.o: ../src_heat_equation/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-11.2/bin/nvcc -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "src_heat_equation" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-11.2/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


