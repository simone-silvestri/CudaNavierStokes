################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../postproc/comm.cpp \
../postproc/init.cpp \
../postproc/post.cpp 

OBJS += \
./postproc/comm.o \
./postproc/init.o \
./postproc/post.o 

CPP_DEPS += \
./postproc/comm.d \
./postproc/init.d \
./postproc/post.d 


# Each subdirectory must supply rules for building sources it contributes
postproc/%.o: ../postproc/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-11.2/bin/nvcc -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "postproc" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-11.2/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


