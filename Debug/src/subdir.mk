################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/tracker.cu \
../src/tracking.cu 

OBJS += \
./src/tracker.o \
./src/tracking.o 

CU_DEPS += \
./src/tracker.d \
./src/tracking.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.0/bin/nvcc -G -g -O0 -gencode arch=compute_52,code=sm_52  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.0/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

