# imx296_for_nvidia

## Linux kernel from nvidia
The source code of the Linux kernel from nvidia can be taken here:  
https://developer.nvidia.com/embedded/L4T/r32_Release_v4.3/Sources/T210/public_sources.tbz2  

## Build kernel with imx296 driver
cd Jetson_Nano_P3450/  
./build_kernel.sh

## Demo program for test imx296 sensor
./soft/v4l2_cuda_gist  

## Build 
cd ./imx296_for_nvidia/soft/v4l2_cuda_gist  
mkdir build  
cd build/  
cmake ..  
cmake --build .  

## Run
./v4l_cuda_gist -h  
Options:  
 -d | --device name 	Video device name (default: /dev/video0)  
 -g | --gain value 	Set gain (default: 50, min: 1, max 480)  
 -s | --shutter value 	Set shutter (default: 5000, min: 29, max 15110711)  
 -h | --help 		Print help  
