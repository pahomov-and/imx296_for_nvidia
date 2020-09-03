# imx296_for_nvidia
Linux camera driver imx296 for nvidia and software for processing video from the camera
git clone https://github.com/pahomov-and/imx296_for_nvidia.git

## Demo program for test imx296 sensor
./soft/v4l2_cuda_gist

Build
> cd ./imx296_for_nvidia/soft/v4l2_cuda_gist
> mkdir build
> cd build/
> cmake ..
> cmake --build .

Run
> ./v4l_cuda_gist -h
> Options:
> -d | --device name 	Video device name (default: /dev/video0)
> -g | --gain value 	Set gain (default: 50, min: 1, max 480)
> -s | --shutter value 	Set shutter (default: 5000, min: 29, max 15110711)
> -h | --help 		Print help
