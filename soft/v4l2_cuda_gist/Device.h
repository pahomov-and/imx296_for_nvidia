//
// Created by Andrey Pahomov on 28.08.20.
//

#ifndef MAPPING_BUFFERS_SENSOR_H
#define MAPPING_BUFFERS_SENSOR_H

#include <thread>
#include <assert.h>

#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <malloc.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include <time.h>
#include <sys/time.h>
#include <asm/types.h>

#include <linux/videodev2.h>
#include "tegra-v4l2-camera.h"

typedef enum {
    IO_METHOD_READ,
    IO_METHOD_MMAP,
    IO_METHOD_USERPTR,
} io_method;


class Device {
public:
    void Init(std::string dev = "/dev/video0");

    void Start();

    void Stop();

    void Join();

    void SetCallProcessImage(void(*call)(void *));

    void SetCallDrawImage(void(*call)(void *));

    void SetShutter(int shutter = 5000);

    void SetGain(int gain = 50);

private:

    bool cuda_zero_copy = false;
    unsigned int pixel_format = V4L2_PIX_FMT_Y10;//V4L2_PIX_FMT_UYVY;
    unsigned int field = V4L2_FIELD_NONE;

    std::string dev_name;
    io_method io;
    int fd;

    unsigned int width;
    unsigned int height;

    bool isRun;
    std::thread _runThread;

    void (*process_image)(void *) = 0;

    void (*draw_image)(void *) = 0;

    void errno_exit(const char *s);

    int xioctl(int fd, int request, void *arg);

    int read_frame(void);

    void timemeasurement_start(struct timeval *timer);

    void timemeasurement_stop(struct timeval *timer, long int *s, long int *us);

    void mainloop(void);

    void stop_capturing(void);

    void start_capturing(void);

    void uninit_device(void);

    void init_read(unsigned int buffer_size);

    void init_mmap(void);

    void init_userp(unsigned int buffer_size);

    int set_ctrl(int fd, unsigned int id, int val);

    int get_ctrl(int fd, int id, int *val);

    int sensor_set_parameters(int newGain, int newShutter);

    void init_device(void);

    void close_device(void);

    void open_device(void);

    void init_cuda(void);


};


#endif //MAPPING_BUFFERS_SENSOR_H
