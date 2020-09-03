//
// Created by Andrey Pahomov on 28.08.20.
//

#include <iostream>
#include <climits>

#include "Device.h"
#include "LOG.h"
#include "MemAlloc.h"


#define CLEAR(x) memset (&(x), 0, sizeof (x))
#define ARRAY_SIZE(a)   (sizeof(a)/sizeof((a)[0]))

struct timeval timer;
long int seconds, useconds;
int timerCycles = 100;
int run = 0;


void Device::Init(std::string dev) {
    dev_name = dev;//"/dev/video0";
    io = IO_METHOD_MMAP;
    fd = -1;
    MEM_ALLOC->buffers = NULL;
    MEM_ALLOC->n_buffers = 0;
    MEM_ALLOC->width = width = UINT_MAX;
    MEM_ALLOC->height = height = UINT_MAX;
    MEM_ALLOC->cuda_out_buffer = NULL;
    cuda_zero_copy = true;

    pixel_format = V4L2_PIX_FMT_Y10;
    field = V4L2_FIELD_NONE;

    MEM_ALLOC->intensity_num = NULL;
    MEM_ALLOC->intensity_pro = NULL;
    MEM_ALLOC->min_index = NULL;
    MEM_ALLOC->max_index = NULL;

    open_device();
    init_device();
    init_cuda();

}


void Device::Start() {
    start_capturing();

    _runThread = std::thread([&]() {
        isRun = true;
        mainloop();

        stop_capturing();
        uninit_device();
        close_device();
        exit(EXIT_SUCCESS);
    });

}

void Device::Stop() {
    isRun = false;
}

void Device::Join() {
    _runThread.join();
}

void Device::SetCallProcessImage(void(*call)(void *)) {
    process_image = call;
}

void Device::SetCallDrawImage(void(*call)(void *)) {
    draw_image = call;
}


void Device::SetShutter(int shutter) {
    unsigned int val;
    int rc;
    if (shutter > 0) {
        rc = set_ctrl(fd, TEGRA_CAMERA_CID_EXPOSURE, 0);
        rc = get_ctrl(fd, TEGRA_CAMERA_CID_EXPOSURE, (int *) &val);
        rc = set_ctrl(fd, TEGRA_CAMERA_CID_EXPOSURE, shutter);
        rc = get_ctrl(fd, TEGRA_CAMERA_CID_EXPOSURE, (int *) &val);
        LOG_INFO("TEGRA_CAMERA_CID_EXPOSURE val= ", rc);
    } else
        LOG_ERROR("shutter <= 0");
}

void Device::SetGain(int gain) {
    unsigned int val;
    int rc;

    if (gain > 0) {
        rc = set_ctrl(fd, TEGRA_CAMERA_CID_GAIN, 0);
        rc = get_ctrl(fd, TEGRA_CAMERA_CID_GAIN, (int *) &val);
        rc = set_ctrl(fd, TEGRA_CAMERA_CID_GAIN, gain);
        rc = get_ctrl(fd, TEGRA_CAMERA_CID_GAIN, (int *) &val);
        LOG_INFO("TEGRA_CAMERA_CID_GAIN val= ", val);
    } else
        LOG_ERROR("gain <= 0");

}

void Device::errno_exit(const char *s) {

    LOG_ERROR(std::strerror(errno));

    exit(EXIT_FAILURE);
}

int Device::xioctl(int fd, int request, void *arg) {
    int r;

    do r = ioctl(fd, request, arg);
    while (-1 == r && EINTR == errno);

    return r;
}

int Device::read_frame(void) {
    struct v4l2_buffer buf;
    unsigned int i;

    switch (io) {
        case IO_METHOD_READ:
            if (-1 == read(fd, MEM_ALLOC->buffers[0].start, MEM_ALLOC->buffers[0].length)) {
                switch (errno) {
                    case EAGAIN:
                        return 0;

                    case EIO:
                        /* Could ignore EIO, see spec. */

                        /* fall through */

                    default:
                        errno_exit("read");
                }
            }

            process_image(MEM_ALLOC->buffers[0].start);

            break;

        case IO_METHOD_MMAP:
            CLEAR(buf);

            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;

            if (-1 == xioctl(fd, VIDIOC_DQBUF, &buf)) {
                switch (errno) {
                    case EAGAIN:
                        return 0;

                    case EIO:
                        /* Could ignore EIO, see spec. */

                        /* fall through */

                    default:
                        errno_exit("VIDIOC_DQBUF");
                }
            }

            assert(buf.index < MEM_ALLOC->n_buffers);

            process_image(MEM_ALLOC->buffers[buf.index].start);

            if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
                errno_exit("VIDIOC_QBUF");

            break;

        case IO_METHOD_USERPTR:
            CLEAR(buf);

            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_USERPTR;

            if (-1 == xioctl(fd, VIDIOC_DQBUF, &buf)) {
                switch (errno) {
                    case EAGAIN:
                        return 0;

                    case EIO:
                        /* Could ignore EIO, see spec. */

                        /* fall through */

                    default:
                        errno_exit("VIDIOC_DQBUF");
                }
            }

            for (i = 0; i < MEM_ALLOC->n_buffers; ++i)
                if (buf.m.userptr == (unsigned long) MEM_ALLOC->buffers[i].start
                    && buf.length == MEM_ALLOC->buffers[i].length)
                    break;

            assert(i < MEM_ALLOC->n_buffers);

            process_image((void *) buf.m.userptr);

            if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
                errno_exit("VIDIOC_QBUF");

            break;
    }


    draw_image(MEM_ALLOC->cuda_out_buffer);

    return 1;
}


void Device::timemeasurement_start(struct timeval *timer) {
    gettimeofday(timer, (struct timezone *) 0);
}

void Device::timemeasurement_stop(struct timeval *timer, long int *s, long int *us) {
    struct timeval end;

    gettimeofday(&end, (struct timezone *) 0);
    *s = end.tv_sec - timer->tv_sec;
    *us = end.tv_usec - timer->tv_usec;
    if (*us < 0) {
        *us += 1000000;
        *s = *s - 1;
    }
}

void Device::mainloop(void) {
    timemeasurement_start(&timer);

    while (isRun) {
        fd_set fds;
        struct timeval tv;
        int r;

        FD_ZERO (&fds);
        FD_SET (fd, &fds);

        /* Timeout. */
        tv.tv_sec = 0;//2;
        tv.tv_usec = 50000; // 50000;

        r = select(fd + 1, &fds, NULL, NULL, &tv);

        if (-1 == r) {
            if (EINTR == errno)
                continue;
        }

        if (0 == r) {
            LOG_ERROR("select timeout");
            continue;
        }

        if (read_frame()) {
            // Print Out Duration.
            if (((timerCycles) - 1) == (run % (timerCycles))) {
                timemeasurement_stop(&timer, &seconds, &useconds);
                LOG_INFO("Duration:\t", seconds, "s\t", useconds, " for ", (run % timerCycles) + 1,
                         " Cycles ==  ", ((float) 1000000 * ((run % timerCycles) + 1) / (seconds * 1000000 + useconds))
                );
                timemeasurement_start(&timer);
            }
            run++;
        }
    }
}

void Device::stop_capturing(void) {
    enum v4l2_buf_type type;

    switch (io) {
        case IO_METHOD_READ:
            /* Nothing to do. */
            break;

        case IO_METHOD_MMAP:
        case IO_METHOD_USERPTR:
            type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

            if (-1 == xioctl(fd, VIDIOC_STREAMOFF, &type))
                errno_exit("VIDIOC_STREAMOFF");

            break;
    }
}

void Device::start_capturing(void) {
    unsigned int i;
    enum v4l2_buf_type type;

    switch (io) {
        case IO_METHOD_READ:
            /* Nothing to do. */
            break;

        case IO_METHOD_MMAP:
            for (i = 0; i < MEM_ALLOC->n_buffers; ++i) {
                struct v4l2_buffer buf;

                CLEAR(buf);

                buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buf.memory = V4L2_MEMORY_MMAP;
                buf.index = i;

                if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
                    errno_exit("VIDIOC_QBUF");
            }

            type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

            if (-1 == xioctl(fd, VIDIOC_STREAMON, &type))
                errno_exit("VIDIOC_STREAMON");

            break;

        case IO_METHOD_USERPTR:
            for (i = 0; i < MEM_ALLOC->n_buffers; ++i) {
                struct v4l2_buffer buf;

                CLEAR(buf);

                buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buf.memory = V4L2_MEMORY_USERPTR;
                buf.index = i;
                buf.m.userptr = (unsigned long) MEM_ALLOC->buffers[i].start;
                buf.length = MEM_ALLOC->buffers[i].length;

                if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
                    errno_exit("VIDIOC_QBUF");
            }

            type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

            if (-1 == xioctl(fd, VIDIOC_STREAMON, &type))
                errno_exit("VIDIOC_STREAMON");

            break;
    }
}

void Device::uninit_device(void) {
    unsigned int i;

    switch (io) {
        case IO_METHOD_READ:
            free(MEM_ALLOC->buffers[0].start);
            break;

        case IO_METHOD_MMAP:
            for (i = 0; i < MEM_ALLOC->n_buffers; ++i)
                if (-1 == munmap(MEM_ALLOC->buffers[i].start, MEM_ALLOC->buffers[i].length))
                    errno_exit("munmap");
            break;

        case IO_METHOD_USERPTR:
            for (i = 0; i < MEM_ALLOC->n_buffers; ++i) {
                if (cuda_zero_copy) {
                    cudaFree(MEM_ALLOC->buffers[i].start);
                } else {
                    free(MEM_ALLOC->buffers[i].start);
                }
            }
            break;
    }

    free(MEM_ALLOC->buffers);

    if (cuda_zero_copy) {
        LOG_INFO("cudaFree");
        cudaFree(MEM_ALLOC->cuda_out_buffer);
        cudaFree(MEM_ALLOC->intensity_num);
        cudaFree(MEM_ALLOC->intensity_pro);
        cudaFree(MEM_ALLOC->min_index);
        cudaFree(MEM_ALLOC->max_index);
    }
}

void Device::init_read(unsigned int buffer_size) {
    MEM_ALLOC->buffers = (struct buffer *) calloc(1, sizeof(*MEM_ALLOC->buffers));

    if (!MEM_ALLOC->buffers) {
        LOG_ERROR("Out of memory");
        exit(EXIT_FAILURE);
    }

    MEM_ALLOC->buffers[0].length = buffer_size;
    MEM_ALLOC->buffers[0].start = malloc(buffer_size);

    if (!MEM_ALLOC->buffers[0].start) {
        LOG_ERROR("Out of memory");
        exit(EXIT_FAILURE);
    }
}

void Device::init_mmap(void) {
    struct v4l2_requestbuffers req;

    CLEAR(req);

    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req)) {
        if (EINVAL == errno) {
            LOG_ERROR(dev_name, " does not support memory mapping");
            exit(EXIT_FAILURE);
        } else {
            errno_exit("VIDIOC_REQBUFS");
        }
    }

    if (req.count < 2) {
        LOG_ERROR("Insufficient buffer memory on ", dev_name);
        exit(EXIT_FAILURE);
    }

    MEM_ALLOC->buffers = (struct buffer *) calloc(req.count, sizeof(*MEM_ALLOC->buffers));

    if (!MEM_ALLOC->buffers) {
        LOG_ERROR("Out of memory");
        exit(EXIT_FAILURE);
    }

    for (MEM_ALLOC->n_buffers = 0; MEM_ALLOC->n_buffers < req.count; ++MEM_ALLOC->n_buffers) {
        struct v4l2_buffer buf;

        CLEAR(buf);

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = MEM_ALLOC->n_buffers;

        if (-1 == xioctl(fd, VIDIOC_QUERYBUF, &buf))
            errno_exit("VIDIOC_QUERYBUF");

        MEM_ALLOC->buffers[MEM_ALLOC->n_buffers].length = buf.length;
        MEM_ALLOC->buffers[MEM_ALLOC->n_buffers].start =
                mmap(NULL /* start anywhere */,
                     buf.length,
                     PROT_READ | PROT_WRITE /* required */,
                     MAP_SHARED /* recommended */,
                     fd, buf.m.offset);

        if (MAP_FAILED == MEM_ALLOC->buffers[MEM_ALLOC->n_buffers].start)
            errno_exit("mmap");
    }
}

void Device::init_userp(unsigned int buffer_size) {
    struct v4l2_requestbuffers req;
    unsigned int page_size;

    page_size = getpagesize();
    buffer_size = (buffer_size + page_size - 1) & ~(page_size - 1);

    CLEAR(req);

    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_USERPTR;

    if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req)) {
        if (EINVAL == errno) {
            LOG_ERROR(dev_name, " does not support user pointer i/o");
            exit(EXIT_FAILURE);
        } else {
            errno_exit("VIDIOC_REQBUFS");
        }
    }

    MEM_ALLOC->buffers = (struct buffer *) calloc(4, sizeof(*MEM_ALLOC->buffers));

    if (!MEM_ALLOC->buffers) {
        LOG_ERROR("Out of memory");
        exit(EXIT_FAILURE);
    }

    for (MEM_ALLOC->n_buffers = 0; MEM_ALLOC->n_buffers < 4; ++MEM_ALLOC->n_buffers) {
        MEM_ALLOC->buffers[MEM_ALLOC->n_buffers].length = buffer_size;
        if (cuda_zero_copy) {
            cudaMallocManaged(&MEM_ALLOC->buffers[MEM_ALLOC->n_buffers].start, buffer_size, cudaMemAttachGlobal);
        } else {
            MEM_ALLOC->buffers[MEM_ALLOC->n_buffers].start = memalign(/* boundary */ page_size,
                                                                                     buffer_size);
        }

        if (!MEM_ALLOC->buffers[MEM_ALLOC->n_buffers].start) {
            LOG_ERROR("Out of memory");
            exit(EXIT_FAILURE);
        }
    }
}

int Device::set_ctrl(int fd, unsigned int id, int val) {
    int rc;
    struct v4l2_ext_controls ctrls;
    struct v4l2_ext_control ctrl;

    memset(&ctrls, 0, sizeof(ctrls));
    memset(&ctrl, 0, sizeof(ctrl));

    ctrls.ctrl_class = V4L2_CTRL_ID2CLASS(id);
    ctrls.count = 1;
    ctrls.controls = &ctrl;


    ctrl.id = id;
    // if (is_64 == NV_TRUE)
    //ctrl.value64 = val;
    //else
    ctrl.value = val;

    rc = xioctl(fd, VIDIOC_S_EXT_CTRLS, &ctrls);
    if (rc) {
        struct v4l2_control old;

        old.id = id;
        old.value = val;
        rc = xioctl(fd, VIDIOC_S_CTRL, &old);
        LOG_INFO("rc: ", rc);
        if (rc) {
            LOG_ERROR("Failed to set control ", id, "-> ", strerror(errno));
            return rc;
        }
    }


    return 0;
}


int Device::get_ctrl(int fd, int id, int *val) {
    int ee, rc;
    struct v4l2_ext_controls ctrls;
    struct v4l2_ext_control ctrl;

    memset(&ctrls, 0, sizeof(ctrls));
    memset(&ctrl, 0, sizeof(ctrl));

    ctrls.ctrl_class = V4L2_CTRL_ID2CLASS(id);
    ctrls.count = 1;
    ctrls.controls = &ctrl;


    ctrl.id = id;
    // if (is_64 == NV_TRUE)
    //ctrl.value64 = val;
    //else
    //ctrl.value = val;

    rc = xioctl(fd, VIDIOC_G_EXT_CTRLS, &ctrls);
    LOG_INFO("G_EXT_CTRLS rc= ", rc);
    if (rc) {
        struct v4l2_control old;

        old.id = id;
        //old.value = val;
        rc = xioctl(fd, VIDIOC_G_CTRL, &old);
        printf("%s:%d rc:%d\n", __FILE__, __LINE__, rc);
        if (rc) {
            LOG_ERROR("Failed to get control ", id, "->", strerror(errno));
            return rc;
        } else {
            *val = old.value;
        }
    } else {
        *val = ctrl.value;
    }

    return 0;
}


int Device::sensor_set_parameters(int newGain, int newShutter) {
    int rc;
    unsigned int val;

    struct v4l2_ext_controls ctrls;
    struct v4l2_ext_control ctrl;

    rc = set_ctrl(fd, TEGRA_CAMERA_CID_LOW_LATENCY, 0);
    LOG_INFO("LOW_LATENCY rc= ", rc);

    rc = set_ctrl(fd, TEGRA_CAMERA_CID_OVERRIDE_ENABLE, 0);
    LOG_INFO("OVERRIDE_ENABLE rc= ", rc);

    rc = set_ctrl(fd, TEGRA_CAMERA_CID_VI_BYPASS_MODE, 0);
    LOG_INFO("BYPASS_MODE rc= ", rc);

    rc = set_ctrl(fd, TEGRA_CAMERA_CID_FRAME_RATE, 60000000);
    LOG_INFO("FRAME_RATE rc= ", rc);

    rc = set_ctrl(fd, TEGRA_CAMERA_CID_VI_HEIGHT_ALIGN, 1);
    LOG_INFO("HEIGHT_ALIGN rc= ", rc);

    rc = set_ctrl(fd, TEGRA_CAMERA_CID_VI_SIZE_ALIGN, 0);
    LOG_INFO("SIZE_ALIGN rc= ", rc);

    rc = set_ctrl(fd, TEGRA_CAMERA_CID_GROUP_HOLD, 1);
    LOG_INFO("GROUP_HOLD rc= ", rc);

    rc = set_ctrl(fd, TEGRA_CAMERA_CID_WRITE_ISPFORMAT, 0);
    LOG_INFO("WRITE_ISPFORMAT rc= ", rc);

    rc = set_ctrl(fd, TEGRA_CAMERA_CID_VI_PREFERRED_STRIDE, 0);
    LOG_INFO("PREFERRED_STRIDE rc= ", rc);


    rc = set_ctrl(fd, TEGRA_CAMERA_CID_EXPOSURE, 0);
    LOG_INFO("CID_EXPOSURE rc= ", rc);

    rc = get_ctrl(fd, TEGRA_CAMERA_CID_EXPOSURE, (int *) &val);
    LOG_INFO("CID_EXPOSURE rc= ", rc);


    rc = set_ctrl(fd, TEGRA_CAMERA_CID_EXPOSURE, newShutter);
    LOG_INFO("CID_EXPOSURE rc= ", rc);


    rc = set_ctrl(fd, TEGRA_CAMERA_CID_GAIN, 0);
    LOG_INFO("CID_GAIN rc= ", rc);

    rc = get_ctrl(fd, TEGRA_CAMERA_CID_GAIN, (int *) &val);
    LOG_INFO("CID_GAIN val= ", val);


    rc = set_ctrl(fd, TEGRA_CAMERA_CID_GAIN, newGain);
    LOG_INFO("CID_GAIN rc= ", rc);

    return (rc);
}


void Device::init_device(void) {
    struct v4l2_capability cap;
    struct v4l2_cropcap cropcap;
    struct v4l2_crop crop;
    struct v4l2_format fmt;
    unsigned int min;

    int optShutter, optFBOutIff1, optStdOutIff1, optBufCount;
    int optGain;

    optStdOutIff1 = +1;
    optFBOutIff1 = -1;
    optShutter = 5000;
    optGain = 50;
    optBufCount = 3;


    if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &cap)) {
        if (EINVAL == errno) {
            LOG_ERROR(dev_name, " is no V4L2 device");
            exit(EXIT_FAILURE);
        } else {
            errno_exit("VIDIOC_QUERYCAP");
        }
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        LOG_ERROR(dev_name, " is no video capture device");
        exit(EXIT_FAILURE);
    }

    switch (io) {
        case IO_METHOD_READ:
            if (!(cap.capabilities & V4L2_CAP_READWRITE)) {
                LOG_ERROR(dev_name, " does not support read i/o");
                exit(EXIT_FAILURE);
            }

            break;

        case IO_METHOD_MMAP:
        case IO_METHOD_USERPTR:
            if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
                LOG_ERROR(dev_name, " does not support streaming i/o");
                exit(EXIT_FAILURE);
            }

            break;
    }


    CLEAR(fmt);


    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = UINT_MAX;
    fmt.fmt.pix.height = UINT_MAX;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_Y10;

    if (-1 == xioctl(fd, VIDIOC_G_FMT, &fmt))
        errno_exit("VIDIOC_G_FMT");

    LOG_INFO("width: ", fmt.fmt.pix.width, "\theight: ", fmt.fmt.pix.height);
    MEM_ALLOC->width = width = fmt.fmt.pix.width;
    MEM_ALLOC->height = height = fmt.fmt.pix.height;


    if (-1 == sensor_set_parameters(optGain, optShutter))
        errno_exit("sensor_set_parameters");


    switch (io) {
        case IO_METHOD_READ:
            init_read(fmt.fmt.pix.sizeimage);
            break;

        case IO_METHOD_MMAP:
            init_mmap();
            break;

        case IO_METHOD_USERPTR:
            init_userp(fmt.fmt.pix.sizeimage);
            break;
    }
}

void Device::close_device(void) {
    if (-1 == close(fd))
        errno_exit("close");

    fd = -1;
}

void Device::open_device(void) {
    struct stat st;

    if (-1 == stat(dev_name.c_str(), &st)) {
        LOG_ERROR("Cannot identify ", dev_name, "->", errno, "->", strerror(errno));
        exit(EXIT_FAILURE);
    }

    if (!S_ISCHR(st.st_mode)) {
        LOG_ERROR(dev_name, " is no device");
        exit(EXIT_FAILURE);
    }

    fd = open(dev_name.c_str(), O_RDWR /* required */ | O_NONBLOCK, 0);

    if (-1 == fd) {
        LOG_ERROR("Cannot open ", dev_name, "->", errno, "->", strerror(errno));
        exit(EXIT_FAILURE);
    }
}

void Device::init_cuda(void) {
    /* Check unified memory support. */
    if (cuda_zero_copy) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);
        if (!devProp.managedMemory) {
            LOG_INFO("CUDA device does not support managed memory");
            cuda_zero_copy = false;
        }
    }

    /* Allocate output buffer. */
//    size_t size = width * height * 3;
    size_t size = width * height * sizeof(unsigned char);
    if (cuda_zero_copy) {
        LOG_INFO("cudaMallocManaged");
        cudaMallocManaged(&MEM_ALLOC->cuda_out_buffer, size, cudaMemAttachGlobal);
        cudaMallocManaged(&MEM_ALLOC->intensity_num, INTENSITY_RANGE * sizeof(unsigned int), cudaMemAttachGlobal);
        cudaMallocManaged(&MEM_ALLOC->intensity_pro, INTENSITY_RANGE * sizeof(double), cudaMemAttachGlobal);
        cudaMallocManaged(&MEM_ALLOC->min_index, 1 * sizeof(unsigned int), cudaMemAttachGlobal);
        cudaMallocManaged(&MEM_ALLOC->max_index, 1 * sizeof(unsigned int), cudaMemAttachGlobal);

    } else {
        LOG_INFO("malloc");
        MEM_ALLOC->cuda_out_buffer = (unsigned char *) malloc(size);
        MEM_ALLOC->intensity_num = (unsigned int *) malloc(INTENSITY_RANGE * sizeof(unsigned int));
        MEM_ALLOC->intensity_pro = (double *) malloc(INTENSITY_RANGE * sizeof(double));
        MEM_ALLOC->min_index = (unsigned int *) malloc(1 * sizeof(unsigned int));
        MEM_ALLOC->max_index = (unsigned int *) malloc(1 * sizeof(unsigned int));
    }

    cudaDeviceSynchronize();
}
