//
// Created by Andrey Pahomov on 03.09.20.
//

#ifndef V4L_CUDA_GIST_MEMALLOC_H
#define V4L_CUDA_GIST_MEMALLOC_H

#include <mutex>
#include <atomic>

#include <cuda_runtime.h>
#include "histogram.cuh"

struct buffer {
    void *start;
    size_t length;
};


class MemAlloc {
    static std::atomic<MemAlloc *> instance;
    static std::mutex mutexConfigs;

    MemAlloc() = default;

    ~MemAlloc() = default;

    MemAlloc(const MemAlloc &) = delete;

    MemAlloc &operator=(const MemAlloc &) = delete;

public:
    struct buffer *buffers;
    unsigned int n_buffers;
    unsigned char *cuda_out_buffer;
    unsigned int *intensity_num;
    double *intensity_pro;
    unsigned int *min_index;
    unsigned int *max_index;

    unsigned int width;
    unsigned int height;

    static MemAlloc *get() {

        MemAlloc *sin = instance.load(std::memory_order_acquire);

        if (!sin) {
            std::lock_guard<std::mutex> myLock(mutexConfigs);
            sin = instance.load(std::memory_order_relaxed);
            if (!sin) {
                sin = new MemAlloc();
                instance.store(sin, std::memory_order_release);
            }
        }
        return sin;

    };


};


#define MEM_ALLOC MemAlloc::get()

#endif //V4L_CUDA_GIST_MEMALLOC_H
