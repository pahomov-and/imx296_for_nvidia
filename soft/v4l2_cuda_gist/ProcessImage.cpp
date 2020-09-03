//
// Created by Andrey Pahomov on 03.09.20.
//
#include <iostream>

#include "ProcessImage.h"
#include "MemAlloc.h"
#include "histogram.cuh"
#include "LOG.h"

cv::Mat image;
cv::Rect imageROI;

std::string windows_name;

void ProcessImage::Init(std::string windowsName) {
    image = cv::Mat(MEM_ALLOC->height, MEM_ALLOC->width, CV_8UC1);
    imageROI = cv::Rect(0, 0, 800, 800);

    windows_name = windowsName;

    cv::namedWindow(windows_name, cv::WINDOW_AUTOSIZE);
}

int counHistFarme = 0;
bool isMatGist = true;

void ProcessImage::Processing(void *p) {
    if (counHistFarme > 5) {
        counHistFarme = 0;
        isMatGist = true;
    }

/* simple convert 10bit to 80bit */
/*
    gpuConvertY10to8uc1 ((unsigned short *) p,
            MEM_ALLOC->cuda_out_buffer,
            MEM_ALLOC->width,
            MEM_ALLOC->height);
*/

/* automatic adjustment of camera dynamic range from 10 bit to 8 bit*/
    gpuConvertY10to8uc1_gist(
            (unsigned short *) p,
            MEM_ALLOC->cuda_out_buffer,
            MEM_ALLOC->width,
            MEM_ALLOC->height,
            MEM_ALLOC->intensity_num,
            MEM_ALLOC->intensity_pro,
            MEM_ALLOC->min_index,
            MEM_ALLOC->max_index, isMatGist);

//    LOG_INFO("Gistogram left: ", *MEM_ALLOC->min_index,  " right: ", *MEM_ALLOC->max_index);

    isMatGist = false;
    counHistFarme++;

    std::fill(MEM_ALLOC->intensity_num, MEM_ALLOC->intensity_num + INTENSITY_RANGE, 0);
}

void ProcessImage::DrawImage(void *p) {
    image.data = reinterpret_cast<unsigned char *>(p);

    image = image(imageROI);
    cv::imshow(windows_name, image);
    cv::waitKey(1);
}
