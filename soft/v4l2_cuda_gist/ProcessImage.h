//
// Created by Andrey Pahomov on 03.09.20.
//

#ifndef V4L_CUDA_GIST_PROCESSIMAGE_H
#define V4L_CUDA_GIST_PROCESSIMAGE_H

#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

class ProcessImage {

public:
    static void Init(std::string windowsName);

    static void Processing(void *p);

    static void DrawImage(void *p);

};


#endif //V4L_CUDA_GIST_PROCESSIMAGE_H
