//
// Created by Andrey Pahomov on 02.09.20.
//

#ifndef V4L_CUDA_LOG_H
#define V4L_CUDA_LOG_H

#include <iostream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <initializer_list>


template<typename T>
void log_cout(T t) {
    std::cout << t;
}

template<typename T, typename... Args>
void log_cout(T t, Args... args) {
    std::cout << t;
    log_cout(args...);
}


template<typename T>
void log_cerr(T t) {
    std::cerr << t;
}

template<typename T, typename... Args>
void log_cerr(T t, Args... args) {
    std::cerr << t;
    log_cerr(args...);
}


#define LOG_INFO(...) (log_cout("[INFO]\t[",std::time(nullptr),"]\t",__FILE__,":", __LINE__, ": ", __func__ , ": ", ##__VA_ARGS__, "\n") )
#define LOG_ERROR(...) (log_cerr("[ERROR]\t[",std::time(nullptr),"]\t",__FILE__,":", __LINE__, ": ", __func__ , ": ", ##__VA_ARGS__, "\n") )

#endif //V4L_CUDA_LOG_H
