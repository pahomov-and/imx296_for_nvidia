//
// Created by Andrey Pahomov on 03.09.20.
//

#include "MemAlloc.h"

std::atomic<MemAlloc *> MemAlloc::instance;
std::mutex MemAlloc::mutexConfigs;
