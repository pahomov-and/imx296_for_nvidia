//
// Created by Andrey Pahomov on 28.08.20.
//

#ifndef __HISTOGRAM_CUH__
#define __HISTOGRAM_CUH__

#define TILE_SIZE 512

#define INTENSITY_RANGE 1024
#define INTENSITY_MASK 0x3FF


void gpuConvertY10to8uc1(unsigned short *src, unsigned char *dst,
                         unsigned int width, unsigned int height);


void gpuConvertY10to8uc1_gist(  unsigned short *src, unsigned char *dst,
                                unsigned int width, unsigned int height,
                                unsigned int *intensity_num,
                                double *intensity_pro,
                                unsigned int *min_index,
                                unsigned int *max_index,
                                bool isMatGist);


#endif
