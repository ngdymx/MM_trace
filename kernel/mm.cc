#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>
#include "mm.h"

#ifndef DIM_M
#define DIM_M 8
#endif

#ifndef DIM_N
#define DIM_N 128
#endif

extern "C"{

void mm(
    bfloat16 *restrict a,
    bfloat16 *restrict b,
    bfloat16 *restrict c
    ){
    mm_bf16<DIM_N>(a, b, c);
}

}
