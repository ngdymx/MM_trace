#ifndef __MM_H__
#define __MM_H__
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

template <unsigned colA, unsigned r, unsigned s, unsigned t>
static inline void mm_aie(const bfloat16 *__restrict pA, const bfloat16 *__restrict pB, bfloat16 *__restrict pC) {
    using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16, accfloat>;

    event0();

    bfloat16 *__restrict pC1 = pC;

    const bfloat16 *__restrict pA1 = pA;
    const bfloat16 *__restrict pB1 = pB; 
    const bfloat16 *__restrict pB2 = pB + 1 * MMUL::size_B;

    aie::vector<bfloat16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
    pA1 += MMUL::size_A;
    aie::vector<bfloat16, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
    pB1 += MMUL::size_B * 2;
    aie::vector<bfloat16, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);
    pB2 += MMUL::size_B * 2;

    aie::vector<bfloat16, MMUL::size_C> acc_C00 = aie::zeros<bfloat16, MMUL::size_C>(); 
    aie::vector<bfloat16, MMUL::size_C> acc_C01 = aie::zeros<bfloat16, MMUL::size_C>(); 

    MMUL C00(acc_C00);
    MMUL C01(acc_C01);

    C00.mac(A0, B0);
    C01.mac(A0, B1);

    for (unsigned i = 1; i < colA; ++i)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
        A0 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += MMUL::size_A;

        B0 = aie::load_v<MMUL::size_B>(pB1);
        pB1 += MMUL::size_B * 2;
        B1 = aie::load_v<MMUL::size_B>(pB2);
        pB2 += MMUL::size_B * 2;

        C00.mac(A0, B0);
        C01.mac(A0, B1);
    }

    aie::store_v(pC1, C00.template to_vector<bfloat16>());
    pC1 += MMUL::size_C;
    aie::store_v(pC1, C01.template to_vector<bfloat16>());
    pC1 += MMUL::size_C;
    event1();
}

template <unsigned m>
static inline void mm_bf16(const bfloat16 *__restrict pA, const bfloat16 *__restrict pB, bfloat16 *__restrict pC) {
    constexpr int r = 8;
    constexpr int s = 8;
    constexpr int t = 8;
    return mm_aie<m / s, r, s, t>(pA, pB, pC);
}

#endif
