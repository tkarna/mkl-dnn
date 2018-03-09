/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "mkldnn_traits.hpp"
#include "math_utils.hpp"

#include "jit_avx512_common_convolution3D_1ch.hpp"
#include <iostream>

namespace mkldnn {
namespace impl {
namespace cpu {

using math::saturate;

class MultiviewOffset {
    /* Computes offsets for multidimensional arrays */
    size_t dims[5];
public:
    MultiviewOffset(size_t n0, size_t n1, size_t n2, size_t n3, size_t n4) {
        dims[0] = n0;
        dims[1] = n1;
        dims[2] = n2;
        dims[3] = n3;
        dims[4] = n4;
    };
    inline size_t off(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
        return (((i0*dims[1] + i1)*dims[2] + i2)*dims[3] + i3)*dims[4] + i4;
    };
};

struct jit_decomp
{
    size_t MB;
    size_t ODB;
    size_t OHB;
    size_t OWB;
};

template <bool with_relu, data_type_t src_type, data_type_t wei_type,
         data_type_t dst_type, data_type_t acc_type>
void _jit_avx512_common_convolution3D_1ch_fwd_t<with_relu, src_type, wei_type, dst_type, acc_type>
        ::execute_forward() {

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    // const bool with_groups = conf_.with_groups();

    const int G = conf_.G();
    const int MB = conf_.MB();

    const int OH = conf_.OH();
    const int OW = conf_.OW();
    const int OD = conf_.OD();

    const int OBLOCK = 28;  // Don't change without changing jit
    const int OWREM = conf_.OW() % OBLOCK;
    const int OWB = conf_.OW() / OBLOCK + (OWREM > 0);

    const int IH = conf_.IH();
    const int IW = conf_.IW();
    const int ID = conf_.ID();

    const int NBLOCK = 16;
    const int OCB = conf_.OC() / G / NBLOCK;
    const int IC = conf_.IC() / G;

    const int KH = conf_.KH();
    const int KW = conf_.KW();
    const int KD = conf_.KD();

    const int KSH = conf_.KSH();
    const int KSW = conf_.KSW();
    const int KSD = conf_.KSD();

    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    auto src_ix = MultiviewOffset(MB, IC, ID, IH, IW);
    auto dst_ix = MultiviewOffset(MB, OCB, OD, OH, OW);
    auto w_ix = MultiviewOffset(OCB, IC, KD, KH, KW);

    const float nslope = conf_.negative_slope();

    typedef void (*jitkernel_t)(const float *, const float *, float *, uint8_t *, const float *);
    jitkernel_t jitkernel = (jitkernel_t) kernel_->kernel_->jit_ker;
    jitkernel_t jitkernelr = (jitkernel_t) kernel_->kernelrem_->jit_ker;

    auto get_bias = [=, &bias](size_t off) -> acc_data_t {
#       define CASE(dt) case dt: \
        return (acc_data_t)(*((const prec_traits<dt>::type *)bias + off))
        switch (conf_.cdesc()->bias_desc.data_type) {
        CASE(data_type::s8);
        CASE(data_type::u8);
        CASE(data_type::s32);
        CASE(data_type::f32);
        default: assert(!"unimplemented");
        }
#       undef CASE
        return 0;
    };

#   pragma omp parallel for collapse(5) schedule(static)
    for (int g = 0; g < G; ++g) {
        for (int mb = 0; mb < MB; ++mb) {
            for (int ocb = 0; ocb < OCB; ++ocb) {
                for (int od = 0; od < OD; ++od) {
                    for (int oh = 0; oh < OH; ++oh) {
                        // case 1: full OBLOCKs
                        for (int owb = 0; owb < OWB - 1 + (OWREM==0); ++owb) {
                            for (int _ow = 0; _ow < OBLOCK; ++_ow) {
#                               pragma omp simd
                                for (int _oc = 0; _oc < NBLOCK; ++_oc) {
                                    dst[dst_ix.off(mb, ocb, od, oh, owb*OBLOCK + _ow)*NBLOCK + _oc] =
                                        (bias) ? get_bias((g*OCB + ocb)*NBLOCK + _oc) : (acc_data_t)0;
                                }
                            }
                            const int id = od * KSD;
                            const int ih = oh * KSH;
                            for (int ic = 0; ic < IC; ++ic) {
                                int iwbase = owb*OBLOCK*KSW;
                                jitkernel(&src[src_ix.off(mb, ic, id, ih, iwbase)],
                                       &weights[w_ix.off(ocb, ic, 0, 0, 0)*NBLOCK],
                                       &dst[dst_ix.off(mb, ocb, od, oh, owb*OBLOCK)*NBLOCK],
                                       0 /* TODO: bias */, &nslope);
                            }
                            for (int _ow = 0; _ow < OBLOCK; ++_ow) {
#                               pragma omp simd
                                for (int _oc = 0; _oc < NBLOCK; ++_oc) {
                                    auto d = dst[dst_ix.off(mb, ocb, od, oh, owb*OBLOCK + _ow)*NBLOCK + _oc];
                                    if (with_relu && d < (acc_data_t)0)
                                        d = (acc_data_t)((float)d * nslope);
                                    dst[dst_ix.off(mb, ocb, od, oh, owb*OBLOCK + _ow)*NBLOCK + _oc] = saturate<dst_data_t>(d);
                                }
                            }
                        }
                        // case 2: remainder
                        if (OWREM > 0)
                        for (int owb = OWB-1; owb < OWB; ++owb) {
                            for (int _ow = 0; _ow < OWREM; ++_ow) {
#                               pragma omp simd
                                for (int _oc = 0; _oc < NBLOCK; ++_oc) {
                                    dst[dst_ix.off(mb, ocb, od, oh, owb*OBLOCK + _ow)*NBLOCK + _oc] =
                                        (bias) ? get_bias((g*OCB + ocb)*NBLOCK + _oc) : (acc_data_t)0;
                                }
                            }
                            const int id = od * KSD;
                            const int ih = oh * KSH;
                            for (int ic = 0; ic < IC; ++ic) {
                                int iwbase = owb*OBLOCK*KSW;
                                jitkernelr(&src[src_ix.off(mb, ic, id, ih, iwbase)],
                                       &weights[w_ix.off(ocb, ic, 0, 0, 0)*NBLOCK],
                                       &dst[dst_ix.off(mb, ocb, od, oh, owb*OBLOCK)*NBLOCK],
                                       0 /* TODO: bias */, &nslope);
                            }
                            for (int _ow = 0; _ow < OWREM; ++_ow) {
#                               pragma omp simd
                                for (int _oc = 0; _oc < NBLOCK; ++_oc) {
                                    auto d = dst[dst_ix.off(mb, ocb, od, oh, owb*OBLOCK + _ow)*NBLOCK + _oc];
                                    if (with_relu && d < (acc_data_t)0)
                                        d = (acc_data_t)((float)d * nslope);
                                    dst[dst_ix.off(mb, ocb, od, oh, owb*OBLOCK + _ow)*NBLOCK + _oc] = saturate<dst_data_t>(d);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <data_type_t src_type, data_type_t diff_wei_type,
         data_type_t diff_dst_type, data_type_t acc_type>
void jit_avx512_common_convolution3D_1ch_bwd_weights_t<src_type, diff_wei_type, diff_dst_type,
     acc_type>::execute_backward_weights() {
    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const diff_dst_data_t *>(
            this->input_memory(1));
    auto diff_weights = reinterpret_cast<diff_wei_data_t*>(this->memory(0));
    auto diff_bias = reinterpret_cast<diff_wei_data_t *>(this->memory(1));

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_weights_d(conf_.diff_weights_pd(0));
    const memory_desc_wrapper diff_bias_d(conf_.diff_weights_pd(1));

    // const int G = conf_.G();
    const int MB = conf_.MB();
    const int OH = conf_.OH();
    const int OW = conf_.OW();
    const int OD = conf_.OD();
    const int IH = conf_.IH();
    const int IW = conf_.IW();
    // const int ID = conf_.ID();

    const int NBLOCK = 16;
    // const int OCB = conf_.OC() / G / NBLOCK;
    // const int IC = conf_.IC() / G;

    const int KH = conf_.KH();
    const int KW = conf_.KW();
    const int KD = conf_.KD();

    // const int KSH = conf_.KSH();
    // const int KSW = conf_.KSW();
    // const int KSD = conf_.KSD();

    const int32_t max_nthr = omp_get_max_threads();
    diff_wei_data_t private_weights[KD*KH*KW][max_nthr][NBLOCK] __attribute__((aligned(64)));
    acc_data_t private_bias[max_nthr][NBLOCK] __attribute__((aligned(64)));

    void (*kernel)(const diff_dst_data_t*,const src_data_t*,diff_wei_data_t*,diff_wei_data_t*, struct jit_decomp*) = (void (*)(const diff_dst_data_t*,const src_data_t*,diff_wei_data_t*,diff_wei_data_t*,struct jit_decomp*)) kernel_->jit_ker;
#   pragma omp parallel
    {
        const int tid = omp_get_thread_num();

        // TODO: These numbers need to be set according to # threads
        const size_t ODBLOCK = 32;
        const size_t OHBLOCK = 32;
        const size_t OWBLOCK = 32;
#       pragma omp for collapse(3)
        for (size_t odb = 0; odb < (size_t)OD; odb += ODBLOCK)
        {
            for (size_t ohb = 0; ohb < (size_t)OH; ohb += OHBLOCK)
            {
                for (size_t owb = 0; owb < (size_t)OW; owb += OWBLOCK)
                {
                    jit_decomp decomp = {(size_t)MB, std::min(ODBLOCK, (size_t)(OD - (int)odb)), std::min(OHBLOCK, (size_t)(OH - (int)ohb)), std::min(OWBLOCK, (size_t)(OW - (int)owb)) };
                    kernel(&diff_dst[odb*OH*OW*NBLOCK + ohb*OW*NBLOCK + owb*NBLOCK], &src[odb*IH*IW + ohb*IW + owb], &private_weights[0][tid][0], &private_bias[tid][0], &decomp);
                }
            }
        }

        // TODO: Put the reduction code into a function.
#       pragma omp for
        for (int k = 0; k < KD*KH*KW; ++k)
        {
#           pragma omp simd
            for (int _oc = 0; _oc < NBLOCK; ++_oc)
            {
                acc_data_t sum = 0;
                for (int t = 0; t < max_nthr; ++t)
                {
                    sum += private_weights[k][t][_oc];
                }
                diff_weights[k*NBLOCK + _oc] = sum;
            }
        }
        if (diff_bias)
        {
#           pragma omp master
#           pragma omp simd
            for (int _oc = 0; _oc < NBLOCK; ++_oc)
            {
                acc_data_t sum = 0;
                for (int t = 0; t < max_nthr; ++t)
                {
                    sum += private_bias[t][_oc];
                }
                diff_bias[_oc] = sum;
            }
        }
    }
}

using namespace data_type;

template struct _jit_avx512_common_convolution3D_1ch_fwd_t<false, f32>;
template struct _jit_avx512_common_convolution3D_1ch_fwd_t<true, f32>;
template struct jit_avx512_common_convolution3D_1ch_bwd_weights_t<f32, f32, f32, f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
