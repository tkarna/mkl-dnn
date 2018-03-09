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

struct jit_decomp
{
    size_t MB;
    size_t ODB;
    size_t OHB;
    size_t OWB;
};

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

template struct jit_avx512_common_convolution3D_1ch_bwd_weights_t<f32, f32, f32, f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
