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

#include "cpu_convolution3D_fk1s1.hpp"
#include <iostream>


class MultiviewOffset2 {
    /* Computes offsets for multidimensional arrays */
    size_t dims[2];
public:
    MultiviewOffset2(size_t n0, size_t n1) {
        dims[0] = n0;
        dims[1] = n1;
    };
    inline size_t off(size_t i0, size_t i1) {
        return i0*dims[1] + i1;
    };
};

class MultiviewOffset5 {
    /* Computes offsets for multidimensional arrays */
    size_t dims[5];
public:
    MultiviewOffset5(size_t n0, size_t n1, size_t n2, size_t n3, size_t n4) {
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


namespace mkldnn {
namespace impl {
namespace cpu {

using math::saturate;

template <bool with_relu, data_type_t src_type, data_type_t wei_type,
         data_type_t dst_type, data_type_t acc_type>
void _convolution3D_fk1s1_fwd_t<with_relu, src_type, wei_type, dst_type, acc_type>
        ::execute_forward() {

    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));
    const memory_desc_wrapper bias_d(conf_.weights_pd(1));

    //Assumes KD() == 1 && KH() == 1 && KW() == 1
    //        KSD() == 1 && KSH() == 1 && KSW() == 1
    //        KDD() == 0 && KDH() == 0 && KDW() == 0
    //        padT() == 0 && padL() == 0 && padD1() == 0
    //        G() == 1

    const int MB = conf_.MB();
    const int OH = conf_.OH();
    const int OW = conf_.OW();
    const int OD = conf_.OD();

    const int IH = conf_.IH();
    const int IW = conf_.IW();
    const int ID = conf_.ID();

    const int OC = conf_.OC();

    const int NBLOCK = 16;
    const int ICB = conf_.IC() / NBLOCK;

    const float nslope = conf_.negative_slope();

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


    auto src_ix = MultiviewOffset5(MB, ICB, ID, IH, IW);
    auto dst_ix = MultiviewOffset5(MB, OC, OD, OH, OW);
    auto w_ix = MultiviewOffset2(OC, ICB);

#   pragma omp parallel for collapse(4) schedule(static)
        for (int mb = 0; mb < MB; ++mb) {
            for (int oc = 0; oc < OC; ++oc) {
                for (int od = 0; od < OD; ++od) {
                    for (int oh = 0; oh < OH; ++oh) {
                        for (int ow = 0; ow < OW; ++ow) {
                            acc_data_t a = bias
                                ? get_bias(bias_d.off(oc)) : (acc_data_t)0;
                            
                            for (int icb = 0; icb < ICB; ++icb) 
                            #pragma omp simd reduction(+:a)
                            for (int _ic = 0; _ic < NBLOCK; ++_ic) 
                                a += (acc_data_t)src[src_ix.off(mb, icb, od, oh, ow)*NBLOCK + _ic]
                                     * weights[w_ix.off(oc, icb)*NBLOCK + _ic];                            
                            
                            if (with_relu && a < (acc_data_t)0)
                                a = (acc_data_t)((float)a * nslope);
                            dst[dst_ix.off(mb, oc, od, oh, ow)]
                                = saturate<dst_data_t>(a);
                        }
                    }
                }
            }
        }
}

using namespace data_type;

template struct _convolution3D_fk1s1_fwd_t<false, f32>;
template struct _convolution3D_fk1s1_fwd_t<true, f32>;
// template struct _convolution3D_fk1s1_fwd_t<false, s16, s16, s32, s32>;
// template struct _convolution3D_fk1s1_fwd_t<true, s16, s16, s32, s32>;
// template struct _convolution3D_fk1s1_fwd_t<false, u8, s8, s32, s32>;
// template struct _convolution3D_fk1s1_fwd_t<true, u8, s8, s32, s32>;
// template struct _convolution3D_fk1s1_fwd_t<false, u8, s8, s8, s32>;
// template struct _convolution3D_fk1s1_fwd_t<true, u8, s8, s8, s32>;
// template struct _convolution3D_fk1s1_fwd_t<false, u8, s8, u8, s32>;
// template struct _convolution3D_fk1s1_fwd_t<true, u8, s8, u8, s32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
