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

#include <assert.h>
#include <math.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "math_utils.hpp"
#include "nstl.hpp"

#include "avx512_common_pooling3D.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t data_type, data_type_t acc_type>
void avx512_common_pooling3D_fwd_t<data_type, acc_type>::execute_forward() {
    using namespace alg_kind;
    using namespace prop_kind;
    // auto alg = conf_.desc()->alg_kind;

    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t *>(this->memory(0));
    auto ws =  reinterpret_cast<unsigned char *>(this->memory(1));

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper ws_d(conf_.workspace_pd());

    const int IH = 128;
    const int IW = 128;
    const int ID = 128;
    const int KH = 2;
    const int KW = 2;
    const int KD = 2;
    const int SH = 1;
    const int SW = 1;
    const int SD = 1;
    const int padT = 0;
    const int padL = 0;
    const int padD1 = 0;

    const int MB = conf_.MB();
    const int OC = 16;
    const int OD = 127;
    const int OH = 127;
    const int OW = 127;

    const int s_dstride = src_d.off(0, 0, 1, 0, 0) - src_d.off(0, 0, 0, 0, 0);
    const int s_hstride = src_d.off(0, 0, 0, 1, 0) - src_d.off(0, 0, 0, 0, 0);


#   pragma omp parallel for collapse(4) schedule(static)
    for (int mb = 0; mb < MB; ++mb) {
        for (int oc = 0; oc < OC; ++oc) {
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    data_t *d = &dst[dst_d.off(mb, oc, od, oh, 0)];
                    const data_t *s = &src[src_d.off(mb, oc, od, oh, 0)];
                    auto *w = &ws[ws_d.off(mb, oc, od, oh, 0)];
#                   pragma omp simd
                    for (int ow = 0; ow < OW; ++ow) {
                        data_t cand_d = nstl::numeric_limits<data_t>::lowest();
                        auto cand_w = 0;
                        for (int kd = 0; kd < KD; ++kd) {
                            for (int kh = 0; kh < KH; ++kh) {
                                for (int kw = 0; kw < KW; ++kw) {
                                    const int id = od * SD - padD1 + kd;
                                    const int ih = oh * SH - padT + kh;
                                    const int iw = ow * SW - padL + kw;

                                    const bool outside = id >= ID || ih >= IH || iw >= IW;

                                    auto cand_s = s[s_dstride*kd + s_hstride*kh + iw];
                                    if (!outside && cand_s > cand_d) {
                                        cand_d = cand_s;
                                        cand_w = kd*KH*KW + kh*KW + kw;
                                    }
                                }
                            }
                        }
                        d[ow] = cand_d;
                        w[ow] = cand_w;
                    }
                }
            }
        }
    }
}

template struct avx512_common_pooling3D_fwd_t<data_type::f32>;
template struct avx512_common_pooling3D_fwd_t<data_type::s32>;
template struct avx512_common_pooling3D_fwd_t<data_type::s16, data_type::s32>;
template struct avx512_common_pooling3D_fwd_t<data_type::s8, data_type::s32>;
template struct avx512_common_pooling3D_fwd_t<data_type::u8, data_type::s32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
