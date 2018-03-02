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
    auto alg = conf_.desc()->alg_kind;

    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t *>(this->memory(0));
    auto ws = alg == pooling_max && conf_.desc()->prop_kind == forward_training
        ? reinterpret_cast<unsigned char *>(this->memory(1)) : nullptr;

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper ws_d(conf_.workspace_pd());
    const data_type_t ws_dt = ws ? ws_d.data_type() : data_type::undef;

    const int IH = conf_.IH();
    const int IW = conf_.IW();
    const int ID = conf_.ID();
    const int KH = conf_.KH();
    const int KW = conf_.KW();
    const int KD = conf_.KD();
    const int SH = conf_.KSH();
    const int SW = conf_.KSW();
    const int SD = conf_.KSD();
    const int padT = conf_.padT();
    const int padL = conf_.padL();
    const int padD1 = conf_.padD1();

    auto apply_offset = [=](int index, int offset) {
        return (index > offset) ? index - offset : 0;
    };

    auto ker_max = [=](data_t *d, int mb, int oc, int od, int oh, int ow) {
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    const int id = od * SD - padD1 + kd;
                    const int ih = oh * SH - padT + kh;
                    const int iw = ow * SW - padL + kw;

                    if (id < 0 || id >= ID) continue;
                    if (ih < 0 || ih >= IH) continue;
                    if (iw < 0 || iw >= IW) continue;

                    auto s = src[src_d.off(mb, oc, id, ih, iw)];
                    if (s > d[0]) {
                        d[0] = s;
                        if (ws) {
                            size_t off = ws_d.off(mb, oc, od, oh, ow);
                            if (ws_dt == data_type::u8) {
                                ws[off] = kd*KH*KW + kh*KW + kw;
                            } else {
                                assert(ws_dt == data_type::s32);
                                ((int *)ws)[off] = kd*KH*KW + kh*KW + kw;
                            }
                        }
                    }
                }
            }
        }
    };

    auto ker_avg = [=](data_t *d, int mb, int oc, int od, int oh, int ow) {
        auto id_start = apply_offset(od*SD, padD1);
        auto ih_start = apply_offset(oh*SH, padT);
        auto iw_start = apply_offset(ow*SW, padL);
        auto id_end = nstl::min(od*SD - padD1 + KD, ID);
        auto ih_end = nstl::min(oh*SH - padT + KH, IH);
        auto iw_end = nstl::min(ow*SW - padL + KW, IW);

        auto num_summands = (alg == pooling_avg_include_padding) ? KD*KW*KH
            : (id_end - id_start)*(ih_end - ih_start)*(iw_end - iw_start);

        acc_data_t dst = 0;
        for (int id = id_start; id < id_end; ++id) {
            for (int ih = ih_start; ih < ih_end; ++ih) {
                for (int iw = iw_start; iw < iw_end; ++iw) {
                    dst += src[src_d.off(mb, oc, id, ih, iw)];
                }
            }
        }

        d[0] = math::out_round<data_t>((float)dst / num_summands);
    };

    const int MB = conf_.MB();
    const int OC = conf_.C();
    const int OD = conf_.OD();
    const int OH = conf_.OH();
    const int OW = conf_.OW();

    if (alg == pooling_max) {
#       pragma omp parallel for collapse(5) schedule(static)
        for (int mb = 0; mb < MB; ++mb) {
            for (int oc = 0; oc < OC; ++oc) {
                for (int od = 0; od < OD; ++od) {
                    for (int oh = 0; oh < OH; ++oh) {
                        for (int ow = 0; ow < OW; ++ow) {
                            data_t *d = &dst[dst_d.off(mb, oc, od, oh, ow)];
                            d[0] = nstl::numeric_limits<data_t>::lowest();
                            if (ws) {
                                ws[ws_d.off(mb, oc, od, oh, ow)] = 0;
                            }
                            ker_max(d, mb, oc, od, oh, ow);
                        }
                    }
                }
            }
        }
    } else {
#       pragma omp parallel for collapse(5) schedule(static)
        for (int mb = 0; mb < MB; ++mb) {
            for (int oc = 0; oc < OC; ++oc) {
                for (int od = 0; od < OD; ++od) {
                    for (int oh = 0; oh < OH; ++oh) {
                        for (int ow = 0; ow < OW; ++ow) {
                            data_t *d = &dst[dst_d.off(mb, oc, od, oh, ow)];
                            d[0] = 0;
                            ker_avg(d, mb, oc, od, oh, ow);
                        }
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
