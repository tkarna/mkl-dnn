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

#include "cpu_convolution3D_1ch.hpp"
#include <iostream>

namespace mkldnn {
namespace impl {
namespace cpu {

using math::saturate;

template <bool with_relu, data_type_t src_type, data_type_t wei_type,
         data_type_t dst_type, data_type_t acc_type>
void _cpu_convolution3D_1ch_fwd_t<with_relu, src_type, wei_type, dst_type, acc_type>
        ::execute_forward() {

    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

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

    const int IH = conf_.IH();
    const int IW = conf_.IW();
    const int ID = conf_.ID();

    const int NBLOCK = 16;
    const int OC = conf_.OC() / G / NBLOCK;
    const int IC = conf_.IC() / G;

    const int KH = conf_.KH();
    const int KW = conf_.KW();
    const int KD = conf_.KD();

    const int KSH = conf_.KSH();
    const int KSW = conf_.KSW();
    const int KSD = conf_.KSD();

    const int KDH = conf_.KDH();
    const int KDW = conf_.KDW();
    const int KDD = conf_.KDD();

    const int padT = conf_.padT();
    const int padL = conf_.padL();
    const int padD1 = conf_.padD1();

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

    assert(G == 0); // NOTE groups not implemented
    assert(KDH == 0 && KDW == 0 && KDD == 0); // NOTE dilation not implemented
    assert(padT == 0 && padL == 0 && padD1 == 0); // NOTE pad not implemented

#   pragma omp parallel for collapse(6) schedule(static)
    for (int g = 0; g < G; ++g) {
        for (int mb = 0; mb < MB; ++mb) {
            for (int oc = 0; oc < OC; ++oc) {
                for (int od = 0; od < OD; ++od) {
                    for (int oh = 0; oh < OH; ++oh) {
                        for (int ow = 0; ow < OW; ++ow) {
                            // HACK assume no bias for now
                            acc_data_t a[NBLOCK] = {0};
                            if (bias) {
#                               pragma omp simd
                                for (int ocb = 0; ocb < NBLOCK; ++ocb) {
                                    a[ocb] = get_bias(bias_d.off((int)(g*oc + ocb)));
                                }
                            }
                            for (int ic = 0; ic < IC; ++ic) {
                                for (int kd = 0; kd < KD; ++kd) {
                                    for (int kh = 0; kh < KH; ++kh) {
                                        const int id = od * KSD - padD1 + kd * (1 + KDD);
                                        const int ih = oh * KSH - padT  + kh * (1 + KDH);
                                        const int iw = ow * KSW - padL;
                                        // NOTE skip bounds checking for now
                                        // not needed if no padding/dilation

                                        const size_t src_ix = ((((mb*IC + ic)*ID + id)*IH + ih)*IW + iw);
                                        const size_t w_ix = ((((oc*IC + ic)*KD +kd)*KH + kh)*KW + 0)*NBLOCK;
                                        for (int kw = 0; kw < KW; ++kw) {
                                            // NOTE assume no groups for now
#                                           pragma omp simd
                                            for (int ocb = 0; ocb < NBLOCK; ++ocb) {
                                                a[ocb] += src[src_ix + (1 + KDW)*kw] * weights[w_ix + NBLOCK*kw + ocb];
                                            }
                                        }
                                    }
                                }
                            }
                            const size_t dst_ix = ((((mb*OC + oc)*OD + od)*OH + oh)*OW + ow)*NBLOCK;
#                           pragma omp simd
                            for (int ocb = 0; ocb < NBLOCK; ++ocb) {
                                if (with_relu && a[ocb] < (acc_data_t)0)
                                    a[ocb] = (acc_data_t)((float)a[ocb] * nslope);
                                dst[dst_ix + ocb] = saturate<dst_data_t>(a[ocb]);
                            }
                        }
                    }
                }
            }
        }
    }
}

template <data_type_t diff_src_type, data_type_t wei_type,
         data_type_t diff_dst_type, data_type_t acc_type>
void cpu_convolution3D_1ch_bwd_data_t<diff_src_type, wei_type, diff_dst_type,
     acc_type>::execute_backward_data() {
    auto diff_dst = reinterpret_cast<const diff_dst_data_t*>(
            this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t*>(this->input_memory(1));
    auto diff_src = reinterpret_cast<diff_src_data_t*>(this->memory());

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_src_d(conf_.diff_src_pd());
    const memory_desc_wrapper weights_d(conf_.weights_pd(0));

    // const bool with_groups = conf_.with_groups();

    const int G = conf_.G();
    const int MB = conf_.MB();
    const int OH = conf_.OH();
    const int OW = conf_.OW();
    const int OD = conf_.OD();
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

    const int KDH = conf_.KDH();
    const int KDW = conf_.KDW();
    const int KDD = conf_.KDD();

    const int padT = conf_.padT();
    const int padL = conf_.padL();
    const int padD1 = conf_.padD1();

#   pragma omp parallel for collapse(6) schedule(static)
    for (int g = 0; g < G; ++g) {
        for (int mb = 0; mb < MB; ++mb) {
            for (int ic = 0; ic < IC; ++ic) {
                for (int id = 0; id < ID; ++id) {
                    for (int ih = 0; ih < IH; ++ih) {
                        for (int iw = 0; iw < IW; ++iw) {
                            acc_data_t a = (acc_data_t)0;
                            for (int ocb = 0; ocb < OCB; ++ocb) {
                                for (int kd = 0; kd < KD; ++kd) {
                                    for (int kh = 0; kh < KH; ++kh) {
                                        for (int kw = 0; kw < KW; ++kw) {
                                            if (iw + padL < kw * (1 + KDW)
                                                || ih + padT < kh * (1 + KDH)
                                                || id + padD1 < kd * (1 + KDD))
                                                continue;
                                            int od = id - kd * (1 + KDD) + padD1;
                                            int oh = ih - kh * (1 + KDH) + padT;
                                            int ow = iw - kw * (1 + KDW) + padL;
                                            if (ow % KSW != 0 || oh % KSH != 0 || od % KSD != 0 ||
                                                ow >= OW || oh >= OH || od >= OD)
                                                continue;

                                            od /= KSD;
                                            oh /= KSH;
                                            ow /= KSW;

                                            auto dst_ix = diff_dst_d.off(mb, (g*OCB + ocb)*NBLOCK, od, oh, ow);
                                            auto w_ix = weights_d.off(ocb*NBLOCK, ic, kd, kh, kw);
#                                           pragma omp simd
                                            for (int _oc = 0; _oc < NBLOCK; ++_oc) {
                                                a += (acc_data_t)diff_dst[dst_ix + _oc] * weights[w_ix + _oc];
                                            }
                                        }
                                    }
                                }
                            }
                            auto ds_idx = diff_src_d.off(mb, g*IC + ic, id, ih, iw);
                            diff_src[ds_idx] = saturate<diff_src_data_t>(a);
                        }
                    }
                }
            }
        }
    }
}

template <data_type_t src_type, data_type_t diff_wei_type,
         data_type_t diff_dst_type, data_type_t acc_type>
void cpu_convolution3D_1ch_bwd_weights_t<src_type, diff_wei_type, diff_dst_type,
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

    // const bool with_groups = conf_.with_groups();

    const int G = conf_.G();
    const int MB = conf_.MB();
    const int OH = conf_.OH();
    const int OW = conf_.OW();
    const int OD = conf_.OD();
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

    const int KDH = conf_.KDH();
    const int KDW = conf_.KDW();
    const int KDD = conf_.KDD();

    const int padT = conf_.padT();
    const int padL = conf_.padL();
    const int padD1 = conf_.padD1();

    if (diff_bias) {
#       pragma omp parallel for collapse(2) schedule(static)
        for (int g = 0; g < G; ++g) {
            for (int ocb = 0; ocb < OCB; ++ocb) {
                acc_data_t db[NBLOCK] = {0};
                for (int mb = 0; mb < MB; ++mb) {
                    for (int od = 0; od < OD; ++od) {
                        for (int oh = 0; oh < OH; ++oh) {
                            for (int ow = 0; ow < OW; ++ow) {
                                auto dst_ix = diff_dst_d.off(mb, g*OCB*NBLOCK + ocb*NBLOCK, od, oh, ow);
#                               pragma omp simd
                                for (int _oc=0; _oc < NBLOCK; ++_oc) {
                                    db[_oc] += (acc_data_t)diff_dst[dst_ix + _oc];
                                }
                            }
                        }
                    }
                }
                auto bias_ix = diff_bias_d.off(g*OCB*NBLOCK + ocb*NBLOCK);
#               pragma omp simd
                for (int _oc=0; _oc < NBLOCK; ++_oc) {
                    diff_bias[bias_ix + _oc] = saturate<diff_wei_data_t>(db[_oc]);
                }
            }
        }
    }

#   pragma omp parallel for collapse(6) schedule(static)
    for (int g = 0; g < G; ++g) {
        for (int ocb = 0; ocb < OCB; ++ocb) {
            for (int ic = 0; ic < IC; ++ic) {
                for (int kd = 0; kd < KD; ++kd) {
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            acc_data_t dw[NBLOCK] = {0};
                            for (int mb = 0; mb < MB; ++mb) {
                                for (int od = 0; od < OD; ++od) {
                                    for (int oh = 0; oh < OH; ++oh) {
                                        for (int ow = 0; ow < OW; ++ow) {
                                            // NOTE this if statement is nasty!
                                            if (ow*KSW + kw * (1 + KDW) < padL
                                                    || oh*KSH + kh * (1 + KDH) < padT
                                                    || od*KSD + kd * (1 + KDD) < padD1
                                                    || ow*KSW + kw * (1 + KDW) >= IW + padL
                                                    || oh*KSH + kh * (1 + KDH) >= IH + padT
                                                    || od*KSD + kd * (1 + KDD) >= ID + padD1)
                                                continue;

                                            int id = od*KSD - padD1 + kd * (1 + KDD);
                                            int ih = oh*KSH - padT + kh * (1 + KDH);
                                            int iw = ow*KSW - padL + kw * (1 + KDW);

                                            auto dst_ix = diff_dst_d.off(mb, g*OCB*NBLOCK + ocb*NBLOCK, od, oh, ow);
                                            auto src_ix = src_d.off(mb, g*IC + ic, id, ih, iw);
#                                           pragma omp simd
                                            for (int _oc=0; _oc < NBLOCK; ++_oc) {
                                                dw[_oc] += (acc_data_t)diff_dst[dst_ix + _oc] * src[src_ix];
                                            }
                                        }
                                    }
                                }
                            }
                            auto idx = diff_weights_d.off(ocb*NBLOCK, ic, kd, kh, kw);
#                           pragma omp simd
                            for (int _oc=0; _oc < NBLOCK; ++_oc) {
                                diff_weights[idx + _oc] = saturate<diff_wei_data_t>(dw[_oc]);
                            }
                        }
                    }
                }
            }
        }
    }
}

using namespace data_type;

template struct _cpu_convolution3D_1ch_fwd_t<false, f32>;
template struct _cpu_convolution3D_1ch_fwd_t<true, f32>;
template struct _cpu_convolution3D_1ch_fwd_t<false, s16, s16, s32, s32>;
template struct _cpu_convolution3D_1ch_fwd_t<true, s16, s16, s32, s32>;
template struct _cpu_convolution3D_1ch_fwd_t<false, u8, s8, s32, s32>;
template struct _cpu_convolution3D_1ch_fwd_t<true, u8, s8, s32, s32>;
template struct _cpu_convolution3D_1ch_fwd_t<false, u8, s8, s8, s32>;
template struct _cpu_convolution3D_1ch_fwd_t<true, u8, s8, s8, s32>;
template struct _cpu_convolution3D_1ch_fwd_t<false, u8, s8, u8, s32>;
template struct _cpu_convolution3D_1ch_fwd_t<true, u8, s8, u8, s32>;

template struct cpu_convolution3D_1ch_bwd_data_t<f32, f32, f32, f32>;
template struct cpu_convolution3D_1ch_bwd_data_t<s32, s16, s16, s32>;

template struct cpu_convolution3D_1ch_bwd_weights_t<f32, f32, f32, f32>;
template struct cpu_convolution3D_1ch_bwd_weights_t<s16, s32, s16, s32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
