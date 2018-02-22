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

namespace mkldnn {
namespace impl {
namespace cpu {

using math::saturate;

template <bool with_relu, data_type_t src_type, data_type_t wei_type,
         data_type_t dst_type, data_type_t acc_type>
void _cpu_convolution3D_1ch_fwd_t<with_relu, src_type, wei_type, dst_type, acc_type>
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

    const int OBLOCK = 14;
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

    // const int KDH = conf_.KDH();
    // const int KDW = conf_.KDW();
    // const int KDD = conf_.KDD();

    // const int padT = conf_.padT();
    // const int padL = conf_.padL();
    // const int padD1 = conf_.padD1();

    auto src = reinterpret_cast<const src_data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const wei_data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const char *>(this->input_memory(2));
    auto dst = reinterpret_cast<dst_data_t *>(this->memory());

    auto src_ix_ = MultiviewOffset(MB, IC, ID, IH, IW);
    auto dst_ix_ = MultiviewOffset(MB, OCB, OD, OH, OW);
    auto w_ix_ = MultiviewOffset(OCB, IC, KD, KH, KW);

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

#   pragma omp parallel for collapse(5) schedule(static)
    for (int g = 0; g < G; ++g) {
        for (int mb = 0; mb < MB; ++mb) {
            for (int ocb = 0; ocb < OCB; ++ocb) {
                for (int od = 0; od < OD; ++od) {
                    for (int oh = 0; oh < OH; ++oh) {
                        // case 1: full OBLOCKs
                        for (int owb = 0; owb < OWB - 1 + (OWREM==0); ++owb) {
                            acc_data_t a[OBLOCK][NBLOCK];
                            for (int _ow = 0; _ow < OBLOCK; ++_ow) {
#                               pragma omp simd
                                for (int _oc = 0; _oc < NBLOCK; ++_oc) {
                                    a[_ow][_oc] = (bias) ? get_bias((g*OCB + ocb)*NBLOCK + _oc) : (acc_data_t)0;
                                }
                            }
                            for (int ic = 0; ic < IC; ++ic) {
                                for (int kd = 0; kd < KD; ++kd) {
                                    for (int kh = 0; kh < KH; ++kh) {
                                        for (int kw = 0; kw < KW; ++kw) {
#                                           pragma omp simd
                                            for (int _oc = 0; _oc < NBLOCK; ++_oc) {
#                                               pragma unroll (OBLOCK)
                                                for (int _ow = 0; _ow < OBLOCK; ++_ow) {
                                                    a[_ow][_oc] += src[src_ix_.off(mb, ic, od * KSD + kd,
                                                                                   oh * KSH + kh, (owb*OBLOCK + _ow) * KSW + kw)] *
                                                        weights[w_ix_.off(ocb, ic, kd, kh, kw)*NBLOCK + _oc];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            for (int _ow = 0; _ow < OBLOCK; ++_ow) {
#                               pragma omp simd
                                for (int _oc = 0; _oc < NBLOCK; ++_oc) {
                                    if (with_relu && a[_ow][_oc] < (acc_data_t)0)
                                        a[_ow][_oc] = (acc_data_t)((float)a[_ow][_oc] * nslope);
                                    dst[dst_ix_.off(mb, ocb, od, oh, owb*OBLOCK + _ow)*NBLOCK + _oc] = saturate<dst_data_t>(a[_ow][_oc]);
                                }
                            }
                        }
                        // case 2: remainder
                        if (OWREM > 0) {
                            for (int owb = OWB - 1; owb < OWB; ++owb) {
                                acc_data_t a[OBLOCK][NBLOCK];
                                for (int _ow = 0; _ow < OBLOCK; ++_ow) {
#                                   pragma omp simd
                                    for (int _oc = 0; _oc < NBLOCK; ++_oc) {
                                        a[_ow][_oc] = (bias) ? get_bias((g*OCB + ocb)*NBLOCK + _oc) : (acc_data_t)0;
                                    }
                                }
                                for (int ic = 0; ic < IC; ++ic) {
                                    for (int kd = 0; kd < KD; ++kd) {
                                        for (int kh = 0; kh < KH; ++kh) {
                                            for (int kw = 0; kw < KW; ++kw) {
#                                               pragma omp simd
                                                for (int _oc = 0; _oc < NBLOCK; ++_oc) {
#                                                   pragma unroll
                                                    for (int _ow = 0; _ow < OWREM; ++_ow) {
                                                        a[_ow][_oc] += src[src_ix_.off(mb, ic, od * KSD + kd,
                                                                                       oh * KSH + kh, (owb*OBLOCK + _ow) * KSW + kw)] *
                                                            weights[w_ix_.off(ocb, ic, kd, kh, kw)*NBLOCK + _oc];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                for (int _ow = 0; _ow < OWREM; ++_ow) {
#                                   pragma omp simd
                                    for (int _oc = 0; _oc < NBLOCK; ++_oc) {
                                        if (with_relu && a[_ow][_oc] < (acc_data_t)0)
                                            a[_ow][_oc] = (acc_data_t)((float)a[_ow][_oc] * nslope);
                                        dst[dst_ix_.off(mb, ocb, od, oh, owb*OBLOCK + _ow)*NBLOCK + _oc] = saturate<dst_data_t>(a[_ow][_oc]);
                                    }
                                }
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
