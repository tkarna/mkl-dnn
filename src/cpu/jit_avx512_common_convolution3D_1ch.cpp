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
#include "utils.hpp"

#include "jit_avx512_common_convolution3D_1ch.hpp"
#include <iostream>
#include <cstring>

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;
using namespace nstl;

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
                            for (int ic = 0; ic < IC; ++ic) {
                                jitkernel(&src[src_ix.off(mb, ic, od*KSD, oh*KSH, owb*OBLOCK*KSW)],
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
                            for (int ic = 0; ic < IC; ++ic) {
                                jitkernelr(&src[src_ix.off(mb, ic, od*KSD, oh*KSH, owb*OBLOCK*KSW)],
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
struct jit_avx512_common_convolution3D_1ch_bwd_weights_t<src_type, diff_wei_type, diff_dst_type,
     acc_type>::thread_info_t {
    const src_data_t *src;
    const diff_dst_data_t *diff_dst;
    const diff_wei_data_t *diff_weights, *diff_bias;

    int ithr;
    int ithr_ow, ithr_oh, ithr_od, ithr_ic_b, ithr_oc_b, ithr_mb;

    int img_start, img_end, img_work;
    int oc_b_start, oc_b_end, oc_b_work;
    int ic_b_start, ic_b_end, ic_b_work;
    int od_start, od_end, od_work;
    int oh_start, oh_end, oh_work;
    int ow_start, ow_end, ow_work;

    thread_info_t(const jit_avx512_common_convolution3D_1ch_bwd_weights_t *self,
        int ithr): ithr(ithr) {
        src = reinterpret_cast<const src_data_t *>(self->input_memory(0));
        diff_dst = reinterpret_cast<const diff_dst_data_t *>(
            self->input_memory(1));
        diff_weights = reinterpret_cast<diff_wei_data_t*>(self->memory(0));
        diff_bias = reinterpret_cast<diff_wei_data_t *>(self->memory(1));

        ithr_ow = ithr % self->nthr_ow_;
        ithr_oh = ithr / self->nthr_ow_ % self->nthr_oh_;
        ithr_od = ithr / self->nthr_ow_ / self->nthr_oh_ % self->nthr_od_;
        ithr_ic_b = ithr / self->nthr_ow_ / self->nthr_oh_ / self->nthr_od_ % self->nthr_ic_b_;
        ithr_oc_b = ithr / self->nthr_ow_ / self->nthr_oh_ / self->nthr_od_ / self->nthr_ic_b_ % self->nthr_oc_b_;
        ithr_mb = ithr / self->nthr_ow_ / self->nthr_oh_ / self->nthr_od_ / self->nthr_ic_b_ / self->nthr_oc_b_;

        const auto &jcp = self->kernel_->jcp;

        /* reduction dimensions */
        balance211(jcp.mb, self->nthr_mb_, ithr_mb, img_start, img_end);
        img_work = img_end - img_start;

        balance211(jcp.od, self->nthr_od_, ithr_od, od_start, od_end);
        od_work = od_end - od_start;

        balance211(jcp.oh, self->nthr_oh_, ithr_oh, oh_start, oh_end);
        oh_work = oh_end - oh_start;

        balance211(jcp.ow, self->nthr_ow_, ithr_ow, ow_start, ow_end);
        ow_work = ow_end - ow_start;

        /* independent dimensions */
        balance211(jcp.nb_oc, self->nthr_oc_b_, ithr_oc_b, oc_b_start,
                oc_b_end);
        oc_b_work = oc_b_end - oc_b_start;

        balance211(jcp.nb_ic, self->nthr_ic_b_, ithr_ic_b, ic_b_start,
                ic_b_end);
        ic_b_work = ic_b_end - ic_b_start;
    }
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
    // const int MB = conf_.MB();
    const int OH = conf_.OH();
    const int OW = conf_.OW();
    const int OD = conf_.OD();
    const int IH = conf_.IH();
    const int IW = conf_.IW();
    const int ID = conf_.ID();

    const int NBLOCK = 16;
    // const int OCB = conf_.OC() / G / NBLOCK;
    // const int IC = conf_.IC() / G;

    const int KH = conf_.KH();
    const int KW = conf_.KW();
    const int KD = conf_.KD();

    // const int KSH = conf_.KSH();
    // const int KSW = conf_.KSW();
    // const int KSD = conf_.KSD();

    // const auto &jcp = kernel_->jcp;

    const int nthr_ndhw = nthr_mb_*nthr_od_*nthr_oh_*nthr_ow_;
    assert(nthr_ndhw == nthr_);

    void (*kernel)(const diff_dst_data_t*,const src_data_t*,diff_wei_data_t*,diff_wei_data_t*,struct jit_decomp*,size_t) = (void (*)(const diff_dst_data_t*,const src_data_t*,diff_wei_data_t*,diff_wei_data_t*,struct jit_decomp*,size_t)) kernel_->jit_ker;

    diff_wei_data_t private_weights[KD*KH*KW][nthr_ndhw][NBLOCK] __attribute__((aligned(64)));
    acc_data_t private_bias[nthr_ndhw][NBLOCK] __attribute__((aligned(64)));
    #pragma omp parallel num_threads(nthr_)
    {
        int ithr = omp_get_thread_num(); //, nthr = omp_get_num_threads();
        // assert(nthr_ == nthr);

        thread_info_t thread_info(this, ithr);

        // TODO: Remove this old jit_decomp struct and use thread_info_t directly
        size_t mb = thread_info.img_start;
        size_t od = thread_info.od_start;
        size_t oh = thread_info.oh_start;
        size_t ow = thread_info.ow_start;
        jit_decomp decomp = { (size_t)thread_info.img_work, (size_t)thread_info.od_work, (size_t)thread_info.oh_work, (size_t)thread_info.ow_work };
        kernel(&diff_dst[mb*OD*OH*OW*NBLOCK + od*OH*OW*NBLOCK + oh*OW*NBLOCK + ow*NBLOCK], &src[mb*ID*IH*IW + od*IH*IW + oh*IW + ow], &private_weights[0][ithr][0], &private_bias[ithr][0], &decomp, nthr_ndhw);

        #pragma omp barrier

        // TODO: Put the reduction code into a function.
        // TODO: Investigate using an MKL-DNN reducer.
#       pragma omp for
        for (int k = 0; k < KD*KH*KW; ++k)
        {
#           pragma omp simd
            for (int _oc = 0; _oc < NBLOCK; ++_oc)
            {
                acc_data_t sum = 0;
                for (int t = 0; t < nthr_ndhw; ++t)
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
                for (int t = 0; t < nthr_ndhw; ++t)
                {
                    sum += private_bias[t][_oc];
                }
                diff_bias[_oc] = sum;
            }
        }
    }
}

template <data_type_t src_type, data_type_t diff_wei_type,
         data_type_t diff_dst_type, data_type_t acc_type>
void jit_avx512_common_convolution3D_1ch_bwd_weights_t<src_type, diff_wei_type, diff_dst_type,
     acc_type>::balance() {
    const int max_threads = omp_get_max_threads();
    const auto &j = conf_.jcp_;

    nthr_ = nthr_mb_ = nthr_oc_b_ = nthr_ic_b_ = nthr_od_ = nthr_oh_ = nthr_ow_ = 1;

    auto calc_mem_cost = [=](int nthr_mb, int nthr_oc_b, int nthr_ic_b, int nthr_od, int nthr_oh, int nthr_ow)
    {
        // TODO: These may need tweaking depending on loop order.
        // const int src_coef = 1;
        // const int dst_coef = 1;
        // const int weight_coef = 1;

        return 0
            + 1 /* src */
            * div_up(j.mb, nthr_mb)
            * div_up(j.nb_ic, nthr_ic_b) * j.ic_block
            * (div_up(j.od, nthr_od) + j.kd - 1)
            * (div_up(j.oh, nthr_oh) + j.kh - 1)
            * (div_up(j.ow, nthr_ow) + j.kw - 1)
            / j.stride_d / j.stride_h / j.stride_w
            + 1 /* dst */
            * div_up(j.mb, nthr_mb)
            * div_up(j.nb_oc, nthr_oc_b) * j.oc_block
            * div_up(j.od, nthr_od)
            * div_up(j.oh, nthr_oh)
            * div_up(j.ow, nthr_ow)
            + 1 /* weights */
            * div_up(j.nb_oc, nthr_oc_b) * div_up(j.nb_ic, nthr_ic_b)
            * j.kd * j.kh * j.kw * j.ic_block * j.oc_block;
    };

    int best_mem_cost = calc_mem_cost(nthr_mb_, nthr_oc_b_, nthr_ic_b_, nthr_od_, nthr_oh_, nthr_ow_);

    /* find the best thread distribution with lowest memory cost */
    const int nthr = max_threads;
    const int nthr_mb_max = nstl::min(nthr, j.mb);
    for (int nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb)
    {
        const int nthr_oc_b_max = nstl::min(nthr / nthr_mb, j.nb_oc);
        for (int nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b)
        {
            int nthr_ic_b_max = nstl::min((nthr / nthr_mb) / nthr_oc_b, j.nb_ic);
            for (int nthr_ic_b = 1; nthr_ic_b <= nthr_ic_b_max; ++nthr_ic_b)
            {
                int nthr_od_max = nstl::min(((nthr / nthr_mb) / nthr_oc_b) / nthr_ic_b, j.od);
                for (int nthr_od = 1; nthr_od <= nthr_od_max; ++nthr_od)
                {
                    int nthr_oh_max = nstl::min((((nthr / nthr_mb) / nthr_oc_b) / nthr_ic_b) / nthr_od, j.oh);
                    for (int nthr_oh = 1; nthr_oh <= nthr_oh_max; ++nthr_oh)
                    {
                        int nthr_ow = nstl::min(((((nthr / nthr_mb) / nthr_oc_b) / nthr_ic_b) / nthr_od) / nthr_oh, j.ow);
                        int mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b, nthr_od, nthr_oh, nthr_ow);
                        if (mem_cost <= best_mem_cost)
                        {
                            best_mem_cost = mem_cost;
                            nthr_mb_ = nthr_mb;
                            nthr_oc_b_ = nthr_oc_b;
                            nthr_ic_b_ = nthr_ic_b;
                            nthr_od_ = nthr_od;
                            nthr_oh_ = nthr_oh;
                            nthr_ow_ = nthr_ow;
                        }
                    }
                }
            }
        }
    }

    if (nthr_mb_ > max_threads/2 && nthr_mb_ < max_threads)
        nthr_mb_ = nstl::min(j.mb, max_threads);

    nthr_ = nthr_mb_ * nthr_oc_b_ * nthr_ic_b_ * nthr_od_ * nthr_oh_ * nthr_ow_;
    assert(nthr_ <= max_threads);
}

using namespace data_type;

template struct _jit_avx512_common_convolution3D_1ch_fwd_t<false, f32>;
template struct _jit_avx512_common_convolution3D_1ch_fwd_t<true, f32>;
template struct jit_avx512_common_convolution3D_1ch_bwd_weights_t<f32, f32, f32, f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
