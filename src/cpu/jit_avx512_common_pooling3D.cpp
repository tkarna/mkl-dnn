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

#include <limits>
#include <cmath>

#include "jit_avx512_common_pooling3D.hpp"

struct loop_bounds
{
    size_t od_max;
    size_t oh_max;
    size_t ow_max;
};

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

static std::vector<uint32_t> get_factors(uint32_t num)
{
    std::vector<uint32_t> res;
    while(num > 1)
    {
        if((num % 2) == 0)
        {
            num /= 2;
            res.push_back(2);
        }
        else
        {
            const uint32_t limit = std::sqrt(num);
            bool divisor = false;
            for(uint32_t i = 3; i <= limit; i+=2)
            {
                if((num % i) == 0)
                {
                    num /= i;
                    res.push_back(i);
                    divisor = true;
                    break;
                }
            }
            if(!divisor)
            {
                res.push_back(num);
                num = 1;
            }
        }
    }
    return res;
}

// Compute products of a subset in log space
static double nth_subset_sum(uint64_t num, const std::vector<double> *set)
{
    double res = 0.0;
    for(uint32_t i = 0; i < set->size(); ++i)
    {
        if(num & (1ULL << i))
        {
            res += (*set)[i];
        }
    }
    return res;
}

// Compute products of a subset in 'normal' space.
static uint32_t nth_subset_prod(uint64_t num, const std::vector<uint32_t> *set)
{
    uint32_t res = 1;
    for(uint32_t i = 0; i < set->size(); ++i)
    {
        if(num & (1ULL << i))
        {
            res *= (*set)[i];
        }
    }
    return res;
}

static uint64_t log_best_subset(const double split, const std::vector<double> *ln_factors)
{
    double best = std::numeric_limits<double>::max();
    uint64_t argbest = -1;
    for( int i = 1; i < (1 << ln_factors->size()); ++i)
    {
        const double prod = std::abs(nth_subset_sum(i, ln_factors) - split);
        if(prod < best)
        {
            best = prod;
            argbest = i;
        }
    }
    return argbest;
}

static double log_weight0(uint32_t N, const uint32_t *weights, double scale)
{
    double res = std::log(scale);
    for(uint32_t j = 1; j < N; ++j)
    {
        res -= std::log(weights[j]);
    }
    res += (N-1)*std::log(weights[0]);
    res /= N;
    return res;
}

// Discard items in set & compact list
template <typename T>
static void compress_set(uint64_t num, std::vector<T> *set)
{
    uint32_t fillp = 0;
    for(uint32_t i = 0; i < set->size(); ++i)
    {
        if(num & (1ULL << i))
        {
            (*set)[fillp] = (*set)[i];
            ++fillp;
        }
    }
}

std::vector<uint32_t> best_decomp(uint32_t num, const std::vector<uint32_t> &weights)
{
    std::vector<uint32_t> factors = get_factors(num);
    assert(factors.size() < 64);

    std::vector<double> ln_factors(factors.size());
    for(uint32_t j = 0; j < ln_factors.size(); ++j)
    {
        ln_factors[j] = std::log(factors[j]);
    }

    const uint32_t groups = weights.size();
    std::vector<uint32_t> res(groups);

    uint32_t todiv = num;
    for(uint32_t i = 0; i < groups-1; ++i)
    {
        const double split = log_weight0(groups-i, &(weights[i]), todiv);

        // Splits very close to zero should result in a factor of 1 being used.
        if(split <= 1e-6)
        {
            res[i] = 1;
            continue;
        }

        const uint64_t set_idx = log_best_subset(split, &ln_factors);
        res[i] = nth_subset_prod(set_idx, &factors);
        // Discard factors that we have used.
        compress_set(set_idx, &factors);
        compress_set(set_idx, &ln_factors);
        todiv /= res[i];
    }
    // Since we need a total partition, the last multiplicand is already found.
    res[groups-1] = todiv;
    return res;
}

static void divvy(int *start, int *end, const int nitems, int chunkno, int nchunks) {
    const int items_per_chunk = nitems / nchunks;
    const int remainder       = nitems - nchunks * items_per_chunk;

    *start = chunkno * items_per_chunk + std::min(chunkno, remainder);
    *end   = (chunkno + 1) * items_per_chunk + std::min(chunkno + 1, remainder);
}

void multi_decomp(int *start_ends, int rank, int nranks, int ndims, int *dims, uint32_t *decomp)
{
    int eff_rank = rank;
    int eff_nranks = nranks;

    for(int d = 0; d < ndims; ++d)
    {
        divvy(start_ends + 2*d + 0, start_ends + 2*d + 1, dims[d], decomp[d] * eff_rank / eff_nranks, decomp[d]);
        eff_nranks /= decomp[d];
        eff_rank = eff_rank % eff_nranks;
    }

}

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t data_type, data_type_t acc_type>
void jit_avx512_common_pooling3D_fwd_t<data_type, acc_type>::execute_forward() {
    using namespace alg_kind;
    using namespace prop_kind;

    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t *>(this->memory(0));
    // auto ws = alg == pooling_max && conf_.desc()->prop_kind == forward_training
    //     ? reinterpret_cast<unsigned char *>(this->memory(1)) : nullptr;

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    // const memory_desc_wrapper ws_d(conf_.workspace_pd());
    // const data_type_t ws_dt = ws ? ws_d.data_type() : data_type::undef;

    const int MB = conf_.MB();
    const int NBLOCK = 16;
    const int OCB = conf_.C() / NBLOCK;

    const int OD = conf_.OD();
    const int OH = conf_.OH();
    const int OW = conf_.OW();

    const int IH = conf_.IH();
    const int IW = conf_.IW();
    const int ID = conf_.ID();

    const int SH = conf_.KSH();
    const int SW = conf_.KSW();
    const int SD = conf_.KSD();

    auto src_ix = MultiviewOffset(MB, OCB, ID, IH, IW);
    auto dst_ix = MultiviewOffset(MB, OCB, OD, OH, OW);

    const int nthreads = omp_get_max_threads();

    void (*kernel)(const data_t*, const data_t*) = (void (*)(const data_t*, const data_t*)) kernel_->jit_ker;

    std::vector<uint32_t> decomp(best_decomp(nthreads, std::vector<uint32_t>(3,1)));
    while(decomp.size() < 5) {
        decomp.insert(decomp.begin(), 1);
    }

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        int start_ends[2*5];
        std::vector<int> dims = {MB, OCB, OD, OH, OW};
        multi_decomp(start_ends, tid, nthreads, 5, &dims[0], &decomp[0]);

        for (int mb = start_ends[2*0+0]; mb < start_ends[2*0+1]; ++mb) {
            for (int ocb = start_ends[2*1+0]; ocb < start_ends[2*1+1]; ++ocb) {
                for (int od = start_ends[2*2+0]; od < start_ends[2*2+1]; ++od) {
                    for (int oh = start_ends[2*3+0]; oh < start_ends[2*3+1]; ++oh) {
                        for (int ow = start_ends[2*4+0]; ow < start_ends[2*4+1]; ++ow) {
                            data_t *dst_vec = (data_t *)&dst[dst_ix.off(mb, ocb, od, oh, ow)*NBLOCK];
                            const data_t *src_vec = (data_t *)&src[src_ix.off(mb, ocb, od*SD, oh*SH, ow*SW)*NBLOCK];
                            kernel(dst_vec, src_vec);
                        }
                    }
                }
            }
        }
    }

}

template <data_type_t data_type, data_type_t acc_type>
void jit_avx512_common_pooling3D_bwd_t<data_type, acc_type>::execute_backward() {
    using namespace alg_kind;

    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto ws = conf_.desc()->alg_kind != alg_kind::pooling_max ? nullptr
        : reinterpret_cast<const unsigned char *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t *>(this->memory(0));

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper ws_d(conf_.workspace_pd());
    const memory_desc_wrapper diff_src_d(conf_.diff_src_pd());

    const int MB = conf_.MB();
    const int NBLOCK = 16;
    const int OCB = conf_.C() / NBLOCK;

    const int OD = conf_.OD();
    const int OH = conf_.OH();
    const int OW = conf_.OW();

    const int ID = conf_.ID();
    const int IH = conf_.IH();
    const int IW = conf_.IW();

    const int KD = conf_.KD();
    const int KH = conf_.KH();
    const int KW = conf_.KW();

    const int SD = conf_.KSD();
    const int SH = conf_.KSH();
    const int SW = conf_.KSW();

    // largest used input pixel (stride round-off)
    const int ID_MAX = (OD - 1)*SD + KD;
    const int IH_MAX = (OH - 1)*SH + KH;
    const int IW_MAX = (OW - 1)*SW + KW;

    auto src_ix = MultiviewOffset(MB, OCB, ID, IH, IW);
    auto dst_ix = MultiviewOffset(MB, OCB, OD, OH, OW);

    typedef void (*jitkernel_t)(const data_t*, const data_t*, struct loop_bounds*);
    jitkernel_t kernel = (jitkernel_t) kernel_->jit_ker;

    auto alg = conf_.desc()->alg_kind;

    if (alg == alg_kind::pooling_max) {
#       pragma omp parallel for collapse(2) schedule(static)
        for (int mb = 0; mb < MB; ++mb) {
            for (int ocb = 0; ocb < OCB; ++ocb) {
                for (int id = 0; id < ID_MAX; ++id) {
                    for (int ih = 0; ih < IH_MAX; ++ih) {
                        for (int iw = 0; iw < IW_MAX; ++iw) {
                            for (int _oc = 0; _oc < NBLOCK; ++_oc) {
                                diff_src[src_ix.off(mb, ocb, id, ih, iw)*NBLOCK + _oc] = data_type_t(0);
                            }
                        }
                    }
                }
                for (int od = 0; od < OD; ++od) {
                    for (int oh = 0; oh < OH; ++oh) {
                        for (int ow = 0; ow < OW; ++ow) {
                            const data_t *diff_dst_vec = &diff_dst[dst_ix.off(mb, ocb, od, oh, ow)*NBLOCK];

                            for (int _oc = 0; _oc < NBLOCK; ++_oc) {
                                const int oc = ocb*NBLOCK + _oc;
                                const size_t ws_off = ws_d.off(mb, oc, od, oh, ow);
                                const int index = ws_d.data_type() == data_type::u8
                                    ? (int)ws[ws_off] : ((int *)ws)[ws_off];
                                const int kd = index / (KH*KW);
                                const int kw = (index % (KH*KW)) % KW;
                                const int kh = (index % (KH*KW)) / KW;
                                const int id = od * SD + kd;
                                const int ih = oh * SH + kh;
                                const int iw = ow * SW + kw;
                                diff_src[src_ix.off(mb, ocb, id, ih, iw)*NBLOCK + _oc] += diff_dst_vec[_oc];
                            }
                        }
                    }
                }
            }
        }
    } else {
        const int nthreads = omp_get_max_threads();

        std::vector<uint32_t> decomp(best_decomp(nthreads, std::vector<uint32_t>(3,1)));
        while(decomp.size() < 5) {
            decomp.insert(decomp.begin(), 1);
        }

        if (KD == SD && KH == SH && KW == SW) {
            // special case where kernel == stride
            // no need for bounds checking
            // each output pixel maps to exactly one input pixel
#           pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                int start_ends[2*5];
                std::vector<int> dims = {MB, OCB, OD, OH, OW};
                multi_decomp(start_ends, tid, nthreads, 5, &dims[0], &decomp[0]);

                data_t inv_summands = 1.0/(KD*KH*KW);
                for (int mb = start_ends[2*0+0]; mb < start_ends[2*0+1]; ++mb) {
                    for (int ocb = start_ends[2*1+0]; ocb < start_ends[2*1+1]; ++ocb) {
                        for (int od = start_ends[2*2+0]; od < start_ends[2*2+1]; ++od) {
                            for (int oh = start_ends[2*3+0]; oh < start_ends[2*3+1]; ++oh) {
                                for (int ow = start_ends[2*4+0]; ow < start_ends[2*4+1]; ++ow) {
                                    const data_t *diff_dst_vec = (data_t *)&diff_dst[dst_ix.off(mb, ocb, od, oh, ow)*NBLOCK];
                                    data_t *diff_src_vec = (data_t *)&diff_src[src_ix.off(mb, ocb, od*SD, oh*SH, ow*SW)*NBLOCK];
                                    for (int kd = 0; kd < KD; ++kd) {
                                        for (int kh = 0; kh < KH; ++kh) {
                                            for (int kw = 0; kw < KW; ++kw) {
        #                                       pragma omp simd
                                                for(size_t oc = 0; oc < NBLOCK; ++oc) {
                                                    diff_src_vec[kd*IH*IW*NBLOCK + kh*IW*NBLOCK + kw*NBLOCK + oc] = inv_summands*diff_dst_vec[oc];
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
        } else {
#           pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                int start_ends[2*5];
                std::vector<int> dims = {MB, OCB, ID_MAX, IH_MAX, IW_MAX};
                multi_decomp(start_ends, tid, nthreads, 5, &dims[0], &decomp[0]);

                // precompute loop bounds
                const size_t LOC_ID_START = start_ends[2*2+0];
                const size_t LOC_IH_START = start_ends[2*3+0];
                const size_t LOC_IW_START = start_ends[2*4+0];
                const size_t LOC_ID_LEN = start_ends[2*2+1] - LOC_ID_START;
                const size_t LOC_IH_LEN = start_ends[2*3+1] - LOC_IH_START;
                const size_t LOC_IW_LEN = start_ends[2*4+1] - LOC_IW_START;
                std::vector<std::pair<size_t, size_t> > od_lims(LOC_ID_LEN);
                std::vector<std::pair<size_t, size_t> > oh_lims(LOC_IH_LEN);
                std::vector<std::pair<size_t, size_t> > ow_lims(LOC_IW_LEN);
                for (size_t i = 0; i < LOC_ID_LEN; ++i) {
                    const int id = i + LOC_ID_START;
                    const int od_start = nstl::max((id - KD + SD)/SD, 0);
                    const int od_end = nstl::min(id/SD + 1, OD);
                    od_lims[i] = std::make_pair((size_t)od_start, (size_t)(od_end - od_start));
                }
                for (size_t i = 0; i < LOC_IH_LEN; ++i) {
                    const int ih = i + LOC_IH_START;
                    const int oh_start = nstl::max((ih - KH + SH)/SH, 0);
                    const int oh_end = nstl::min(ih/SH + 1, OH);
                    oh_lims[i] = std::make_pair((size_t)oh_start, (size_t)(oh_end - oh_start));
                }
                for (size_t i = 0; i < LOC_IW_LEN; ++i) {
                    const int iw = i + LOC_IW_START;
                    const int ow_start = nstl::max((iw - KW + SW)/SW, 0);
                    const int ow_end = nstl::min(iw/SW + 1, OW);
                    ow_lims[i] = std::make_pair((size_t)ow_start, (size_t)(ow_end - ow_start));
                }

                for (int mb = start_ends[2*0+0]; mb < start_ends[2*0+1]; ++mb) {
                    for (int ocb = start_ends[2*1+0]; ocb < start_ends[2*1+1]; ++ocb) {
                        for (int id = LOC_ID_START; id < start_ends[2*2+1]; ++id) {
                            for (int ih = LOC_IH_START; ih < start_ends[2*3+1]; ++ih) {
                                for (int iw = LOC_IW_START; iw < start_ends[2*4+1]; ++iw) {
                                    const size_t od_start = od_lims[id - LOC_ID_START].first;
                                    const size_t od_len = od_lims[id - LOC_ID_START].second;
                                    const size_t oh_start = oh_lims[ih - LOC_IH_START].first;
                                    const size_t oh_len = oh_lims[ih - LOC_IH_START].second;
                                    const size_t ow_start = ow_lims[iw - LOC_IW_START].first;
                                    const size_t ow_len = ow_lims[iw - LOC_IW_START].second;
                                    const data_t *diff_dst_vec = (data_t *)&diff_dst[dst_ix.off(mb, ocb, od_start, oh_start, ow_start)*NBLOCK];
                                    data_t *diff_src_vec = (data_t *)&diff_src[src_ix.off(mb, ocb, id, ih, iw)*NBLOCK];
                                    loop_bounds lbounds = {od_len, oh_len, ow_len};
                                    kernel(diff_src_vec, diff_dst_vec, &lbounds);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template struct jit_avx512_common_pooling3D_fwd_t<data_type::f32>;

template struct jit_avx512_common_pooling3D_bwd_t<data_type::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
