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
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "cpu_memory.hpp"

#include "jit_avx512_common_conv3D_1ch_kernel.hpp"
#include <omp.h>

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

void jit_avx512_common_conv3D_1ch_bwd_weights_kernel_f32::generate()
{
    preamble();

    // Base pointers.
    const reg64_t rdst = rdi;
    const reg64_t rsrc = rsi;
    const reg64_t rweights = rdx;
    const reg64_t rbias = rcx;

    // Decomposition.
    reg64_t rdecomp = r8;
    const int MB_OFFSET = 0;
    const int ODB_OFFSET = MB_OFFSET + sizeof(uint64_t);
    const int OHB_OFFSET = ODB_OFFSET + sizeof(uint64_t);
    const int OWB_OFFSET = OHB_OFFSET + sizeof(uint64_t);

    // Loop counters.
    reg64_t rMB = r9;
    reg64_t rODB = r10;
    reg64_t rOHB = r11;
    reg64_t rOWB = r12;

    const int KT = jcp.kd * jcp.kh * jcp.kw;

    // Zero the accumulators.
    Zmm accum[KT];
    for (int k = 0; k < KT; ++k)
    {
        accum[k] = Zmm(k);
        vpxord(accum[k], accum[k], accum[k]);
    }
    Zmm dst = Zmm(KT);

    // Need an additional accumulator if computing the bias.
    Zmm bias = Zmm(KT + 1);
    if (jcp.with_bias)
    {
        vpxord(bias, bias, bias);
    }

    Label mb_loop, od_loop, oh_loop, ow_loop;
    mov(rMB, ptr[rdecomp + MB_OFFSET]);
    L(mb_loop);
    {
        mov(rODB, ptr[rdecomp + ODB_OFFSET]);
        L(od_loop);
        {
            mov(rOHB, ptr[rdecomp + OHB_OFFSET]);
            L(oh_loop);
            {
                mov(rOWB, ptr[rdecomp + OWB_OFFSET]);
                L(ow_loop);
                {
                    // accum[k][oc] += dst[mb][od][oh][ow][oc] * src[mb][ic][id][ih][iw];
                    // Re-use vector of dst values across several src values.
                    vmovups(dst, ptr[rdst]);
                    prefetcht1(ptr[rdst + jcp.ow*64]);
                    for (int kd = 0; kd < jcp.kd; ++kd)
                    {
                        for (int kh = 0; kh < jcp.kh; ++kh)
                        {
                            for (int kw = 0; kw < jcp.kw; ++kw)
                            {
                                int k = kd*jcp.kh*jcp.kw + kh*jcp.kw + kw;
                                vfmadd231ps(accum[k], dst, ptr_b[rsrc + (kd*jcp.ih*jcp.iw + kh*jcp.iw + kw)*sizeof(float)]); // TODO: Use smarter addressing.
                            }
                        }
                    }
                    if (jcp.with_bias)
                    {
                        vaddps(bias, bias, dst);
                    }
                    add(rdst, 64);
                    add(rsrc, sizeof(float)); // TODO: Support other strides.
                    sub(rOWB, 1);
                    jnz(ow_loop);
                }
                imul(rOWB, ptr[rdecomp + OWB_OFFSET], 64); // TODO: Can we do this with fewer instructions?
                sub(rdst, rOWB);
                add(rdst, jcp.ow*64);
                imul(rOWB, ptr[rdecomp + OWB_OFFSET], sizeof(float)); // TODO: Can we do this with fewer instructions?
                sub(rsrc, rOWB);
                add(rsrc, jcp.iw*sizeof(float)); // TODO: Support other strides.
                sub(rOHB, 1);
                jnz(oh_loop);
            }
            imul(rOHB, ptr[rdecomp + OHB_OFFSET], jcp.ow*64); // TODO: Can we do this with fewer instructions?
            sub(rdst, rOHB);
            add(rdst, jcp.oh*jcp.ow*64);
            imul(rOHB, ptr[rdecomp + OHB_OFFSET], jcp.iw*sizeof(float)); // TODO: Can we do this with fewer instructions?
            sub(rsrc, rOHB);
            add(rsrc, jcp.ih*jcp.iw*sizeof(float)); // TODO: Support other strides.
            sub(rODB, 1);
            jnz(od_loop);
        }
        sub(rMB, 1);
        jnz(mb_loop);
    }

    // TODO: Saturate.

    // Write out the final results.
    // TODO: A little scared about calling omp_get_max_threads() here.
    for (int k = 0; k < KT; ++k)
    {
        vmovntps(ptr[rweights + k*omp_get_max_threads()*64], accum[k]);
    }
    if (jcp.with_bias)
    {
        vmovntps(ptr[rbias], bias);
    }

    postamble();
}

status_t jit_avx512_common_conv3D_1ch_bwd_weights_kernel_f32::init_conf(
        jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        cpu_memory_t::pd_t &src_pd, cpu_memory_t::pd_t &diff_weights_pd,
        cpu_memory_t::pd_t &diff_bias_pd, cpu_memory_t::pd_t &diff_dst_pd)
{
    if (!mayiuse(avx512_common))
        return status::unimplemented;

    const int simd_w = cpu_isa_traits<avx512_common>::vlen / sizeof(float);

    const memory_desc_wrapper src_d(&src_pd);
    const memory_desc_wrapper diff_weights_d(&diff_weights_pd);
    const memory_desc_wrapper diff_bias_d(&diff_bias_pd);
    const memory_desc_wrapper diff_dst_d(&diff_dst_pd);

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;

    jcp = zero<decltype(jcp)>();
    jcp.prop_kind = cd.prop_kind;

    if (with_groups)
        return status::unimplemented;
    jcp.ngroups = 1;

    jcp.mb = src_d.dims()[0];
    jcp.oc = diff_dst_d.dims()[1];
    jcp.ic = src_d.dims()[1];

    jcp.id = src_d.dims()[2];
    jcp.ih = src_d.dims()[3];
    jcp.iw = src_d.dims()[4];

    jcp.od = diff_dst_d.dims()[2];
    jcp.oh = diff_dst_d.dims()[3];
    jcp.ow = diff_dst_d.dims()[4];

    jcp.kd = diff_weights_d.dims()[2];
    jcp.kh = diff_weights_d.dims()[3];
    jcp.kw = diff_weights_d.dims()[4];

    jcp.d1_pad = cd.padding[0][0];
    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];

    jcp.stride_d = cd.strides[0];
    jcp.stride_h = cd.strides[1];
    jcp.stride_w = cd.strides[2];

    if (jcp.stride_d != 1 || jcp.stride_h != 1 || jcp.stride_w != 1)
        return status::unimplemented;

    jcp.dilate_d = cd.dilates[0];
    jcp.dilate_h = cd.dilates[1];
    jcp.dilate_w = cd.dilates[2];

    if (jcp.dilate_d != 0 || jcp.dilate_h != 0 || jcp.dilate_w != 0)
        return status::unimplemented;

    /* conditions on bias memory */
    jcp.with_bias = cd.diff_bias_desc.format != memory_format::undef;
    if (jcp.with_bias) {
        if (diff_bias_d.format() == any)
            CHECK(diff_bias_pd.set_format(x));
        if (diff_bias_d.format() != x)
            return status::unimplemented;
    }

    /* conditions on destination memory */
    jcp.oc_block = simd_w;
    if (jcp.oc % jcp.oc_block)
        return status::unimplemented;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    if (diff_dst_d.format() == any)
        CHECK(diff_dst_pd.set_format(nCdhw16c));
    if (diff_dst_d.format() != nCdhw16c)
        return status::unimplemented;

    /* kernel applicability check wrt boundaries
     * the conditions are quite general across the kernels we have,
     * but ideally the check should belong to a specific kernel... */
    const bool boundaries_ok = true
        && jcp.t_pad <= jcp.kh / 2
        && jcp.b_pad <= jcp.kh / 2
        && jcp.kh <= jcp.t_pad + jcp.ih /* [bwd_w:r1] */
        && jcp.kh <= jcp.ih; /* [bwd_w:r2] */
    if (!boundaries_ok)
        return status::unimplemented;

    if (jcp.kd * jcp.kh * jcp.kw > 28)
        return status::unimplemented;

    if (src_d.format() == any)
        CHECK(src_pd.set_format(ncdhw));

    const bool src_ok = true
        && jcp.ic == 1
        && implication(jcp.ic == 1, one_of(src_d.format(), ncdhw, ndhwc))
        && jcp.ngroups == 1;
    if (!src_ok)
        return status::unimplemented;

    jcp.ver = ver_fma;
    jcp.ic_block = 1;
    jcp.src_fmt = src_d.format();

    return status::success;
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
