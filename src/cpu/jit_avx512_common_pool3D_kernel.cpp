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

#include "jit_avx512_common_pool3D_kernel.hpp"
#include <omp.h>

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace Xbyak;
using namespace alg_kind;

void jit_avx512_common_pool3D_fwd_kernel_f32::generate()
{
    preamble();

    // Base pointers
    const reg64_t rdst = rdi;
    const reg64_t rsrc = rsi;

    // loop counters
    Reg64 rKD = r8;
    Reg64 rKH = r9;
    Reg64 rKW = rax;

    // src pointers
    Reg64 rsrc_kh_start = r11;
    Reg64 rsrc_kw_start = r12;

    // increments
    const int ID_OFF = jpp.ih*jpp.iw*16*sizeof(float);
    const int IH_OFF = jpp.iw*16*sizeof(float);
    const int IW_OFF = 16*sizeof(float);

    Zmm accum = Zmm(0);
    Zmm tmp_src = Zmm(1);
    Reg64 rdenom = r10;
    Zmm vdenom = Zmm(2);

    if (jpp.alg == pooling_max) {

        // set accumulator to min value
        // NOTE is there an easier way to do a broadcast?
        mov(rdenom, float2int(std::numeric_limits<float>::min()));
        movq(xmm1, rdenom);
        vbroadcastss(accum, xmm1);

        Label kd_loop, kh_loop, kw_loop;
        mov(rKD, jpp.kd);
        L(kd_loop);
            mov(rsrc_kh_start, rsrc);
            add(rsrc, ID_OFF);
            mov(rKH, jpp.kh);
            L(kh_loop);
                mov(rsrc_kw_start, rsrc_kh_start);
                add(rsrc_kh_start, IH_OFF);
                mov(rKW, jpp.kw);
                L(kw_loop);
                    vmovups(tmp_src, ptr[rsrc_kw_start]);
                    vmaxps(accum, accum, tmp_src);
                    add(rsrc_kw_start, IW_OFF);
                sub(rKW, 1);
                jnz(kw_loop);
            sub(rKH, 1);
            jnz(kh_loop);
        sub(rKD, 1);
        jnz(kd_loop);

        vmovntps(ptr[rdst], accum);

    } else {

        // zero accumulator
        vpxord(accum, accum, accum);

        // compute denominator
        // NOTE is there an easier way to do a broadcast?
        mov(rdenom, float2int(1.0f/(jpp.kd*jpp.kw*jpp.kh)));
        movq(xmm1, rdenom);
        vbroadcastss(vdenom, xmm1);

        Label kd_loop, kh_loop, kw_loop;
        mov(rKD, jpp.kd);
        L(kd_loop);
            mov(rsrc_kh_start, rsrc);
            add(rsrc, ID_OFF);
            mov(rKH, jpp.kh);
            L(kh_loop);
                mov(rsrc_kw_start, rsrc_kh_start);
                add(rsrc_kh_start, IH_OFF);
                mov(rKW, jpp.kw);
                L(kw_loop);
                    vmovups(tmp_src, ptr[rsrc_kw_start]);
                    // vfmadd231ps(accum, vdenom, tmp_src);
                    vaddps(accum, accum, tmp_src);
                    add(rsrc_kw_start, IW_OFF);
                sub(rKW, 1);
                jnz(kw_loop);
            sub(rKH, 1);
            jnz(kh_loop);
        sub(rKD, 1);
        jnz(kd_loop);

        // scale by denominator (if using vaddps)
        vmulps(accum, accum, vdenom);
        vmovntps(ptr[rdst], accum);

    }

    postamble();
}

status_t jit_avx512_common_pool3D_fwd_kernel_f32::init_conf(
        jit_pool_conf_t &jpp, const pooling_desc_t &pd,
        cpu_memory_t::pd_t &src_pd, cpu_memory_t::pd_t &dst_pd)
{
    if (!mayiuse(avx512_common)) {
        printf(" No AVX512 available!\n");
        return status::unimplemented;
    }

    // const int simd_w = cpu_isa_traits<avx512_common>::vlen / sizeof(float);

    const memory_desc_wrapper src_d(&src_pd);
    const memory_desc_wrapper dst_d(&dst_pd);

    jpp = zero<decltype(jpp)>();

    // jpp.ngroups = 1;

    jpp.mb = src_d.dims()[0];

    jpp.c = dst_d.dims()[1];

    jpp.id = src_d.dims()[2];
    jpp.ih = src_d.dims()[3];
    jpp.iw = src_d.dims()[4];

    jpp.od = dst_d.dims()[2];
    jpp.oh = dst_d.dims()[3];
    jpp.ow = dst_d.dims()[4];

    jpp.kd = pd.kernel[0];
    jpp.kh = pd.kernel[1];
    jpp.kw = pd.kernel[2];

    jpp.d1_pad = pd.padding[0][0];
    jpp.t_pad = pd.padding[0][1];
    jpp.l_pad = pd.padding[0][2];

    jpp.stride_d = pd.strides[0];
    jpp.stride_h = pd.strides[1];
    jpp.stride_w = pd.strides[2];

    jpp.alg = pd.alg_kind;

    // if (jpp.stride_d != 1 || jpp.stride_h != 1 || jpp.stride_w != 1)
    //     return status::unimplemented;

    if (jpp.d1_pad != 0 || jpp.t_pad != 0 || jpp.l_pad != 0)
        return status::unimplemented;

    /* conditions on destination memory */
//     jpp.oc_block = simd_w;
//     if (jpp.oc % jpp.oc_block)
//         return status::unimplemented;
//     jpp.nb_oc = jpp.oc / jpp.oc_block;

    if (dst_d.format() == any)
        CHECK(dst_pd.set_format(nCdhw16c));
    if (dst_d.format() != nCdhw16c)
        return status::unimplemented;

    /* kernel applicability check wrt boundaries
     * the conditions are quite general across the kernels we have,
     * but ideally the check should belong to a specific kernel... */
//     const bool boundaries_ok = true
//         && jpp.t_pad <= jpp.kh / 2
//         && jpp.b_pad <= jpp.kh / 2
//         && jpp.kh <= jpp.t_pad + jpp.ih /* [bwd_w:r1] */
//         && jpp.kh <= jpp.ih; /* [bwd_w:r2] */
//     if (!boundaries_ok)
//         return status::unimplemented;

//     if (jpp.kd * jpp.kh * jpp.kw > 28)
//         return status::unimplemented;

    if (src_d.format() == any)
        CHECK(src_pd.set_format(nCdhw16c));

    const bool src_ok = true
        && src_d.format() == nCdhw16c;
    if (!src_ok)
        return status::unimplemented;

    // jpp.ver = ver_fma;

    return status::success;
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
