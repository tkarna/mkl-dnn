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

#include "jit_avx512_common_conv3D_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)
#define KNx_L2_EFFECTIVE_CAPACITY ((512-64)*1024)


namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace Xbyak;


void jit_avx512_common_conv3D_fwd_kernel::common::genkernel(jit_conv_conf_t &jcp, int now)
{
    // Arguments
    const Reg64 rsrc = rdi;
    const Reg64 rweights = rsi;
    const Reg64 rdst = rdx;
    const Reg64 rbias = rcx;
    const Reg64 rslope = r8;

    // loop counters
    Reg64 rKD = r8;
    Reg64 rKH = r9;
    Reg64 rKW = rax;

    // src pointers
    Reg64 rsrcp1 = r10;
    Reg64 rsrcp2 = r11;

    // increments
    int incIW = jcp.iw * 64;
    int incIWIH = incIW * jcp.ih;

    // load the outputs
    for (int ow = 0; ow < now;++ow)
        vmovups(Zmm(ow+4), ptr[rdst + ow*64]);

    // loops
    Label kd_loop, kh_loop, kw_loop;
    mov(rKD, jcp.kd);
    L(kd_loop);
    mov(rsrcp1, rsrc);
    add(rsrc, incIWIH);
    mov(rKH, jcp.kh);
    L(kh_loop);
    mov(rsrcp2, rsrcp1);
    add(rsrcp1, incIW);
    mov(rKW, jcp.kw);
    L(kw_loop);

    if (jcp.ver == ver_4fma) {
        // 4 sets of 4 inputs
        for (int icx = 0; icx < 4; ++icx)
        {
            // load 4 vectors of weights
            for (int ic = 0; ic < 4; ++ic)
                vmovups(Zmm(ic), ptr[rweights + 64*(4*icx+ic)]);
            for (int ow = 0; ow < now; ++ow)
                v4fmaddps(Zmm(ow+4), Zmm(0), ptr[rsrcp2 + 64*ow*jcp.stride_w + 4*4*icx]);
        }
    } else {
        assert(jcp.ver == ver_fma);
        // 16 inputs
        for (int icx = 0; icx < 16; ++icx) {
            // load next vector of weights
            vmovups(Zmm(0), ptr[rweights + 64*icx]);
            for (int ow = 0; ow < now; ++ow)
                vfmadd231ps(Zmm(ow+4), Zmm(0), ptr_b[rsrcp2 + 64*ow*jcp.stride_w + 4*icx]);
        }
    }
    add(rweights, 4*16*16);
    add(rsrcp2, 4*16);
    sub(rKW, 1);
    jnz(kw_loop);
    sub(rKH, 1);
    jnz(kh_loop);
    sub(rKD, 1);
    jnz(kd_loop);

    for (int ow = 0; ow < now; ++ow)
        vmovups(ptr[rdst + ow*64], Zmm(ow+4));

    ret();
}

status_t jit_avx512_common_conv3D_fwd_kernel::init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, cpu_memory_t::pd_t &src_pd,
            cpu_memory_t::pd_t &weights_pd, cpu_memory_t::pd_t &dst_pd,
            cpu_memory_t::pd_t &bias_pd, const primitive_attr_t &attr,
            bool with_relu, float relu_negative_slope)
{
    using namespace prop_kind;

    if (!mayiuse(avx512_common))
        return status::unimplemented;

    const memory_desc_wrapper src_d(&src_pd);
    const memory_desc_wrapper weights_d(&weights_pd);
    const memory_desc_wrapper dst_d(&dst_pd);
    const memory_desc_wrapper bias_d(&bias_pd);

    const int regs = 28;
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;

    // we don't understand groups for 3D conv
    assert(!with_groups);

    jcp = zero<decltype(jcp)>();
    jcp.ngroups = 1;
    jcp.prop_kind = cd.prop_kind;

    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1];
    jcp.ic = src_d.dims()[1];

    jcp.id = src_d.dims()[2];
    jcp.ih = src_d.dims()[3];
    jcp.iw = src_d.dims()[4];

    jcp.od = dst_d.dims()[2];
    jcp.oh = dst_d.dims()[3];
    jcp.ow = dst_d.dims()[4];

    jcp.kd = weights_d.dims()[2];
    jcp.kh = weights_d.dims()[3];
    jcp.kw = weights_d.dims()[4];


    jcp.stride_d = cd.strides[0];
    jcp.stride_h = cd.strides[1];
    jcp.stride_w = cd.strides[2];

    jcp.d1_pad = cd.padding[0][0];
    jcp.t_pad = cd.padding[0][1];
    jcp.l_pad = cd.padding[0][2];
    jcp.src_fmt = src_d.format();
    jcp.with_relu = with_relu;
    jcp.relu_negative_slope = relu_negative_slope;
    jcp.ur_h = 1;

    jcp.dilate_d = cd.dilates[0];
    jcp.dilate_h = cd.dilates[1];
    jcp.dilate_w = cd.dilates[2];
    if (jcp.dilate_d != 0 || jcp.dilate_h != 0 || jcp.dilate_w != 0)
        return status::unimplemented;

    if (dst_d.format() == any)
        CHECK(dst_pd.set_format(nCdhw16c));
    if (dst_d.format() != nCdhw16c)
        return status::unimplemented;

    if (src_d.format() == any)
        CHECK(src_pd.set_format(nCdhw16c));
    if (src_d.format() != nCdhw16c)
        return status::unimplemented;

    if (weights_d.format() == any)
        CHECK(weights_pd.set_format(OIdhw16i16o));
    if (weights_d.format() != OIdhw16i16o)
        return status::unimplemented;

    jcp.with_bias = cd.bias_desc.format != memory_format::undef;
    if (jcp.with_bias) {
        if (bias_d.format() == any)
            CHECK(bias_pd.set_format(x));
        if (bias_d.format() != x)
            return status::unimplemented;
    }

    jcp.typesize_in = sizeof(float);
    jcp.typesize_out = sizeof(float);

    if (mayiuse(avx512_mic_4ops))
        jcp.ver = ver_4fma;
    else if (mayiuse(avx512_common))
        jcp.ver = ver_fma;
    else
        return status::unimplemented;

    return status::success;
}



void jit_avx512_common_conv3D_bwd_data_kernel_f32::generate()
{
    printf(">>> generate called\n");
}

status_t jit_avx512_common_conv3D_bwd_data_kernel_f32::init_conf(
        jit_conv_conf_t &jcp,
        const convolution_desc_t &cd,
        cpu_memory_t::pd_t &diff_src_pd,
        cpu_memory_t::pd_t &weights_pd,
        cpu_memory_t::pd_t &diff_dst_pd)

{
    if (!mayiuse(avx512_common)) return status::unimplemented;
    const memory_desc_wrapper diff_src_d(&diff_src_pd);
    const memory_desc_wrapper weights_d(&weights_pd);
    const memory_desc_wrapper diff_dst_d(&diff_dst_pd);

    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;
    assert(!with_groups);

    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = diff_src_d.dims()[1] / jcp.ngroups;

    jcp.id = diff_src_d.dims()[2];
    jcp.ih = diff_src_d.dims()[3];
    jcp.iw = diff_src_d.dims()[4];
    jcp.od = diff_dst_d.dims()[2];
    jcp.oh = diff_dst_d.dims()[3];
    jcp.ow = diff_dst_d.dims()[4];

    jcp.kd = weights_d.dims()[with_groups + 2];
    jcp.kh = weights_d.dims()[with_groups + 3];
    jcp.kw = weights_d.dims()[with_groups + 4];

    jcp.d1_pad = cd.padding[0][0];
    jcp.t_pad = cd.padding[0][1];
    jcp.l_pad = cd.padding[0][2];

    jcp.stride_d = cd.strides[0];
    jcp.stride_h = cd.strides[1];
    jcp.stride_w = cd.strides[2];

    jcp.dilate_d = cd.dilates[0];
    jcp.dilate_h = cd.dilates[1];
    jcp.dilate_w = cd.dilates[2];
    if (jcp.dilate_d != 0 || jcp.dilate_h != 0 || jcp.dilate_w != 0)
        return status::unimplemented;

    bool args_ok = true
        && diff_src_d.format() == nCdhw16c
        && diff_dst_d.format() == nCdhw16c;
    if (!args_ok)
        return status::unimplemented;

    if (mayiuse(avx512_common)) {
        if (weights_d.format() != OIdhw16o16i)
            return status::unimplemented;
        jcp.ver = ver_fma;
        jcp.typesize_in = sizeof(float);
        jcp.typesize_out = sizeof(float);
        if (mayiuse(avx512_mic_4ops)
            && jcp.stride_w == 1 && jcp.stride_h == 1) {
                jcp.ver = ver_4fma;
            }
    } else {
        return status::unimplemented;
    }

    return status::success;
}

#if 0

void jit_avx512_common_conv3D_bwd_weights_kernel_f32::generate()
{
    preamble();

    mov(reg_input, ptr[param + GET_OFF(src)]);
    mov(reg_output, ptr[param + GET_OFF(dst)]);
    mov(reg_kernel, ptr[param + GET_OFF(filt)]);

    compute_loop();

    postamble();
}

status_t jit_avx512_common_conv3D_bwd_weights_kernel_f32::init_conf(
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

    jcp.ngroups = with_groups ? diff_weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = diff_dst_d.dims()[2];
    jcp.ow = diff_dst_d.dims()[3];

    jcp.kh = diff_weights_d.dims()[with_groups + 2];
    jcp.kw = diff_weights_d.dims()[with_groups + 3];

    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];

    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];

    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];
    if (jcp.dilate_h != 0 || jcp.dilate_w != 0)
        return status::unimplemented;

    jcp.r_pad = nstl::max(0, (jcp.ow - 1) * jcp.stride_w + jcp.kw - jcp.iw
        - jcp.l_pad);
    jcp.b_pad = nstl::max(0, (jcp.oh - 1) * jcp.stride_h + jcp.kh - jcp.ih
        - jcp.t_pad);

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;

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
        CHECK(diff_dst_pd.set_format(nChw16c));
    if (diff_dst_d.format() != nChw16c)
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

    /* yet another common check */
    if (jcp.kw > 14)
        return status::unimplemented;

    /* setting register strategy */
    for (int ur_w = nstl::min(max_ur_w, jcp.ow); ur_w > 0; --ur_w) {
        if (jcp.ow % ur_w == 0) { jcp.ur_w = ur_w; break; }
    }

    /* check for the 1st convolution */
    jcp.is_1stconv = jcp.ic % simd_w;
    if (jcp.is_1stconv) {
        if (src_d.format() == any)
            CHECK(src_pd.set_format(nchw));

        const bool src_ok = true
            && one_of(jcp.ic, 1, 3)
            && implication(jcp.ic == 1, one_of(src_d.format(), nchw, nhwc))
            && implication(jcp.ic != 1, src_d.format() == nchw)
            && jcp.ngroups == 1;
        if (!src_ok)
            return status::unimplemented;

        const int tr_ld = rnd_up(div_up(jcp.iw + jcp.l_pad + jcp.r_pad,
                    jcp.stride_w), 16);
        const int kh_step = nstl::max((28 - jcp.with_bias) / jcp.kw, 1);
        const int kh_step_rem = jcp.kh % kh_step;
        const auto want_4fma_wfmt = with_groups ? gOihw16o : Oihw16o;
        const bool use_4fma = true
            && mayiuse(avx512_mic_4ops)
            && everyone_is(0, jcp.l_pad, jcp.r_pad, jcp.t_pad, jcp.b_pad)
            && jcp.kw <= 28 - jcp.with_bias
            && jcp.stride_w == 4
            && tr_ld / simd_w <= 4 /* [bwd_w:tr_src:r1] */
            && implication(jcp.with_bias, kh_step_rem == 1) /* [bwd_w:b:r1] */
            && implication(diff_weights_d.format() != any,
                    diff_weights_d.format() == want_4fma_wfmt);

        if (use_4fma) {
            jcp.ver = ver_4fma;
            jcp.kh_step = kh_step;
            jcp.tr_ld = tr_ld;
            jcp.ic_block = 1;
            if (diff_weights_d.format() == any)
                CHECK(diff_weights_pd.set_format(want_4fma_wfmt));
        } else {
            jcp.ver = ver_fma;
            jcp.ic_block = jcp.ic;

            const auto want_wfmt = with_groups ? gOhwi16o : Ohwi16o;
            if (diff_weights_d.format() == any)
                CHECK(diff_weights_pd.set_format(want_wfmt));
            if (diff_weights_d.format() != want_wfmt)
                return status::unimplemented;
        }

        jcp.nb_ic = jcp.ic / jcp.ic_block;
        jcp.src_fmt = src_d.format();
    } else {
        if (src_d.format() == any)
            CHECK(src_pd.set_format(nChw16c));
        if (diff_weights_d.format() == any)
            CHECK(diff_weights_pd.set_format(with_groups
                        ? gOIhw16i16o : OIhw16i16o));

        const bool ok = true
            && src_d.format() == nChw16c
            && diff_weights_d.format() == (with_groups
                    ? gOIhw16i16o : OIhw16i16o);
        if (!ok)
            return status::unimplemented;

        jcp.ic_block = simd_w;
        jcp.nb_ic = jcp.ic / jcp.ic_block;
        jcp.src_fmt = src_d.format();

        if (mayiuse(avx512_mic_4ops) && jcp.stride_w == 1)
            jcp.ver = ver_4fma;
        else
            jcp.ver = ver_fma;

        if (jcp.ver == ver_4fma) {
            jcp.ur_w = jcp.ow;
            // XXX, BUGBUGBUG, but not a FIXME: this assumes that it's OK to
            // cross the right boundary. The only requirement is not to have
            // NaNs there because another multiplicand is always guaranteed to
            // be zero. This also may require the top-level driver to allocate
            // four extra guarding elements at the very end of the buffer.
            // I'm not proud of this hack, but it improves performance by
            // about 5-10% depending on the dimensions (Roma)
            jcp.tr_iw = rnd_up(jcp.iw + jcp.kw - 1, 4);
            jcp.tr_src_num_guard_elems = 4; // upper bound
        }
    }

    return status::success;
}
#endif

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
