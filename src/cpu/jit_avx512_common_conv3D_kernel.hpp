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

#ifndef JIT_AVX512_COMMON_CONV3D_KERNEL_F32_HPP
#define JIT_AVX512_COMMON_CONV3D_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "cpu_memory.hpp"

#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx512_common_conv3D_fwd_kernel
{
    struct common : public jit_generator {
        void genkernel(jit_conv_conf_t &jcp, int nocw);
        const unsigned char *jit_ker;
    };
    struct kernel_t : common {
        kernel_t(jit_conv_conf_t &jcp)
        {
            genkernel(jcp, 28);
            jit_ker = getCode();
        }
    };
    struct kernelrem_t: common {
        kernelrem_t(jit_conv_conf_t &jcp)
        {
            genkernel(jcp, jcp.ow % 28);
            jit_ker = getCode();
        }
    };
    jit_avx512_common_conv3D_fwd_kernel(jit_conv_conf_t ajcp) : jcp(ajcp)
    {
        kernel_ = new kernel_t(jcp);
        kernelrem_ = new kernelrem_t(jcp);
    }
    ~jit_avx512_common_conv3D_fwd_kernel()
    {
        delete kernel_;
        delete kernelrem_;
    }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd,
            cpu_memory_t::pd_t &src_pd,
            cpu_memory_t::pd_t &weights_pd,
            cpu_memory_t::pd_t &dst_pd,
            cpu_memory_t::pd_t &bias_pd,
            const primitive_attr_t &attr,
            bool with_relu = false,
            float relu_negative_slope = 0.);

    jit_conv_conf_t jcp;
    kernel_t *kernel_;
    kernelrem_t *kernelrem_;
};

struct jit_avx512_common_conv3D_bwd_data_kernel_f32: public jit_generator {

    jit_avx512_common_conv3D_bwd_data_kernel_f32(jit_conv_conf_t ajcp): jcp(ajcp)
    {
        generate();
        jit_ker = (void (*)(jit_conv_call_s *))getCode();
    }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd,
            const memory_desc_wrapper &diff_src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &diff_dst_d);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    enum {
        typesize = sizeof(float),
        ker_reg_base_idx = 28,
    };

    reg64_t param = abi_param1;
    reg64_t reg_dst = r8;
    reg64_t reg_ker = r9;
    reg64_t reg_src = r10;

    reg64_t reg_dst_prf = r11;
    reg64_t reg_ker_prf = r12;
    reg64_t reg_src_prf = r13;

    reg64_t aux_reg_dst = r14;
    reg64_t aux_reg_ker = r15;

    reg64_t aux_reg_dst_prf = rsi;
    reg64_t aux_reg_ker_prf = rdx;

    reg64_t reg_kj = rax;
    reg64_t reg_oi = rbx;
    reg64_t reg_kh = abi_not_param1;

    reg64_t reg_channel = rsi;

    reg64_t reg_tmp = rbp;

    inline Xbyak::Zmm zmm_ker(int i_ic) {
        assert(i_ic < 4);
        return Xbyak::Zmm(ker_reg_base_idx + i_ic);
    }
    inline Xbyak::Zmm zmm_inp(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return Xbyak::Zmm(idx);
    }
    inline Xbyak::Zmm zmm_out(int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return Xbyak::Zmm(idx);
    }
    inline void vadd(Xbyak::Zmm zmm, reg64_t reg, int offset) {
        if (jcp.ver == ver_4vnni)
            vpaddd(zmm, zmm, EVEX_compress_addr(reg, offset));
        else
            vaddps(zmm, zmm, EVEX_compress_addr(reg, offset));
    }

    Xbyak::Zmm zmm_wei = Xbyak::Zmm(31);

    inline void prepare_output(int ur_w);
    inline void store_output(int ur_w);
    inline void compute_loop_4fma(int ur_w, int l_overflow, int r_overflow);
    inline void compute_loop_4vnni(int ur_w, int l_overflow, int r_overflow);
    inline void compute_loop_fma(int ur_w, int l_overflow, int r_overflow);
    inline void compute_loop_fma_core(int ur_w, int l_overflow, int r_overflow);
    inline void compute_loop(int ur_w, int l_overflow, int r_overflow);
    void generate();

    inline int get_iw_start(int ki, int l_overflow)
    {
        int r_pad = jcp.stride_w * (jcp.ow - 1) + jcp.kw - jcp.iw - jcp.l_pad;
        int k_max = jcp.kw - 1 - (jcp.iw - 1 + r_pad) % jcp.stride_w
            - l_overflow * jcp.stride_w;
        int res = ki - k_max;
        while (res < 0)
            res += jcp.stride_w;

        return res;

    }

    inline int get_iw_end(int ur_w, int ki, int r_overflow)
    {
        if (ur_w == jcp.ur_w_tail) {
            int r_pad = nstl::min(0, jcp.stride_w * (jcp.ow - 1) + jcp.kw
                    - jcp.iw - jcp.l_pad);
            ur_w += r_pad;
        }
        int k_min = (ur_w - 1 + jcp.l_pad) % jcp.stride_w + r_overflow
            * jcp.stride_w;
        int res = k_min - ki;
        while (res < 0)
            res += jcp.stride_w;

        return ur_w - res;
    }
};

struct jit_avx512_common_conv3D_bwd_weights_kernel_f32 : public jit_generator {

    jit_avx512_common_conv3D_bwd_weights_kernel_f32(jit_conv_conf_t ajcp)
        : jcp(ajcp)
    {
        generate();
        jit_ker = (void (*)(jit_conv_call_s *))getCode();
    }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, cpu_memory_t::pd_t &src_pd,
            cpu_memory_t::pd_t &diff_weights_pd,
            cpu_memory_t::pd_t &diff_bias_pd, cpu_memory_t::pd_t &diff_dst_pd);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    enum {typesize = sizeof(float)};
    static const int max_ur_w;

    reg64_t param = abi_param1;
    reg64_t reg_input = rax;
    reg64_t reg_kernel = rdx;
    reg64_t reg_output = rsi;
    reg64_t b_ic = abi_not_param1;
    reg64_t kj = r8;
    reg64_t reg_kh  = r9;
    reg64_t reg_ur_w_trips  = r10;
    reg64_t reg_oj = r15;
    reg64_t reg_ih_count = rbx;
    reg64_t reg_tmp = r14;

    inline void maybe_zero_kernel();
    inline void compute_oh_step_unroll_ow_icblock(int ic_block_step,
            int max_ur_w);
    inline void oh_step_comeback_pointers();
    inline void compute_oh_step_unroll_ow(int ic_block_step, int max_ur_w);
    inline void compute_ic_block_step(int ur_w,
            int pad_l, int pad_r, int ic_block_step,
            int input_offset, int kernel_offset, int output_offset,
            bool input_wraparound = false);
    inline void compute_ic_block_step_fma(int ur_w,
            int pad_l, int pad_r, int ic_block_step,
            int input_offset, int kernel_offset, int output_offset,
            bool input_wraparound);
    inline void compute_ic_block_step_4fma(int ur_w,
            int pad_l, int pad_r, int ic_block_step,
            int input_offset, int kernel_offset, int output_offset,
            bool input_wraparound);
    inline void compute_oh_step_common(int ic_block_step, int max_ur_w);
    inline void compute_oh_step_disp();
    inline void compute_oh_loop_common();

    inline bool compute_full_spat_loop();
    inline bool flat_4ops_compute();

    inline void compute_loop();

    void generate();
};


}
}
}

#endif
