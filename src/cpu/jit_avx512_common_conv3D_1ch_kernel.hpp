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

#ifndef JIT_AVX512_COMMON_CONV_3D_1CH_KERNEL_F32_HPP
#define JIT_AVX512_COMMON_CONV_3D_1CH_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "cpu_memory.hpp"

#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx512_common_conv3D_1ch_fwd_kernel_f32
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
    jit_avx512_common_conv3D_1ch_fwd_kernel_f32(jit_conv_conf_t ajcp) : jcp(ajcp)
    {
        kernel_ = new kernel_t(jcp);
        kernelrem_ = new kernelrem_t(jcp);
    }
    ~jit_avx512_common_conv3D_1ch_fwd_kernel_f32()
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

struct jit_avx512_common_conv3D_1ch_bwd_weights_kernel_f32 : public jit_generator {

    jit_avx512_common_conv3D_1ch_bwd_weights_kernel_f32(jit_conv_conf_t ajcp)
        : jcp(ajcp)
    {
        generate();
        jit_ker = (void*) getCode();
    }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd,
            cpu_memory_t::pd_t &src_pd,
            cpu_memory_t::pd_t &diff_weights_pd,
            cpu_memory_t::pd_t &diff_bias_pd,
            cpu_memory_t::pd_t &diff_dst_pd);

    jit_conv_conf_t jcp;
    void *jit_ker;

private:
    using reg64_t = const Xbyak::Reg64;
    enum {
        typesize = sizeof(float),
    };

    void generate();

};

}
}
}

#endif
