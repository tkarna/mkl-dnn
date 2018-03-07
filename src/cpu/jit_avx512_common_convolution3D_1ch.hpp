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

#ifndef JIT_AVX512_COMMON_CONVOLUTION3D_1CH
#define JIT_AVX512_COMMON_CONVOLUTION3D_1CH

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "mkldnn_thread.hpp"
#include "jit_avx512_common_conv3D_1ch_kernel.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t src_type, impl::data_type_t diff_wei_type,
         impl::data_type_t diff_dst_type,
         impl::data_type_t acc_type = diff_wei_type>
struct jit_avx512_common_convolution3D_1ch_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd),
              jcp_({})
        {}

        DECLARE_COMMON_PD_T(jit_avx512_common_convolution3D_1ch_bwd_weights_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->desc()->prop_kind, backward,
                        backward_weights)
                && this->desc()->alg_kind == alg_kind::convolution_direct
                && this->desc()->src_desc.data_type == src_type
                && this->desc()->diff_weights_desc.data_type == diff_wei_type
                && this->desc()->diff_dst_desc.data_type == diff_dst_type
                && this->desc()->accum_data_type == acc_type
                && this->desc()->conv_kind == conv_kind::conv3D
                && this->desc()->src_desc.dims[1] == 1
                && this->desc()->diff_dst_desc.dims[1] == 16 // TODO: Extend support to % 16 == 0
                && utils::implication(this->with_bias(),
                        this->desc()->diff_bias_desc.data_type
                        == diff_wei_type)
                && this->attr()->has_default_values();
            if (!ok) {
                return status::unimplemented;
            }
            auto jit_status = jit_avx512_common_conv3D_1ch_bwd_weights_kernel_f32::init_conf(
                    jcp_, *this->desc(), this->src_pd_, this->diff_weights_pd_,
                    this->diff_bias_pd_, this->diff_dst_pd_);
            return jit_status;
        }
        jit_conv_conf_t jcp_;

        inline int MB() const { return this->desc()->src_desc.dims[0]; }

        inline int IC() const { return this->desc()->src_desc.dims[1]; }
        inline int OC() const { return this->desc()->diff_dst_desc.dims[1]; }
        inline int G() const
        { return with_groups() ? this->desc()->diff_weights_desc.dims[0] : 1; }

        inline int ID() const { return this->desc()->src_desc.dims[2]; }
        inline int IH() const { return this->desc()->src_desc.dims[3]; }
        inline int IW() const { return this->desc()->src_desc.dims[4]; }
        inline int OD() const { return this->desc()->diff_dst_desc.dims[2]; }
        inline int OH() const { return this->desc()->diff_dst_desc.dims[3]; }
        inline int OW() const { return this->desc()->diff_dst_desc.dims[4]; }
        inline int KD() const
        { return this->desc()->diff_weights_desc.dims[2 + with_groups()]; }
        inline int KH() const
        { return this->desc()->diff_weights_desc.dims[3 + with_groups()]; }
        inline int KW() const
        { return this->desc()->diff_weights_desc.dims[4 + with_groups()]; }

        inline int KSD() const { return this->desc()->strides[0]; }
        inline int KSH() const { return this->desc()->strides[1]; }
        inline int KSW() const { return this->desc()->strides[2]; }

        inline int KDD() const { return this->desc()->dilates[0]; }
        inline int KDH() const { return this->desc()->dilates[1]; }
        inline int KDW() const { return this->desc()->dilates[2]; }

        inline int padD1() const { return this->desc()->padding[0][0]; }
        inline int padD2() const { return this->desc()->padding[1][0]; }
        inline int padT() const { return this->desc()->padding[0][1]; }
        inline int padB() const { return this->desc()->padding[1][1]; }
        inline int padL() const { return this->desc()->padding[0][2]; }
        inline int padR() const { return this->desc()->padding[1][2]; }
      protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;

            if (src_pd_.desc()->format == any)
                CHECK(src_pd_.set_format(ncdhw));
            if (diff_dst_pd_.desc()->format == any)
                CHECK(diff_dst_pd_.set_format(nCdhw16c));
            if (diff_weights_pd_.desc()->format == any)
                CHECK(diff_weights_pd_.set_format(
                            this->with_groups() ? gOidhw16o : Oidhw16o));
            if (diff_bias_pd_.desc()->format == any)
                CHECK(diff_bias_pd_.set_format(x));
            return status::success;
        }
    };

    jit_avx512_common_convolution3D_1ch_bwd_weights_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {
        kernel_ = new jit_avx512_common_conv3D_1ch_bwd_weights_kernel_f32(conf_.jcp_);
    }

    ~jit_avx512_common_convolution3D_1ch_bwd_weights_t()
    {
        delete kernel_;
    }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<diff_wei_type>::type diff_wei_data_t;
    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    virtual void execute(event_t *e) {
        switch (conf_.desc()->prop_kind) {
        case prop_kind::backward:
        case prop_kind::backward_weights:
            execute_backward_weights();
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_weights();
    pd_t conf_;
    jit_avx512_common_conv3D_1ch_bwd_weights_kernel_f32 *kernel_;

};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
