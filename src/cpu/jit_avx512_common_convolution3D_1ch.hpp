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

template <bool with_relu, impl::data_type_t src_type,
         impl::data_type_t wei_type = src_type,
         impl::data_type_t dst_type = src_type,
         impl::data_type_t acc_type = dst_type>
struct _jit_avx512_common_convolution3D_1ch_fwd_t: public cpu_primitive_t {
    struct pd_t: public _cpu_convolution_fwd_pd_t<with_relu> {
        pd_t(engine_t *engine,
                const typename pd_t::base_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : _cpu_convolution_fwd_pd_t<with_relu>(engine, adesc, attr,
                    hint_fwd_pd), jcp_({})
        {}

        DECLARE_COMMON_PD_T(_jit_avx512_common_convolution3D_1ch_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            //debug_me();
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->cdesc_().prop_kind, forward_training,
                        forward_inference)
                && this->cdesc_().alg_kind == alg_kind::convolution_direct
                && this->cdesc_().src_desc.data_type == src_type
                && this->cdesc_().weights_desc.data_type == wei_type
                && this->cdesc_().accum_data_type == acc_type
                && this->cdesc_().dst_desc.data_type == dst_type
                && this->cdesc_().conv_kind == conv_kind::conv3D
                && this->cdesc_().dst_desc.dims[1] % 16 == 0
                && utils::one_of(this->cdesc_().src_desc.format, memory_format::ncdhw, memory_format::any)
                && utils::one_of(this->cdesc_().dst_desc.format, memory_format::nCdhw16c, memory_format::any)
                && padT() == 0 && padL() == 0 && padD1() == 0
                && src_type == data_type::f32
                && dst_type == data_type::f32
                && wei_type == data_type::f32
                && utils::implication(this->with_bias(), this->cdesc_().bias_desc.data_type == data_type::f32)
                && this->attr()->has_default_values();

            if (!ok) {
                return status::unimplemented;
            }
            auto status = jit_avx512_common_conv3D_1ch_fwd_kernel_f32::init_conf(
                    jcp_, this->cdesc_(), this->src_pd_, this->weights_pd_,
                    this->dst_pd_,this->bias_pd_, *this->attr(),
                    with_relu, this->negative_slope());
            return status;
        }
        jit_conv_conf_t jcp_;

        inline int MB() const { return this->cdesc_().src_desc.dims[0]; }

        inline int IC() const { return this->cdesc_().src_desc.dims[1]; }
        inline int OC() const { return this->cdesc_().dst_desc.dims[1]; }
        inline int G() const
        { return this->with_groups() ? this->cdesc_().weights_desc.dims[0] : 1; }

        inline int ID() const { return this->cdesc_().src_desc.dims[2]; }
        inline int IH() const { return this->cdesc_().src_desc.dims[3]; }
        inline int IW() const { return this->cdesc_().src_desc.dims[4]; }

        inline int OD() const { return this->cdesc_().dst_desc.dims[2]; }
        inline int OH() const { return this->cdesc_().dst_desc.dims[3]; }
        inline int OW() const { return this->cdesc_().dst_desc.dims[4]; }

        inline int KD() const
        { return this->cdesc_().weights_desc.dims[2 + this->with_groups()]; }
        inline int KH() const
        { return this->cdesc_().weights_desc.dims[3 + this->with_groups()]; }
        inline int KW() const
        { return this->cdesc_().weights_desc.dims[4 + this->with_groups()]; }

        inline int KSD() const { return this->cdesc_().strides[0]; }
        inline int KSH() const { return this->cdesc_().strides[1]; }
        inline int KSW() const { return this->cdesc_().strides[2]; }

        inline int KDD() const { return this->cdesc_().dilates[0]; }
        inline int KDH() const { return this->cdesc_().dilates[1]; }
        inline int KDW() const { return this->cdesc_().dilates[2]; }

        inline int padD1() const { return this->cdesc_().padding[0][0]; }
        inline int padD2() const { return this->cdesc_().padding[1][0]; }
        inline int padT() const { return this->cdesc_().padding[0][1]; }
        inline int padB() const { return this->cdesc_().padding[1][1]; }
        inline int padL() const { return this->cdesc_().padding[0][2]; }
        inline int padR() const { return this->cdesc_().padding[1][2]; }

      protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;

            if (this->src_pd_.desc()->format == any) {
                CHECK(this->src_pd_.set_format(ncdhw));
            }
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(nCdhw16c));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(this->with_groups() ? gOidhw16o : Oidhw16o));
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            return status::success;
        }
    };

    _jit_avx512_common_convolution3D_1ch_fwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
    {
            kernel_ = new  jit_avx512_common_conv3D_1ch_fwd_kernel_f32(conf_.jcp_);
    }
    ~_jit_avx512_common_convolution3D_1ch_fwd_t() { delete kernel_; };

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    virtual void execute(event_t *e) {
        switch (conf_.cdesc()->prop_kind) {
        case prop_kind::forward_training:
        case prop_kind::forward_inference:
            execute_forward();
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;
    jit_avx512_common_conv3D_1ch_fwd_kernel_f32 *kernel_;

};

template <impl::data_type_t src_type, impl::data_type_t wei_type = src_type,
         impl::data_type_t dst_type = src_type,
         impl::data_type_t acc_type = dst_type>
using jit_avx512_common_convolution3D_1ch_fwd_t = _jit_avx512_common_convolution3D_1ch_fwd_t<false, src_type, wei_type,
      dst_type, acc_type>;

template <impl::data_type_t src_type, impl::data_type_t wei_type = src_type,
         impl::data_type_t dst_type = src_type,
         impl::data_type_t acc_type = dst_type>
using jit_avx512_common_convolution3D_1ch_relu_t = _jit_avx512_common_convolution3D_1ch_fwd_t<true, src_type, wei_type,
      dst_type, acc_type>;


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
        balance();

        // TODO: The private space is this big in the worst case, but likely to be much smaller.
        // TODO: Experiment with alternative mallocs.
        auto &jcp = conf_.jcp_;
        const int nthr_ndhw = nthr_mb_*nthr_od_*nthr_oh_*nthr_ow_;
        private_weights_ = (diff_wei_data_t*) malloc(jcp.kd*jcp.kh*jcp.kw*nthr_ndhw*jcp.oc_block*sizeof(diff_wei_data_t), 2*1024*1024);
        private_bias_ = (acc_data_t*) malloc(nthr_ndhw*jcp.oc_block*sizeof(acc_data_t), 2*1024*1024);
    }

    ~jit_avx512_common_convolution3D_1ch_bwd_weights_t()
    {
        if (private_bias_) free(private_bias_);
        if (private_weights_) free(private_weights_);
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
    void balance();

    struct thread_info_t;

    pd_t conf_;

    jit_avx512_common_conv3D_1ch_bwd_weights_kernel_f32 *kernel_;
    diff_wei_data_t* private_weights_;
    acc_data_t* private_bias_;

    int nthr_, nthr_mb_, nthr_oc_b_, nthr_ic_b_, nthr_od_, nthr_oh_, nthr_ow_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
