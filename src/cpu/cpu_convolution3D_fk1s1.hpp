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

#ifndef CPU_CONVOLUTION3D_FK1S1_HPP
#define CPU_CONVOLUTION3D_FK1S1_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include <iostream>

namespace mkldnn {
namespace impl {
namespace cpu {

template <bool with_relu, impl::data_type_t src_type,
         impl::data_type_t wei_type = src_type,
         impl::data_type_t dst_type = src_type,
         impl::data_type_t acc_type = dst_type>
struct _convolution3D_fk1s1_fwd_t: public cpu_primitive_t {
    struct pd_t: public _cpu_convolution_fwd_pd_t<with_relu> {
        pd_t(engine_t *engine,
                const typename pd_t::base_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : _cpu_convolution_fwd_pd_t<with_relu>(engine, adesc, attr,
                    hint_fwd_pd)
        {}

        DECLARE_COMMON_PD_T(_convolution3D_fk1s1_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace data_type;
            assert(this->engine()->kind() == engine_kind::cpu);
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
                && this->cdesc_().diff_src_desc.dims[1] % 16 == 0
                && utils::one_of(this->cdesc_().src_desc.format, memory_format::nCdhw16c, memory_format::any)
                && utils::one_of(this->cdesc_().dst_desc.format, memory_format::ncdhw, memory_format::any)
                && utils::one_of(this->cdesc_().weights_desc.format, memory_format::oIdhw16i, memory_format::any)
                && KD() == 1 && KH() == 1 && KW() == 1
                && KSD() == 1 && KSH() == 1 && KSW() == 1
                && KDD() == 0 && KDH() == 0 && KDW() == 0
                && padT() == 0 && padL() == 0 && padD1() == 0
                && G() == 1
                && utils::implication(this->with_bias(), true
                        && utils::implication(src_type == u8,
                            utils::one_of(this->cdesc_().bias_desc.data_type,
                                f32, s32, s8, u8))
                        && utils::implication(src_type == f32,
                            this->cdesc_().bias_desc.data_type == f32))
                && this->attr()->has_default_values();
//         std::cout << "G=" << G() << " MB=" << MB()
//               << " OC=" << OC() << " OD=" << OD() << " OH=" << OH() << " OW=" << OW()
//               << " IC=" << IC() << " ID=" << ID() << " IH=" << IH() << " IW=" << OW()
//               << " KD=" << KD() << " KH=" << KH() << " KW=" << KW()
//               << " KSD=" << KSD() << " KSH=" << KSH() << " KSD=" << KSD()
//               << " KDD=" << KDD() << " KDH=" << KDH() << " KDD=" << KDD()
//               << " padT=" << padT() << " padL" << padL() << " padD1=" << padD1()
//               << " ok=" << ok << std::endl;
            return ok ? status::success : status::unimplemented;
        }

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

            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(nCdhw16c));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(ncdhw));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(oIdhw16i));
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            return status::success;
        }

    };

    _convolution3D_fk1s1_fwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {}

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
};

template <impl::data_type_t src_type, impl::data_type_t wei_type = src_type,
         impl::data_type_t dst_type = src_type,
         impl::data_type_t acc_type = dst_type>
using cpu_convolution3D_fk1s1_fwd_t = _convolution3D_fk1s1_fwd_t<false, src_type, wei_type,
      dst_type, acc_type>;

template <impl::data_type_t src_type, impl::data_type_t wei_type = src_type,
         impl::data_type_t dst_type = src_type,
         impl::data_type_t acc_type = dst_type>
using cpu_convolution3D_fk1s1_relu_t = _convolution3D_fk1s1_fwd_t<true, src_type, wei_type,
      dst_type, acc_type>;

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
