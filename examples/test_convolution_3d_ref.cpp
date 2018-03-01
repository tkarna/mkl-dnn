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

#include <iostream>
#include <numeric>
#include <string>
#include <cmath>
#include "mkldnn.hpp"
#include <iomanip>

using namespace mkldnn;

void print_array_3d(std::string name, float* array, int n, int m, int l) {
    std::cout << name << ":" << std::endl;
    for (int i=0; i<n; i++){
        for (int j=0; j<m; j++) {
            for (int k=0; k<l; k++) {
                std::cout << std::setw(5) << array[m*l*i + l*j + k];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

bool check_result(std::string array_name, float* array, float* correct,
                  const int len, float tolerance, bool verbose=false) {
    /* Computes the maximum abs relative error in the output array */
    float rel_error = 0;
    int nerr = 0;
    for (int i = 0; i < len; i++) {
        float re = (array[i] - correct[i])/correct[i];
        if (std::abs(re) > tolerance) {
            ++nerr;
            if (verbose)
            printf(" i=%d res=%.4f cor=%.4f rel_err=%.4g\n", i, array[i], correct[i], re);
        }
        rel_error = std::max(rel_error, std::abs(re));
    }
    bool success =  rel_error < tolerance;
    std::cout << "Test " << array_name << ": ";
    if (success) {
        std::cout << "OK" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
        std::cout << "  Len: " << len << "  Nerrs: " << nerr;
        std::cout << "  Relative error: " << rel_error << std::endl;
    }
    return success;
}

inline size_t ptr_off_f(const memory::desc &md, int mb, int g, int ic, int id, int ih, int iw) {
    const int G = 1; // NOTE currently without groups
    return ((((size_t)mb * md.data.dims[1] +
              g * md.data.dims[1]/G + ic) * md.data.dims[2] + id) *
              md.data.dims[3] + ih) * md.data.dims[4] + iw;
}

void compute_reference_fwd_conv(const memory &src_mem,
                                const memory &wei_mem,
                                const memory &bias_mem,
                                const memory &dst_mem,
                                const memory::dims &strides,
                                const memory::dims &dilation,
                                const memory::dims &padding) {
    // NOTE currently without relu and groups
    // NOTE currently only float data type
    const int G = 1;

    float *src = (float*)src_mem.get_data_handle();
    float *dst = (float*)dst_mem.get_data_handle();
    float *wei = (float*)wei_mem.get_data_handle();
    float *bias = (float*)bias_mem.get_data_handle();

    auto src_pd = src_mem.get_primitive_desc();
    auto dst_pd = dst_mem.get_primitive_desc();
    auto wei_pd = wei_mem.get_primitive_desc();

    auto src_md = src_pd.desc();
    auto dst_md = dst_pd.desc();
    auto wei_md = wei_pd.desc();

    // assuming ncdhw or oidhw layout
    assert(src_md.data.ndims == 5);
    assert(dst_md.data.ndims == 5);
    assert(wei_md.data.ndims == 5);

    const int MB = src_md.data.dims[0];
    const int IC = src_md.data.dims[1];
    const int ID = src_md.data.dims[2];
    const int IH = src_md.data.dims[3];
    const int IW = src_md.data.dims[4];

    const int OC = dst_md.data.dims[1];
    const int OD = dst_md.data.dims[2];
    const int OH = dst_md.data.dims[3];
    const int OW = dst_md.data.dims[4];

    const int KD = wei_md.data.dims[2];
    const int KH = wei_md.data.dims[3];
    const int KW = wei_md.data.dims[4];

    const int KSD = strides[0];
    const int KSH = strides[1];
    const int KSW = strides[2];

    const int KDD = dilation[0];
    const int KDH = dilation[1];
    const int KDW = dilation[2];

    const int padD = padding[0];
    const int padT = padding[1];
    const int padL = padding[2];

    auto ker = [=](float &d, int g, int mb, int oc, int od, int oh, int ow) {
        for (int ic = 0; ic < IC; ++ic) {
            for (int kd = 0; kd < KD; ++kd) {
                for (int kh = 0; kh < KH; ++kh) {
                    for (int kw = 0; kw < KW; ++kw) {

                        const int id = od * KSD - padD + kd * (1 + KDD);
                        const int ih = oh * KSH - padT + kh * (1 + KDH);
                        const int iw = ow * KSW - padL + kw * (1 + KDW);

                        if (id < 0 || id >= ID) continue;
                        if (ih < 0 || ih >= IH) continue;
                        if (iw < 0 || iw >= IW) continue;

                        d += (float)src[ptr_off_f(src_md, mb, 0, ic, id, ih, iw)]
                            * wei[ptr_off_f(wei_md, oc, 0, ic, kd, kh, kw)];
                    }
                }
            }
        }
    };

#   pragma omp parallel for collapse(5) schedule(static)
    for (int g = 0; g < G; ++g) {
        for (int mb = 0; mb < MB; ++mb) {
            for (int oc = 0; oc < OC; ++oc) {
                for (int od = 0; od < OD; ++od) {
                    for (int oh = 0; oh < OH; ++oh) {
                        for (int ow = 0; ow < OW; ++ow) {
                            float a = bias[g*OC + oc];
                            ker(a, g, mb, oc, od, oh, ow);
                            // NOTE omitting saturation in assignment
                            dst[ptr_off_f(dst_md, mb, 0, oc, od, oh, ow)] = a;
                        }
                    }
                }
            }
        }
    }

}

void compute_reference_bkw_weights_conv(const memory &src_mem,
                                        const memory &diff_dst_mem,
                                        const memory &diff_wei_mem,
                                        const memory &diff_bias_mem,
                                        const memory::dims &strides,
                                        const memory::dims &dilation,
                                        const memory::dims &padding) {
    // NOTE currently without relu and groups
    // NOTE currently only float data type
    const int G = 1;

    float *src = (float*)src_mem.get_data_handle();
    float *diff_dst = (float*)diff_dst_mem.get_data_handle();
    float *diff_wei = (float*)diff_wei_mem.get_data_handle();
    float *diff_bias = (float*)diff_bias_mem.get_data_handle();

    auto src_pd = src_mem.get_primitive_desc();
    auto diff_dst_pd = diff_dst_mem.get_primitive_desc();
    auto diff_wei_pd = diff_wei_mem.get_primitive_desc();

    auto src_md = src_pd.desc();
    auto diff_dst_md = diff_dst_pd.desc();
    auto diff_wei_md = diff_wei_pd.desc();

    // assuming ncdhw or oidhw layout
    assert(src_md.data.ndims == 5);
    assert(diff_dst_md.data.ndims == 5);
    assert(diff_wei_md.data.ndims == 5);

    const int MB = src_md.data.dims[0];
    const int IC = src_md.data.dims[1];
    const int ID = src_md.data.dims[2];
    const int IH = src_md.data.dims[3];
    const int IW = src_md.data.dims[4];

    const int OC = diff_dst_md.data.dims[1];
    const int OD = diff_dst_md.data.dims[2];
    const int OH = diff_dst_md.data.dims[3];
    const int OW = diff_dst_md.data.dims[4];

    const int KD = diff_wei_md.data.dims[2];
    const int KH = diff_wei_md.data.dims[3];
    const int KW = diff_wei_md.data.dims[4];

    const int KSD = strides[0];
    const int KSH = strides[1];
    const int KSW = strides[2];

    const int KDD = dilation[0];
    const int KDH = dilation[1];
    const int KDW = dilation[2];

    const int padD = padding[0];
    const int padT = padding[1];
    const int padL = padding[2];

    auto ker = [=](float &d, int g, int oc, int ic, int kd, int kh, int kw) {
        for (int mb = 0; mb < MB; ++mb) {
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        if (ow*KSW + kw * (1 + KDW) < padL
                                || oh*KSH + kh * (1 + KDH) < padT
                                || od*KSD + kd * (1 + KDD) < padD
                                || ow*KSW + kw * (1 + KDW) >= IW + padL
                                || oh*KSH + kh * (1 + KDH) >= IH + padT
                                || od*KSD + kd * (1 + KDD) >= ID + padD)
                            continue;

                        int id = od*KSD - padD + kd * (1 + KDD);
                        int ih = oh*KSH - padT + kh * (1 + KDH);
                        int iw = ow*KSW - padL + kw * (1 + KDW);

                        d += (float)diff_dst[ptr_off_f(diff_dst_md, mb, 0, g*OC + oc, od,
                                oh, ow)] * src[ptr_off_f(src_md, mb, 0, g*IC + ic, id, ih, iw)];
                    }
                }
            }
        }
    };

    auto ker_bias = [=](float &d, int g, int oc) {
        for (int mb = 0; mb < MB; ++mb) {
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        d += (float)diff_dst[ptr_off_f(diff_dst_md, mb, 0, g*OC + oc, od, oh, ow)];
                    }
                }
            }
        }
    };

#   pragma omp parallel for collapse(2) schedule(static)
    for (int g = 0; g < G; ++g) {
        for (int oc = 0; oc < OC; ++oc) {
            if (diff_bias) {
                float db = 0;
                ker_bias(db, g, oc);
                // NOTE assuming linear index
                diff_bias[g*OC+oc] = db;
            }

            for (int ic = 0; ic < IC; ++ic) {
                for (int kd = 0; kd < KD; ++kd) {
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            float dw = 0;
                            ker(dw, g, oc, ic, kd, kh, kw);

                            auto idx = ptr_off_f(diff_wei_md, oc, 0, ic, kd, kh, kw);
                            diff_wei[idx] = dw;
                        }
                    }
                }
            }
        }
    }
}

void compute_reference_bkw_data_conv(const memory &diff_dst_mem,
                                     const memory &wei_mem,
                                     const memory &diff_src_mem,
                                     const memory::dims &strides,
                                     const memory::dims &dilation,
                                     const memory::dims &padding) {
    // NOTE currently without relu, bias and groups
    // NOTE currently only float data type
    const int G = 1;

    float *diff_src = (float*)diff_src_mem.get_data_handle();
    float *diff_dst = (float*)diff_dst_mem.get_data_handle();
    float *wei = (float*)wei_mem.get_data_handle();

    auto diff_src_pd = diff_src_mem.get_primitive_desc();
    auto diff_dst_pd = diff_dst_mem.get_primitive_desc();
    auto wei_pd = wei_mem.get_primitive_desc();

    auto diff_src_md = diff_src_pd.desc();
    auto diff_dst_md = diff_dst_pd.desc();
    auto wei_md = wei_pd.desc();

    // assuming ncdhw or oidhw layout
    assert(diff_src_md.data.ndims == 5);
    assert(diff_dst_md.data.ndims == 5);
    assert(wei_md.data.ndims == 5);

    const int MB = diff_src_md.data.dims[0];
    const int IC = diff_src_md.data.dims[1];
    const int ID = diff_src_md.data.dims[2];
    const int IH = diff_src_md.data.dims[3];
    const int IW = diff_src_md.data.dims[4];

    const int OC = diff_dst_md.data.dims[1];
    const int OD = diff_dst_md.data.dims[2];
    const int OH = diff_dst_md.data.dims[3];
    const int OW = diff_dst_md.data.dims[4];

    const int KD = wei_md.data.dims[2];
    const int KH = wei_md.data.dims[3];
    const int KW = wei_md.data.dims[4];

    const int KSD = strides[0];
    const int KSH = strides[1];
    const int KSW = strides[2];

    const int KDD = dilation[0];
    const int KDH = dilation[1];
    const int KDW = dilation[2];

    const int padD = padding[0];
    const int padT = padding[1];
    const int padL = padding[2];

    auto ker = [=](float &d, int g, int mb, int ic, int id, int ih, int iw) {
        for (int oc = 0; oc < OC; ++oc) {
            for (int kd = 0; kd < KD; ++kd) {
                for (int kh = 0; kh < KH; ++kh) {
                    for (int kw = 0; kw < KW; ++kw) {
                        if (iw + padL < kw * (1 + KDW)
                            || ih + padT < kh * (1 + KDH)
                            || id + padD < kd * (1 + KDD))
                            continue;
                        int od = id - kd * (1 + KDD) + padD;
                        int oh = ih - kh * (1 + KDH) + padT;
                        int ow = iw - kw * (1 + KDW) + padL;
                        if (ow % KSW != 0 || oh % KSH != 0 || od % KSD != 0)
                            continue;

                        od /= KSD;
                        oh /= KSH;
                        ow /= KSW;

                        if (oh < OH && ow < OW && od < OD) {
                            d += (float)diff_dst[ptr_off_f(diff_dst_md, mb, g, g*OC + oc, od, oh, ow)] *
                             wei[ptr_off_f(wei_md, oc, 0, ic, kd, kh, kw)];
                        }
                    }
                }
            }
        }
    };

#   pragma omp parallel for collapse(5) schedule(static)
    for (int g = 0; g < G; ++g) {
        for (int mb = 0; mb < MB; ++mb) {
            for (int ic = 0; ic < IC; ++ic) {
                for (int id = 0; id < ID; ++id) {
                    for (int ih = 0; ih < IH; ++ih) {
                        for (int iw = 0; iw < IW; ++iw) {
                            auto idx = ptr_off_f(diff_src_md, mb, g, g*IC + ic, id, ih, iw);
                            float a = float(0);
                            ker(a, g, mb, ic, id, ih, iw);
                            diff_src[idx] = a;
                        }
                    }
                }
            }
        }
    }
}

void compute_fwd_conv(const memory &src_mem,
                      const memory &weights_mem,
                      const memory &bias_mem,
                      const memory &dst_mem,
                      const memory::dims &strides,
                      const memory::dims &dilation,
                      const memory::dims &padding) {

    auto cpu_engine = engine(engine::cpu, 0);

    auto src_pd = src_mem.get_primitive_desc();
    auto weights_pd = weights_mem.get_primitive_desc();
    auto bias_pd = bias_mem.get_primitive_desc();
    auto dst_pd = dst_mem.get_primitive_desc();

    auto src_md = src_pd.desc();
    auto weights_md = weights_pd.desc();
    auto bias_md = bias_pd.desc();
    auto dst_md = dst_pd.desc();

    // define memory descriptors for fwd op
    auto src_dims = {src_md.data.dims[0], src_md.data.dims[1],
        src_md.data.dims[2], src_md.data.dims[3], src_md.data.dims[4]};
    auto weights_dims = {weights_md.data.dims[0], weights_md.data.dims[1],
        weights_md.data.dims[2], weights_md.data.dims[3], weights_md.data.dims[4]};
    auto dst_dims = {dst_md.data.dims[0], dst_md.data.dims[1],
        dst_md.data.dims[2], dst_md.data.dims[3], dst_md.data.dims[4]};
    auto src_any_md = memory::desc(src_dims, memory::data_type::f32,
                                   memory::format::any);
    auto weights_any_md = memory::desc(weights_dims, memory::data_type::f32,
                                       memory::format::any);
    auto dst_any_md = memory::desc(dst_dims, memory::data_type::f32,
                                   memory::format::any);

    /* op descriptors */
    auto conv_fwd_desc = convolution_forward::desc(prop_kind::forward,
        convolution_direct, src_any_md, weights_any_md, bias_md,
        dst_any_md, strides, padding, padding,
        padding_kind::zero, conv_kind::conv3D);

    /* primitive op descriptors */
    auto conv_fwd_pd =
        convolution_forward::primitive_desc(conv_fwd_desc, cpu_engine);

    auto src_fmt = src_md.data.format;
    auto conv_src_fmt = conv_fwd_pd.src_primitive_desc().desc().data.format;
    bool src_needs_reorder = conv_src_fmt != src_fmt;

    auto weights_fmt = weights_md.data.format;
    auto conv_weights_fmt = conv_fwd_pd.weights_primitive_desc().desc().data.format;
    bool weights_needs_reorder = conv_weights_fmt != weights_fmt;

    auto dst_fmt = dst_md.data.format;
    auto conv_dst_fmt = conv_fwd_pd.dst_primitive_desc().desc().data.format;
    bool dst_needs_reorder = conv_dst_fmt != dst_fmt;

    /* create reorders between user and data if it is needed and
     *  add it to net before convolution */
    auto reorder_src_mem = src_mem;
    if (src_needs_reorder) {
        reorder_src_mem = memory(conv_fwd_pd.src_primitive_desc());
    }
    auto reorder_weights_mem = weights_mem;
    if (weights_needs_reorder) {
        reorder_weights_mem = memory(conv_fwd_pd.weights_primitive_desc());
    }

    auto reorder_dst_mem = dst_mem;
    if (dst_needs_reorder) {
        reorder_dst_mem = memory(conv_fwd_pd.dst_primitive_desc());
    }

    /* create forward convolution primitive */
    auto conv_op = convolution_forward(conv_fwd_pd, reorder_src_mem,
        reorder_weights_mem, bias_mem, reorder_dst_mem);

    // create network array
    std::vector<primitive> net;

    if (src_needs_reorder) {
        net.push_back(reorder(src_mem, reorder_src_mem));
    }
    if (weights_needs_reorder) {
        net.push_back(reorder(weights_mem, reorder_weights_mem));
    }

    // push to net
    net.push_back(conv_op);

    if (dst_needs_reorder) {
        net.push_back(reorder(reorder_dst_mem, dst_mem));
    }

    // Execute
    stream(stream::kind::eager).submit(net).wait();
}

void compute_bkw_weights_conv(const memory &src_mem,
                              const memory &diff_dst_mem,
                              const memory &diff_weights_mem,
                              const memory &diff_bias_mem,
                              const memory::dims &strides,
                              const memory::dims &dilation,
                              const memory::dims &padding) {
    auto cpu_engine = engine(engine::cpu, 0);

    auto src_pd = src_mem.get_primitive_desc();
    auto diff_dst_pd = diff_dst_mem.get_primitive_desc();
    auto diff_weights_pd = diff_weights_mem.get_primitive_desc();
    auto diff_bias_pd = diff_bias_mem.get_primitive_desc();

    auto src_md = src_pd.desc();
    auto diff_dst_md = diff_dst_pd.desc();
    auto diff_weights_md = diff_weights_pd.desc();

    // define memory descriptors
    auto out_channels = diff_dst_md.data.dims[1];
    memory::dims bias_dims = {out_channels};
    auto src_dims = {src_md.data.dims[0], src_md.data.dims[1],
        src_md.data.dims[2], src_md.data.dims[3], src_md.data.dims[4]};
    auto dst_dims = {diff_dst_md.data.dims[0], diff_dst_md.data.dims[1],
        diff_dst_md.data.dims[2], diff_dst_md.data.dims[3], diff_dst_md.data.dims[4]};
    auto weights_dims = {diff_weights_md.data.dims[0], diff_weights_md.data.dims[1],
        diff_weights_md.data.dims[2], diff_weights_md.data.dims[3], diff_weights_md.data.dims[4]};
    auto src_any_md = memory::desc(src_dims, memory::data_type::f32,
                                   memory::format::any);
    auto weights_any_md = memory::desc(weights_dims, memory::data_type::f32,
                                       memory::format::any);
    auto bias_any_md = memory::desc(bias_dims, memory::data_type::f32,
                                     memory::format::any);
    auto dst_any_md = memory::desc(dst_dims, memory::data_type::f32,
                                   memory::format::any);

    /* op descriptors */
    auto fwd_desc = convolution_forward::desc(prop_kind::forward,
        convolution_direct, src_any_md, weights_any_md, bias_any_md,
        dst_any_md, strides, dilation, padding, padding,
        padding_kind::zero, conv_kind::conv3D);
    auto bkww_desc = mkldnn::convolution_backward_weights::desc(
        mkldnn::convolution_direct, src_any_md, weights_any_md, bias_any_md,
        dst_any_md, strides, dilation, padding, padding,
        mkldnn::padding_kind::zero, conv_kind::conv3D);

    /* primitive op descriptors */
    auto fwd_pd =
        convolution_forward::primitive_desc(fwd_desc, cpu_engine);
    auto bkww_pd =
        convolution_backward_weights::primitive_desc(bkww_desc, cpu_engine, fwd_pd);

    auto src_fmt = src_md.data.format;
    auto conv_src_fmt = bkww_pd.src_primitive_desc().desc().data.format;
    bool src_needs_reorder = conv_src_fmt != src_fmt;

    auto weights_fmt = diff_weights_md.data.format;
    auto conv_weights_fmt = bkww_pd.diff_weights_primitive_desc().desc().data.format;
    bool weights_needs_reorder = conv_weights_fmt != weights_fmt;

    auto dst_fmt = diff_dst_md.data.format;
    auto conv_dst_fmt = bkww_pd.diff_dst_primitive_desc().desc().data.format;
    bool dst_needs_reorder = conv_dst_fmt != dst_fmt;

    // NOTE if implemented correctly bias should also need reorder

    /* create reorders between user and data if it is needed and
     *  add it to net before convolution */
    auto reorder_src_mem = src_mem;
    if (src_needs_reorder) {
        reorder_src_mem = memory(bkww_pd.src_primitive_desc());
    }
    auto reorder_diff_weights_mem = diff_weights_mem;
    if (weights_needs_reorder) {
        reorder_diff_weights_mem = memory(bkww_pd.diff_weights_primitive_desc());
    }

    auto reorder_diff_dst_mem = diff_dst_mem;
    if (dst_needs_reorder) {
        reorder_diff_dst_mem = memory(bkww_pd.diff_dst_primitive_desc());
    }

    /* create backward weights convolution primitive */
    auto bkww_op = convolution_backward_weights(bkww_pd,
        reorder_src_mem, reorder_diff_dst_mem, reorder_diff_weights_mem, diff_bias_mem);

    // create network array
    std::vector<primitive> net;

    if (src_needs_reorder) {
        net.push_back(reorder(src_mem, reorder_src_mem));
    }
    if (dst_needs_reorder) {
        net.push_back(reorder(diff_dst_mem, reorder_diff_dst_mem));
    }

    // push to net
    net.push_back(bkww_op);

    if (weights_needs_reorder) {
        net.push_back(reorder(reorder_diff_weights_mem, diff_weights_mem));
    }
    // execute
    stream(stream::kind::eager).submit(net).wait();
}

void compute_bkw_data_conv(const memory &diff_dst_mem,
                           const memory &weights_mem,
                           const memory &diff_src_mem,
                           const memory::dims &strides,
                           const memory::dims &dilation,
                           const memory::dims &padding) {
    auto cpu_engine = engine(engine::cpu, 0);

    auto diff_src_pd = diff_src_mem.get_primitive_desc();
    auto diff_dst_pd = diff_dst_mem.get_primitive_desc();
    auto weights_pd = weights_mem.get_primitive_desc();

    auto diff_src_md = diff_src_pd.desc();
    auto diff_dst_md = diff_dst_pd.desc();
    auto weights_md = weights_pd.desc();

    // define memory descriptors for fwd op
    auto out_channels = diff_dst_md.data.dims[1];
    memory::dims bias_dims = {out_channels};
    auto src_dims = {diff_src_md.data.dims[0], diff_src_md.data.dims[1],
        diff_src_md.data.dims[2], diff_src_md.data.dims[3], diff_src_md.data.dims[4]};
    auto dst_dims = {diff_dst_md.data.dims[0], diff_dst_md.data.dims[1],
        diff_dst_md.data.dims[2], diff_dst_md.data.dims[3], diff_dst_md.data.dims[4]};
    auto weights_dims = {weights_md.data.dims[0], weights_md.data.dims[1],
        weights_md.data.dims[2], weights_md.data.dims[3], weights_md.data.dims[4]};
    auto src_any_md = memory::desc(src_dims, memory::data_type::f32,
                                   memory::format::any);
    auto weights_any_md = memory::desc(weights_dims, memory::data_type::f32,
                                       memory::format::any);
    auto bias_any_md = memory::desc(bias_dims, memory::data_type::f32,
                                     memory::format::any);
    auto dst_any_md = memory::desc(dst_dims, memory::data_type::f32,
                                   memory::format::any);

    /* op descriptors */
    auto fwd_desc = convolution_forward::desc(prop_kind::forward,
        convolution_direct, src_any_md, weights_md, bias_any_md,
        dst_any_md, strides, dilation, padding, padding,
        padding_kind::zero, conv_kind::conv3D);
    auto bkwd_desc = convolution_backward_data::desc(
        convolution_direct, src_any_md, weights_any_md,
        dst_any_md, strides, dilation, padding, padding,
        padding_kind::zero, conv_kind::conv3D);

    /* primitive op descriptors */
    // TODO is fwd pd needed?
    auto fwd_pd =
        convolution_forward::primitive_desc(fwd_desc, cpu_engine);
    auto bkwd_pd =
        convolution_backward_data::primitive_desc(bkwd_desc, cpu_engine, fwd_pd);

    auto src_fmt = diff_src_md.data.format;
    auto conv_src_fmt = bkwd_pd.diff_src_primitive_desc().desc().data.format;
    bool src_needs_reorder = conv_src_fmt != src_fmt;

    auto weights_fmt = weights_md.data.format;
    auto conv_weights_fmt = bkwd_pd.weights_primitive_desc().desc().data.format;
    bool weights_needs_reorder = conv_weights_fmt != weights_fmt;

    auto dst_fmt = diff_dst_md.data.format;
    auto conv_dst_fmt = bkwd_pd.diff_dst_primitive_desc().desc().data.format;
    bool dst_needs_reorder = conv_dst_fmt != dst_fmt;

    /* create reorders between user and data if it is needed and
     *  add it to net before convolution */
    auto reorder_diff_src_mem = diff_src_mem;
    if (src_needs_reorder) {
        reorder_diff_src_mem = memory(bkwd_pd.diff_src_primitive_desc());
    }
    auto reorder_weights_mem = weights_mem;
    if (weights_needs_reorder) {
        reorder_weights_mem = memory(bkwd_pd.weights_primitive_desc());
    }

    auto reorder_diff_dst_mem = diff_dst_mem;
    if (dst_needs_reorder) {
        reorder_diff_dst_mem = memory(bkwd_pd.diff_dst_primitive_desc());
    }

    /* create backward data convolution primitive */
    auto bkwd_op = convolution_backward_data(bkwd_pd,
        reorder_diff_dst_mem, reorder_weights_mem, reorder_diff_src_mem);

    // TODO add reorders and checks
    // check if reorder is needed
    // if yes: allocate new memory, create reorder op, push

    // create network array
    std::vector<primitive> net;

    if (dst_needs_reorder) {
        net.push_back(reorder(diff_dst_mem, reorder_diff_dst_mem));
    }
    if (weights_needs_reorder) {
        net.push_back(reorder(weights_mem, reorder_weights_mem));
    }

    // push to net
    net.push_back(bkwd_op);

    if (src_needs_reorder) {
        net.push_back(reorder(reorder_diff_src_mem, diff_src_mem));
    }

    // execute
    stream(stream::kind::eager).submit(net).wait();
}

bool assert_fwd_convolution(const int nbatch, const int in_channels, const int out_channels,
                            const int in_height, const int in_width, const int in_depth,
                            const int weights_height, const int weights_width, const int weights_depth,
                            const int out_height, const int out_width, const int out_depth,
                            const int stride_d, const int stride_h, const int stride_w,
                            const int pad_d, const int pad_h, const int pad_w,
                            std::vector<float>& in_weights, std::vector<float>& in_bias,
                            float tolerance,
                            bool print_arrays = true){

    auto cpu_engine = engine(engine::cpu, 0);

    // Dimensions of memory to be allocated
    memory::dims src_dims = {nbatch, in_channels, in_depth, in_height, in_width};
    memory::dims weights_dims = {out_channels, in_channels, weights_depth, weights_height, weights_width};
    memory::dims bias_dims = {out_channels};
    memory::dims dst_dims = {nbatch, out_channels, out_depth, out_height, out_width};

    auto strides = {stride_d, stride_h, stride_w};
    auto padding = {pad_d, pad_h, pad_w};
    auto dilation = {0, 0, 0};

    std::vector<float> vect_src(std::accumulate(src_dims.begin(),
        src_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_weights(std::accumulate(weights_dims.begin(),
        weights_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_bias(std::accumulate(bias_dims.begin(),
        bias_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_dst(std::accumulate(dst_dims.begin(),
        dst_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_ref_dst(std::accumulate(dst_dims.begin(),
        dst_dims.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    auto src_memory = memory({{{src_dims},
                             memory::data_type::f32, memory::format::ncdhw},
                             cpu_engine}, vect_src.data());
    auto weights_memory = memory({{{weights_dims},
                                 memory::data_type::f32,memory::format::oidhw},
                                 cpu_engine}, vect_weights.data());
    auto bias_memory = memory({{{bias_dims},
                              memory::data_type::f32, memory::format::x},
                              cpu_engine}, vect_bias.data());
    auto dst_memory = memory({{{dst_dims},
                             memory::data_type::f32, memory::format::ncdhw},
                             cpu_engine}, vect_dst.data());
    auto ref_dst_memory = memory({{{dst_dims},
                                 memory::data_type::f32, memory::format::ncdhw},
                                 cpu_engine}, vect_ref_dst.data());

    // assign input and weights data
    for (size_t i = 0; i < vect_src.size(); i++)
        vect_src[i] = rand() % 25 + 1.0;

    vect_weights = in_weights;
    vect_bias = in_bias;

    bool success = true;

    /* Compute reference solution */
    compute_reference_fwd_conv(src_memory,
                               weights_memory,
                               bias_memory,
                               ref_dst_memory,
                               strides, dilation, padding);

    // Print the output matrix
    if (print_arrays) {
        printf("\n");
        print_array_3d("Input", vect_src.data(), src_dims[2], src_dims[3], src_dims[4]);
        print_array_3d("Kernel", vect_weights.data(), weights_dims[2], weights_dims[3], weights_dims[4]);
        print_array_3d("Reference output", vect_ref_dst.data(), dst_dims[2], dst_dims[3], dst_dims[4]);
    }

    compute_fwd_conv(src_memory,
                     weights_memory,
                     bias_memory,
                     dst_memory,
                     strides, dilation, padding);

    if (print_arrays) {
        // Print the output matrix
        print_array_3d("Output", vect_dst.data(), dst_dims[2], dst_dims[3], dst_dims[4]);
    }
    // Compute error
    success = success && check_result("output", vect_dst.data(), vect_ref_dst.data(), vect_ref_dst.size(), tolerance);

    return success;
}

bool assert_bkw_weights_convolution(const int nbatch, const int in_channels,
                                    const int out_channels, const int in_height,
                                    const int in_width, const int in_depth,
                                    const int weights_height,
                                    const int weights_width,
                                    const int weights_depth, const int out_height,
                                    const int out_width, const int out_depth,
                                    const int stride_d, const int stride_h, const int stride_w,
                                    const int pad_d, const int pad_h, const int pad_w,
                                    std::vector<float>& in_diff_dst,
                                    float tolerance,
                                    bool test_bias = true,
                                    bool print_arrays = true
                                   ){

    auto cpu_engine = engine(engine::cpu, 0);

    // Dimensions of memory to be allocated
    memory::dims src_dims = {nbatch, in_channels, in_depth, in_height, in_width};
    memory::dims weights_dims = {out_channels, in_channels, weights_depth, weights_height, weights_width};
    memory::dims bias_dims = {out_channels};
    memory::dims dst_dims = {nbatch, out_channels, out_depth, out_height, out_width};

    // allocate memory
    std::vector<float> vect_diff_dst(std::accumulate(dst_dims.begin(),
        dst_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_diff_weights(std::accumulate(weights_dims.begin(),
        weights_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_diff_bias(std::accumulate(bias_dims.begin(),
        bias_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_src(std::accumulate(src_dims.begin(),
        src_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_ref_diff_weights(std::accumulate(weights_dims.begin(),
        weights_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_ref_diff_bias(std::accumulate(bias_dims.begin(),
        bias_dims.end(), 1, std::multiplies<uint32_t>()));

    auto strides = {stride_d, stride_h, stride_w};
    auto padding = {pad_d, pad_h, pad_w};
    auto dilation = {0, 0, 0};

    auto diff_dst_memory = memory({{{dst_dims}, memory::data_type::f32, memory::format::ncdhw}, cpu_engine}, vect_diff_dst.data());
    auto diff_weights_memory = memory({{{weights_dims}, memory::data_type::f32,memory::format::oidhw}, cpu_engine}, vect_diff_weights.data());
    auto diff_bias_memory = memory({{{bias_dims}, memory::data_type::f32, memory::format::x}, cpu_engine}, vect_diff_bias.data());
    auto src_memory = memory({{{src_dims}, memory::data_type::f32, memory::format::ncdhw}, cpu_engine}, vect_src.data());
    auto ref_diff_weights_memory = memory({{{weights_dims}, memory::data_type::f32, memory::format::oidhw}, cpu_engine}, vect_ref_diff_weights.data());
    auto ref_diff_bias_memory = memory({{{bias_dims}, memory::data_type::f32, memory::format::x}, cpu_engine}, vect_ref_diff_bias.data());


    // assign input and weights data
    for (size_t i = 0; i < vect_src.size(); i++)
        vect_src[i] = rand() % 25 + 1.0;

    vect_diff_dst = in_diff_dst;

    bool success = true;

    /* Compute reference solution */
    compute_reference_bkw_weights_conv(src_memory,
                                       diff_dst_memory,
                                       ref_diff_weights_memory,
                                       ref_diff_bias_memory,
                                       strides, dilation, padding);

    compute_bkw_weights_conv(src_memory,
                             diff_dst_memory,
                             diff_weights_memory,
                             diff_bias_memory,
                             strides, dilation, padding);

    if (print_arrays) {
        print_array_3d("Input", vect_src.data(), src_dims[2], src_dims[3], src_dims[4]);
        print_array_3d("Diff output", vect_diff_dst.data(), dst_dims[2], dst_dims[3], dst_dims[4]);
        print_array_3d("Diff weights", vect_diff_weights.data(), weights_dims[2], weights_dims[3], weights_dims[4]);
        print_array_3d("Correct diff weights", vect_ref_diff_weights.data(), weights_dims[2], weights_dims[3], weights_dims[4]);
    }

    if (test_bias) {
        if (print_arrays) {
            print_array_3d("Diff bias", vect_diff_bias.data(), 1, 1, bias_dims[0]);
            print_array_3d("Correct diff bias", vect_ref_diff_bias.data(), 1, 1, bias_dims[0]);
        }
        success = success && check_result("diff bias", vect_diff_bias.data(), vect_ref_diff_bias.data(), vect_diff_bias.size(), tolerance);
    }
    success = success && check_result("diff weight", vect_diff_weights.data(), vect_ref_diff_weights.data(), vect_diff_weights.size(), tolerance);

    return success;
}

bool assert_bkw_data_convolution(const int nbatch, const int in_channels,
                                 const int out_channels, const int in_height,
                                 const int in_width, const int in_depth,
                                 const int weights_height,
                                 const int weights_width,
                                 const int weights_depth, const int out_height,
                                 const int out_width, const int out_depth,
                                 const int stride_d, const int stride_h, const int stride_w,
                                 const int pad_d, const int pad_h, const int pad_w,
                                 std::vector<float>& in_weights,
                                 float tolerance,
                                 bool print_arrays = true
                                ){

    auto cpu_engine = engine(engine::cpu, 0);

    // Dimensions of memory to be allocated
    memory::dims src_dims = {nbatch, in_channels, in_depth, in_height, in_width};
    memory::dims weights_dims = {out_channels, in_channels, weights_depth, weights_height, weights_width};
    memory::dims dst_dims = {nbatch, out_channels, out_depth, out_height, out_width};

    // allocate memory
    std::vector<float> vect_diff_dst(std::accumulate(dst_dims.begin(),
        dst_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_weights(std::accumulate(weights_dims.begin(),
        weights_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_diff_src(std::accumulate(src_dims.begin(),
        src_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_ref_diff_src(std::accumulate(src_dims.begin(),
        src_dims.end(), 1, std::multiplies<uint32_t>()));

    auto strides = {stride_d, stride_h, stride_w};
    auto padding = {pad_d, pad_h, pad_w};
    auto dilation = {0, 0, 0};

    auto diff_dst_memory = memory({{{dst_dims}, memory::data_type::f32, memory::format::ncdhw}, cpu_engine}, vect_diff_dst.data());
    auto weights_memory = memory({{{weights_dims}, memory::data_type::f32,memory::format::oidhw}, cpu_engine}, vect_weights.data());
    auto diff_src_memory = memory({{{src_dims}, memory::data_type::f32, memory::format::ncdhw}, cpu_engine}, vect_diff_src.data());
    auto ref_diff_src_memory = memory({{{src_dims}, memory::data_type::f32, memory::format::ncdhw}, cpu_engine}, vect_ref_diff_src.data());


    // assign input and weights data
    for (int mb = 0; mb < nbatch; mb++) {
    for (int c = 0; c < out_channels; c++) {
    for (int i = 0; i < out_depth; i++) {
        for (int j = 0; j < out_height; j++) {
            for (int k = 0; k < out_width; k++) {
                const size_t ix = (((mb*out_channels + c)*out_depth + i)*out_height + j)*out_width + k;
                vect_diff_dst[ix] = (i+1)*(j+1)*(k+1);
            }
        }
    }
    }}
    vect_weights = in_weights;

    bool success = true;

    // create network array
    std::vector<primitive> net;

    /* Compute reference solution */
    compute_reference_bkw_data_conv(diff_dst_memory,
                                    weights_memory,
                                    ref_diff_src_memory,
                                    strides, dilation, padding);

    compute_bkw_data_conv(diff_dst_memory,
                          weights_memory,
                          diff_src_memory,
                          strides, dilation, padding);

    if (print_arrays) {
        print_array_3d("Kernel", vect_weights.data(), weights_dims[2], weights_dims[3], weights_dims[4]);
        print_array_3d("Diff output", vect_diff_dst.data(), dst_dims[2], dst_dims[3], dst_dims[4]);
        print_array_3d("Reference diff input", vect_ref_diff_src.data(), src_dims[2], src_dims[3], src_dims[4]);
        print_array_3d("Diff input", vect_diff_src.data(), src_dims[2], src_dims[3], src_dims[4]);
    }

    success = success && check_result("diff src", vect_diff_src.data(), vect_ref_diff_src.data(), vect_diff_src.size(), tolerance);

    return success;
}

bool test_fwd_simple(const int ic=1, const int oc=1) {
    printf("FWD  simple IC=%d OC=%d ", ic, oc);
    const int bs=1;
    const int ih=5, iw=5, id=4;
    const int oh=3, ow=3, od=2;
    const int kh=3, kw=3, kd=3;
    int weights_len = oc*ic*kh*kw*kd;
    std::vector<float> bias(oc, 0);
    std::vector<float> weights(weights_len, 0);
    for (int o = 0; o < oc; o++) {
        for (int i = 0; i < 1; i++) {
            const int off = kd*kh*kw;
            weights[o*ic*off + i*off + off-1] = 1;
        }
    }
    return assert_fwd_convolution(bs, ic, oc, ih, iw, id, kh, kw, kd, oh, ow, od,
                              1, 1, 1, 0, 0, 0, weights, bias, 1e-25);
}

bool test_fwd_full(const int bs, std::vector<int> insize, std::vector<int> kernel,
               const int ic, const int oc, std::vector<int> stride,
               std::vector<int> pad, bool fill_with_floats=true) {
    auto float_str = fill_with_floats ? "flt" : "int";
    int ih = insize[0], iw = insize[1], id = insize[2];
    int kh = kernel[0], kw = kernel[1], kd = kernel[2];
    int sh = stride[0], sw = stride[1], sd = stride[2];
    int ph = pad[0], pw = pad[1], pd = pad[2];
    printf("FWD  bs=%d %dx%dx%d w=%dx%dx%d st=%dx%dx%d pd=%dx%dx%d ic=%2d oc=%2d %s ",
           bs, ih, iw, id,
           kh, kw, kd,
           sh, sw, sd,
           ph, pw, pd,
           ic, oc, float_str);
    const int oh=(ih-kh+2*ph)/sh+1, ow=(iw-kw+2*pw)/sw+1, od=(id-kd+2*pd)/sd+1;
    int weights_len = oc*ic*kh*kw*kd;
    std::vector<float> bias(oc, 0);
    for (size_t i = 0; i < bias.size(); i++)
        bias[i] = rand() % 8 + 1.0;
    std::vector<float> weights(weights_len, 0);
    if (fill_with_floats) {
        for (size_t i = 0; i < weights.size(); i++)
            weights[i] = (float)(rand() % 100)/11.2 + 1.0;
    } else {
        for (size_t i = 0; i < weights.size(); i++)
            weights[i] = rand() % 8 + 1.0;
    }
    float tolerance = fill_with_floats ? 1e-5 : 1e-25;
    return assert_fwd_convolution(bs, ic, oc, ih, iw, id, kh, kw, kd, oh, ow, od,
                              sh, sw, sd, ph, pw, pd,
                              weights, bias, tolerance, false);
}

bool test_bkw_weights_simple(const int ic=1, const int oc=1) {
    printf("\nBKWW simple IC=%d OC=%d\n", ic, oc);
    const int bs=1;
    const int ih=5, iw=5, id=4;
    const int oh=3, ow=3, od=2;
    const int kh=3, kw=3, kd=3;
    int out_len = bs*oc*oh*ow*od;
    std::vector<float> in_diff_dst(out_len, 0);
    for (int mb = 0; mb < bs; mb++) {
        for (int c = 0; c < oc; c++) {
            const int last_ix = oh*ow*od - 1;
            in_diff_dst[last_ix] = 1;
        }
    }
    bool test_bias = true;
    bool print_arrays = true;
    return assert_bkw_weights_convolution(bs, ic, oc, ih, iw, id, kh, kw, kd, oh, ow, od,
                              1, 1, 1, 0, 0, 0,
                              in_diff_dst, 1e-25, test_bias, print_arrays);
}

bool test_bkw_weights_full(const int bs, std::vector<int> insize, std::vector<int> kernel,
                           const int ic, const int oc, std::vector<int> stride,
                           std::vector<int> pad, bool fill_with_floats=true) {
    auto float_str = fill_with_floats ? "flt" : "int";
    int ih = insize[0], iw = insize[1], id = insize[2];
    int kh = kernel[0], kw = kernel[1], kd = kernel[2];
    int sh = stride[0], sw = stride[1], sd = stride[2];
    int ph = pad[0], pw = pad[1], pd = pad[2];
    printf("BKWW bs=%d %dx%dx%d w=%dx%dx%d st=%dx%dx%d pd=%dx%dx%d ic=%2d oc=%2d %s ",
           bs, ih, iw, id,
           kh, kw, kd,
           sh, sw, sd,
           ph, pw, pd,
           ic, oc, float_str);
    const int oh=(ih-kh+2*ph)/sh+1, ow=(iw-kw+2*pw)/sw+1, od=(id-kd+2*pd)/sd+1;
    int out_len = bs*oc*oh*ow*od;
    std::vector<float> in_diff_dst(out_len, 0);
    if (fill_with_floats) {
        for (size_t i = 0; i < in_diff_dst.size(); i++)
            in_diff_dst[i] = (float)(rand() % 100)/11.2 + 1.0;
    } else {
        for (size_t i = 0; i < in_diff_dst.size(); i++)
            in_diff_dst[i] = rand() % 8 + 1.0;
    }
    bool test_bias = true;
    bool print_arrays = false;
    float tolerance = fill_with_floats ? 1e-4 : 1e-25;
    return assert_bkw_weights_convolution(bs, ic, oc, ih, iw, id, kh, kw, kd, oh, ow, od,
                              sh, sw, sd, ph, pw, pd,
                              in_diff_dst, tolerance, test_bias, print_arrays);
}

bool test_bkw_data_simple(const int ic=1, const int oc=1) {
    printf("\nBKWD simple IC=%d OC=%d\n", ic, oc);
    const int bs=1;
    const int ih=5, iw=5, id=4;
    const int oh=3, ow=3, od=2;
    const int kh=3, kw=3, kd=3;
    int weights_len = oc*ic*kh*kw*kd;
    std::vector<float> weights(weights_len, 0);
    for (int o = 0; o < oc; o++) {
        for (int i = 0; i < 1; i++) {
            const int off = kd*kh*kw;
            weights[o*ic*off + i*off + off-1] = 1;
        }
    }
    return assert_bkw_data_convolution(bs, ic, oc, ih, iw, id, kh, kw, kd, oh, ow, od,
                              1, 1, 1, 0, 0, 0,
                              weights, 1e-25, true);
}

bool test_bkw_data_full(const int bs, std::vector<int> insize, std::vector<int> kernel,
                        const int ic, const int oc, std::vector<int> stride,
                        std::vector<int> pad, bool fill_with_floats=true) {
    auto float_str = fill_with_floats ? "flt" : "int";
    int ih = insize[0], iw = insize[1], id = insize[2];
    int kh = kernel[0], kw = kernel[1], kd = kernel[2];
    int sh = stride[0], sw = stride[1], sd = stride[2];
    int ph = pad[0], pw = pad[1], pd = pad[2];
    printf("BKWD bs=%d %dx%dx%d w=%dx%dx%d st=%dx%dx%d pd=%dx%dx%d ic=%2d oc=%2d %s ",
           bs, ih, iw, id,
           kh, kw, kd,
           sh, sw, sd,
           ph, pw, pd,
           ic, oc, float_str);
    const int oh=(ih-kh+2*ph)/sh+1, ow=(iw-kw+2*pw)/sw+1, od=(id-kd+2*pd)/sd+1;
    int weights_len = oc*ic*kh*kw*kd;
    std::vector<float> weights(weights_len, 0);
    if (fill_with_floats) {
        for (size_t i = 0; i < weights.size(); i++)
            weights[i] = (float)(rand() % 100)/11.2 + 1.0;
    } else {
        for (size_t i = 0; i < weights.size(); i++)
            weights[i] = rand() % 8 + 1.0;
    }
    float tolerance = fill_with_floats ? 1e-5 : 1e-25;
    return assert_bkw_data_convolution(bs, ic, oc, ih, iw, id, kh, kw, kd, oh, ow, od,
                              sh, sw, sd, ph, pw, pd,
                              weights, tolerance, false);
}


bool test_simple(const int ic=1, const int oc=1) {
    bool ok = true;
    ok = ok && test_fwd_simple(ic, oc);
    ok = ok && test_bkw_weights_simple(ic, oc);
    ok = ok && test_bkw_data_simple(ic, oc);
    return ok;
}

bool test_full(const int bs, std::vector<int> insize, std::vector<int> kernel,
               const int ic, const int oc, std::vector<int> stride,
               std::vector<int> pad, bool fill_with_floats=true) {
    bool ok = true;
    ok = ok && test_fwd_full(bs, insize, kernel, ic, oc, stride, pad, fill_with_floats);
    ok = ok && test_bkw_data_full(bs, insize, kernel, ic, oc, stride, pad, fill_with_floats);
    ok = ok && test_bkw_weights_full(bs, insize, kernel, ic, oc, stride, pad, fill_with_floats);
    return ok;
}

int main(int argc, char **argv) {
    srand(2);
    bool success = true;
    try {
        success = success
            && test_simple(1, 1)
            && test_simple(1, 32)
            && test_simple(2, 32)
            && test_simple(32, 1)
            && test_simple(16, 32)
            && test_simple(32, 16)
            && test_simple(32, 32);
        std::vector<int> batch_sizes = {1, 4};
        std::vector<int> in_channels =  {1,  2,  2,  1,  1, 16, 16, 32, 32};
        std::vector<int> out_channels = {1, 16, 32, 16, 32, 16, 32, 16, 32};
        std::vector<std::vector<int>> insizes;
        insizes.push_back(std::vector<int> { 7,  7,  9});
        insizes.push_back(std::vector<int> {19, 19, 17});
        std::vector<std::vector<int>> kernels;
        kernels.push_back(std::vector<int> {1, 1, 1});
        kernels.push_back(std::vector<int> {2, 2, 2});
        kernels.push_back(std::vector<int> {3, 3, 3});
        std::vector<std::vector<int>> strides;
        strides.push_back(std::vector<int> {1, 1, 1});
        strides.push_back(std::vector<int> {2, 2, 2});
        std::vector<std::vector<int>> paddings;
        paddings.push_back(std::vector<int> {0, 0, 0});
        for(std::vector<std::vector<int>>::iterator s = insizes.begin(); s != insizes.end(); ++s) {
        for(std::vector<std::vector<int>>::iterator k = kernels.begin(); k != kernels.end(); ++k) {
        for(std::vector<std::vector<int>>::iterator st = strides.begin(); st != strides.end(); ++st) {
        for(std::vector<std::vector<int>>::iterator pd = paddings.begin(); pd != paddings.end(); ++pd) {
            for(size_t ic = 0; ic < in_channels.size(); ic++) {
                for(std::vector<int>::iterator mb = batch_sizes.begin(); mb != batch_sizes.end(); ++mb) {
                    success = success && test_full(*mb, *s, *k, in_channels[ic], out_channels[ic], *st, *pd, true);
                    success = success && test_full(*mb, *s, *k, in_channels[ic], out_channels[ic], *st, *pd, false);
                    if (!success)
                        break;
                }
            }
        }
        }
        }
        }
        // larger layers
        success = success
            && test_full(1, {63, 63, 63}, {2, 3, 3},  1, 16, {1, 1, 1}, {0, 0, 0}, true)
            && test_full(1, {63, 63, 63}, {2, 3, 3},  2, 16, {1, 1, 1}, {0, 0, 0}, true)
            && test_full(1, {63, 63, 63}, {2, 3, 3},  1, 32, {1, 1, 1}, {0, 0, 0}, true)
            && test_full(1, {63, 63, 63}, {2, 3, 3}, 16, 16, {1, 1, 1}, {0, 0, 0}, true)
            && test_full(1, {63, 63, 63}, {2, 3, 3}, 32, 16, {1, 1, 1}, {0, 0, 0}, true)
            && test_full(1, {63, 63, 63}, {2, 3, 3}, 16, 32, {1, 1, 1}, {0, 0, 0}, true);

        // cosmoflow layers
        success = success && test_full(1, {128, 128, 128}, {3, 3, 3}, 1, 16, {1, 1, 1}, {0, 0, 0}, true);
        success = success && test_full(1, {63, 63, 63}, {4, 4, 4}, 16, 32, {1, 1, 1}, {0, 0, 0}, true);
        success = success && test_full(1, {30, 30, 30}, {4, 4, 4}, 32, 64, {2, 2, 2}, {0, 0, 0}, true);
        success = success && test_full(1, {14, 14, 14}, {3, 3, 3}, 64, 64, {1, 1, 1}, {0, 0, 0}, true);
        success = success && test_full(1, {12, 12, 12}, {2, 2, 2}, 64, 128, {1, 1, 1}, {0, 0, 0}, true);
        success = success && test_full(1, {11, 11, 11}, {2, 2, 2}, 128, 128, {1, 1, 1}, {0, 0, 0}, true);

        // medical imaging layers
        // NOTE these take a while to run
        // success = success && test_full(1, {336, 304, 400}, {3, 5, 5}, 1, 32, {1, 1, 1}, {0, 0, 0}, true);
        // success = success && test_full(1, {167, 150, 198}, {3, 3, 3}, 32, 32, {1, 1, 1}, {0, 0, 0}, true);
        // success = success && test_full(1, {165, 148, 196}, {3, 3, 3}, 32, 32, {1, 1, 1}, {0, 0, 0}, true);
        success = success && test_full(1, {81, 73, 97}, {2, 3, 3}, 32, 48, {1, 1, 1}, {0, 0, 0}, true);
        success = success && test_full(1, {80, 71, 95}, {2, 3, 3}, 48, 48, {1, 1, 1}, {0, 0, 0}, true);
        success = success && test_full(1, {78, 67, 91}, {1, 1, 1}, 48, 2, {1, 1, 1}, {0, 0, 0}, true);

        if (success) {
            std::cout << "All tests passed successfully." << std::endl;
        } else {
            std::cout << "Some tests FAILED." << std::endl;
        }
    }
    catch(error& e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return success - 1;
}
