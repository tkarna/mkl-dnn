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
#include "mkldnn.hpp"
#include <iomanip>

using namespace mkldnn;

float TOLERANCE = 1e-16;

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

bool check_result(std::string array_name, float* array, float* correct, const int len) {
    float error = 0;
    for (int i = 0; i < len; i++) {
        error += abs(array[i] - correct[i]);
    }
    bool success =  error < TOLERANCE;
    std::cout << "Test " << array_name << ": ";
    if (success) {
        std::cout << "SUCCESS" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
        std::cout << "  Error: " << error << std::endl;
    }
    return success;
}

inline size_t ptr_off_f(const memory::desc &md, int mb, int g, int ic, int id, int ih, int iw) {
    const int G = 1; // NOTE currently without groups
    return ((((size_t)mb * md.data.dims[1] +
              g * md.data.dims[1]/G + ic) * md.data.dims[2] + id) *
              md.data.dims[3] + ih) * md.data.dims[4] + iw;
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

    // define memory descriptors for fwd op
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
    printf("data src format: %d\n", src_fmt);
    printf("conv src format: %d\n", conv_src_fmt);
    printf("src format match: %d\n", conv_src_fmt == src_fmt);

    auto weights_fmt = diff_weights_md.data.format;
    auto conv_weights_fmt = bkww_pd.diff_weights_primitive_desc().desc().data.format;
    bool weights_needs_reorder = conv_weights_fmt != weights_fmt;
    printf("data weights format: %d\n", weights_fmt);
    printf("conv weights format: %d\n", conv_weights_fmt);
    printf("weights format match: %d\n", conv_weights_fmt == weights_fmt);

    auto dst_fmt = diff_dst_md.data.format;
    auto conv_dst_fmt = bkww_pd.diff_dst_primitive_desc().desc().data.format;
    bool dst_needs_reorder = conv_dst_fmt != dst_fmt;
    printf("data dst format: %d\n", dst_fmt);
    printf("conv dst format: %d\n", conv_dst_fmt);
    printf("dst format match: %d\n", conv_dst_fmt == dst_fmt);

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
        printf("Creating src reorder\n");
        net.push_back(reorder(src_mem, reorder_src_mem));
    }
    if (dst_needs_reorder) {
        printf("Creating diff dst reorder\n");
        net.push_back(reorder(diff_dst_mem, reorder_diff_dst_mem));
    }

    // push to net
    net.push_back(bkww_op);

    if (weights_needs_reorder) {
        printf("Creating diff weight reorder\n");
        net.push_back(reorder(reorder_diff_weights_mem, diff_weights_mem));
    }
    // execute
    stream(stream::kind::eager).submit(net).wait();
}


bool assert_bkw_weights_convolution(const int nbatch, const int in_channels,
                                    const int out_channels, const int in_height,
                                    const int in_width, const int in_depth,
                                    const int weights_height,
                                    const int weights_width,
                                    const int weights_depth, const int out_height,
                                    const int out_width, const int out_depth,
                                    std::vector<float>& in_diff_dst,
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

    auto strides = {1, 1, 1};
    auto padding = {0, 0, 0};
    auto dilation = {0, 0, 0};

    auto diff_dst_memory = memory({{{dst_dims}, memory::data_type::f32, memory::format::ncdhw}, cpu_engine}, vect_diff_dst.data());
    auto diff_weights_memory = memory({{{weights_dims}, memory::data_type::f32,memory::format::oidhw}, cpu_engine}, vect_diff_weights.data());
    auto diff_bias_memory = memory({{{bias_dims}, memory::data_type::f32, memory::format::x}, cpu_engine}, vect_diff_bias.data());
    auto src_memory = memory({{{src_dims}, memory::data_type::f32, memory::format::ncdhw}, cpu_engine}, vect_src.data());
    auto ref_diff_weights_memory = memory({{{weights_dims}, memory::data_type::f32, memory::format::oidhw}, cpu_engine}, vect_ref_diff_weights.data());
    auto ref_diff_bias_memory = memory({{{bias_dims}, memory::data_type::f32, memory::format::x}, cpu_engine}, vect_ref_diff_bias.data());


    // assign input and weights data
    for (int mb = 0; mb < nbatch; mb++) {
    for (int c = 0; c < in_channels; c++) {
    for (int i = 0; i < in_depth; i++) {
        for (int j = 0; j < in_height; j++) {
            for (int k = 0; k < in_width; k++) {
                const size_t ix = (((mb*in_channels + c)*in_depth + i)*in_height + j)*in_width + k;
                vect_src[ix] = (i+1)*(j+1)*(k+1);
            }
        }
    }
    }}
    vect_diff_dst = in_diff_dst;

    bool success = true;

    /* Compute reference solution */
    compute_reference_bkw_weights_conv(src_memory,
                                       diff_dst_memory,
                                       ref_diff_weights_memory,
                                       ref_diff_bias_memory,
                                       strides, dilation, padding);

    std::vector<float> vect_tmp_src;
    vect_tmp_src = vect_src;

    compute_bkw_weights_conv(src_memory,
                             diff_dst_memory,
                             diff_weights_memory,
                             diff_bias_memory,
                             strides, dilation, padding);

    // check that src did not change
    success = success && check_result("Source", vect_src.data(), vect_tmp_src.data(), vect_src.size());

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
        success = success && check_result("diff bias", vect_diff_bias.data(), vect_ref_diff_bias.data(), vect_diff_bias.size());
    }
    success = success && check_result("diff weight", vect_diff_weights.data(), vect_ref_diff_weights.data(), vect_diff_weights.size());

    return success;
}

bool test_simple(const int ic=1, const int oc=1) {
    printf("\nRunning 3D bkw weights convolution test: simple IC=%d OC=%d\n", ic, oc);
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
                              in_diff_dst, test_bias, print_arrays);
}

bool test_full(const int ic=1, const int oc=1) {
    printf("\nRunning 3D bkw weights convolution test: full IC=%d OC=%d\n", ic, oc);
    const int bs=1;
    const int ih=5, iw=5, id=4;
    const int oh=3, ow=3, od=2;
    const int kh=3, kw=3, kd=3;
    int out_len = bs*oc*oh*ow*od;
    std::vector<float> in_diff_dst(out_len, 0);
    for (int mb = 0; mb < bs; mb++) {
    for (int c = 0; c < oc; c++) {
    for (int i = 0; i < od; i++) {
        for (int j = 0; j < oh; j++) {
            for (int k = 0; k < ow; k++) {
                const size_t ix = (((mb*oc + c)*od + i)*oh + j)*ow + k;
                in_diff_dst[ix] = (i+1)*(j+1)*(k+1) + c;
            }
        }
    }
    }}
    bool test_bias = true;
    bool print_arrays = false;
    return assert_bkw_weights_convolution(bs, ic, oc, ih, iw, id, kh, kw, kd, oh, ow, od,
                              in_diff_dst, test_bias, print_arrays);
}

int main(int argc, char **argv) {
    bool success = true;
    try {
        success = success
            && test_simple(1, 1)
            && test_simple(1, 32)
            && test_simple(2, 32)
            && test_simple(32, 1)
            && test_simple(16, 32)
            && test_simple(32, 16)
            && test_simple(32, 32)
            && test_full(1, 1)
            && test_full(2, 16)
            && test_full(2, 32)
            && test_full(1, 16)
            && test_full(1, 32)
            && test_full(16, 16)
            && test_full(16, 32)
            && test_full(32, 16)
            && test_full(32, 32);
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
