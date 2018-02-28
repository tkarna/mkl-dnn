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
        std::cout << "SUCCESS" << std::endl;
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

bool test_simple(const int ic=1, const int oc=1) {
    printf("\nRunning 3D bkw data convolution test: simple IC=%d OC=%d\n", ic, oc);
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

bool test_full(const int bs, std::vector<int> insize, std::vector<int> kernel,
               const int ic, const int oc, std::vector<int> stride,
               std::vector<int> pad, bool fill_with_floats=true) {
    auto float_str = fill_with_floats ? "flt" : "int";
    int ih = insize[0], iw = insize[1], id = insize[2];
    int kh = kernel[0], kw = kernel[1], kd = kernel[2];
    int sh = stride[0], sw = stride[1], sd = stride[2];
    int ph = pad[0], pw = pad[1], pd = pad[2];
    printf("bs=%d %dx%dx%d w=%dx%dx%d st=%dx%dx%d pd=%dx%dx%d ic=%2d oc=%2d %s ",
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
//         // cosmoflow layers
//         success = success && test_full(1, {128, 128, 128}, {3, 3, 3}, 1, 16, {1, 1, 1}, {0, 0, 0}, true);
//         success = success && test_full(1, {63, 63, 63}, {4, 4, 4}, 16, 32, {1, 1, 1}, {0, 0, 0}, true);
//         success = success && test_full(1, {30, 30, 30}, {4, 4, 4}, 32, 64, {2, 2, 2}, {0, 0, 0}, true);
//         success = success && test_full(1, {14, 14, 14}, {3, 3, 3}, 64, 64, {1, 1, 1}, {0, 0, 0}, true);
//         success = success && test_full(1, {12, 12, 12}, {2, 2, 2}, 64, 128, {1, 1, 1}, {0, 0, 0}, true);
//         success = success && test_full(1, {11, 11, 11}, {2, 2, 2}, 128, 128, {1, 1, 1}, {0, 0, 0}, true);
//         // medical imaging layers
//         // NOTE these take a while to run
//         // success = success && test_full(1, {336, 304, 400}, {3, 5, 5}, 1, 32, {1, 1, 1}, {0, 0, 0}, true);
//         // success = success && test_full(1, {167, 150, 198}, {3, 3, 3}, 32, 32, {1, 1, 1}, {0, 0, 0}, true);
//         // success = success && test_full(1, {165, 148, 196}, {3, 3, 3}, 32, 32, {1, 1, 1}, {0, 0, 0}, true);
//         success = success && test_full(1, {81, 73, 97}, {2, 3, 3}, 32, 48, {1, 1, 1}, {0, 0, 0}, true);
//         success = success && test_full(1, {80, 71, 95}, {2, 3, 3}, 48, 48, {1, 1, 1}, {0, 0, 0}, true);
//         success = success && test_full(1, {78, 67, 91}, {1, 1, 1}, 48, 2, {1, 1, 1}, {0, 0, 0}, true);

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
