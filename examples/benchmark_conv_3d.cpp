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
#include <tuple>
#include "mkldnn.hpp"
#include <iomanip>

#include <cstdio>
#include <ctime>
#include <sstream>

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

using namespace mkldnn;

bool check_result(std::string array_name, float* array, float* correct,
                  const int len, float tolerance, bool verbose=false) {
    /* Computes the average abs relative error in the output array */
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

void compute_reference_fwd_conv(const memory &src_mem,
                                const memory &wei_mem,
                                const memory &bias_mem,
                                const memory &dst_mem,
                                const memory::dims &strides,
                                const memory::dims &padding) {
    // NOTE currently without relu, bias and groups
    // NOTE currently only float data type
    const int G = 1;

    float *src = (float*)src_mem.get_data_handle();
    float *dst = (float*)dst_mem.get_data_handle();
    float *wei = (float*)wei_mem.get_data_handle();
    float *bias = (float *)bias_mem.get_data_handle();

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

    const int KDD = 0;
    const int KDH = 0;
    const int KDW = 0;

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
                            float a = bias[oc];
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

bool assert_fwd_convolution(const int nbatch, const int in_channels, const int out_channels,
                        const int in_height, const int in_width, const int in_depth,
                        const int weights_height, const int weights_width, const int weights_depth,
                        const int out_height, const int out_width, const int out_depth,
                        const int stride,
                        std::vector<float>& in_weights){

    auto cpu_engine = engine(engine::cpu, 0);

    // Dimensions of memory to be allocated
    memory::dims conv_src_dims = {nbatch, in_channels, in_depth, in_height, in_width};
    memory::dims conv_weights_dims = {out_channels, in_channels, weights_depth, weights_height, weights_width};
    memory::dims conv_dst_dims = {nbatch, out_channels, out_depth, out_height, out_width};
    memory::dims conv_bias_dims = {out_channels};
    memory::dims conv_strides = {stride, stride, stride};
    auto conv_padding = {0, 0, 0};


    // User provided memory - in a vector of 1D format.
    // 1D allocations src, dst, weights and biases.
    std::vector<float> net_src(nbatch * in_channels * in_height * in_width * in_depth);
    std::vector<float> net_dst(nbatch * out_channels * out_height * out_width * out_depth);
    // Accumulate dimensions for weights and bias 
    // And allocate vectors for those. 
    std::vector<float> conv_weights(std::accumulate(conv_weights_dims.begin(),
        conv_weights_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> conv_bias(std::accumulate(conv_bias_dims.begin(),
        conv_bias_dims.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    // src, weights and bias.
    auto conv_user_src_memory = memory({{{conv_src_dims}, memory::data_type::f32, memory::format::ncdhw}, cpu_engine},
                                       net_src.data());
    auto conv_user_weights_memory = memory({{{conv_weights_dims}, memory::data_type::f32,memory::format::oidhw}, cpu_engine},
                                           conv_weights.data());
    auto conv_user_bias_memory = memory({{{conv_bias_dims}, memory::data_type::f32, memory::format::x}, cpu_engine},
                                        conv_bias.data());
    auto conv_user_dst_memory = memory({{{conv_dst_dims}, memory::data_type::f32, memory::format::ncdhw}, cpu_engine},
                                       net_dst.data());
    auto ref_dst_memory = memory({{{conv_dst_dims}, memory::data_type::f32, memory::format::ncdhw}, cpu_engine},
                                       net_dst.data());

    // Metadata- These are only descriptors. Not real allocation of data.
    /* create memory descriptors for convolution data w/ no specified format */
    // src, bias, weights, and dst.
    auto conv_src_md = memory::desc({conv_src_dims}, memory::data_type::f32, memory::format::any);
    auto conv_weights_md = memory::desc({conv_weights_dims}, memory::data_type::f32, memory::format::any);
    auto conv_bias_md = memory::desc({conv_bias_dims}, memory::data_type::f32, memory::format::x);
    auto conv_dst_md = memory::desc({conv_dst_dims}, memory::data_type::f32, memory::format::any);

    /* create a convolution */
    // convolution descriptor
    auto conv_fwd_desc = convolution_forward::desc(prop_kind::forward,
        convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
        conv_dst_md, conv_strides, conv_padding, conv_padding,
        padding_kind::zero, conv_kind::conv3D);

    // primitive descriptors
    auto conv_fwd_prim_desc =
        convolution_forward::primitive_desc(conv_fwd_desc, cpu_engine);

    printf("conv src format: %d\n", conv_fwd_prim_desc.src_primitive_desc().desc().data.format);
    printf("user src format: %d\n", conv_user_src_memory.get_primitive_desc().desc().data.format);
    printf("src format match: %d\n", conv_fwd_prim_desc.src_primitive_desc() == conv_user_src_memory.get_primitive_desc());

    /* create reorders between user and data if it is needed and
     *  add it to net before convolution */
    bool src_needs_reorder = memory::primitive_desc(conv_fwd_prim_desc.src_primitive_desc()) !=
        conv_user_src_memory.get_primitive_desc();
    auto conv_src_memory = conv_user_src_memory;
    if (src_needs_reorder) {
        conv_src_memory = memory(conv_fwd_prim_desc.src_primitive_desc());
    }
    bool weights_need_reorder = memory::primitive_desc(conv_fwd_prim_desc.weights_primitive_desc()) !=
        conv_user_weights_memory.get_primitive_desc();
    auto conv_weights_memory = conv_user_weights_memory;
    if (weights_need_reorder) {
        conv_weights_memory = memory(conv_fwd_prim_desc.weights_primitive_desc());
    }

    auto conv_dst_memory = memory(conv_fwd_prim_desc.dst_primitive_desc());

    /* create convolution primitive */
    auto conv_op = convolution_forward(conv_fwd_prim_desc, conv_src_memory,
        conv_weights_memory, conv_user_bias_memory, conv_dst_memory);

    // assign input and weights data
    float *src_data = (float *)conv_user_src_memory.get_data_handle();
    float *src_data_re = (float *)conv_src_memory.get_data_handle();


    for (int mb = 0; mb < nbatch; mb++) {
    for (int c = 0; c < in_channels; c++) {
    for (int i = 0; i < in_depth; i++) {
        for (int j = 0; j < in_height; j++) {
            for (int k = 0; k < in_width; k++) {
                src_data[c*in_depth*in_height*in_width +
                         i*in_height*in_width + j*in_width + k] = (i+1)*(j+1)*(k+1);
                src_data_re[c*in_depth*in_height*in_width +
                            i*in_height*in_width + j*in_width + k] = 4.5;
            }
        }
    }
    }
    }

    conv_weights = in_weights;
    compute_reference_fwd_conv(conv_user_src_memory,
                               conv_user_weights_memory,
                               conv_user_bias_memory,
                               ref_dst_memory,
                               conv_strides, conv_padding);

    // create network array
    std::vector<primitive> net;

    printf("input %dx%dx%d kernel %dx%dx%d in_ch=%d out_ch=%d bs=%d\n",
           in_height, in_width, in_depth,
           weights_height, weights_width, weights_depth,
           in_channels, out_channels, nbatch
          );
    float complexity = 2.0*((float)out_height)*out_width*out_depth*weights_height*weights_width*weights_depth*in_channels*out_channels;
    std::cout << "flops: " << complexity << "\n";

    const int ntime = 100;
    if (src_needs_reorder) {
        printf("Running src reorder\n");
        auto op = reorder(conv_user_src_memory, conv_src_memory);
        net.clear();
        net.push_back(op);
        auto t1 = Clock::now();
        // Execute
        for (int it = 0; it < ntime; it++)
            stream(stream::kind::eager).submit(net).wait();
        auto t2 = Clock::now();
        float duration = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/ntime;
        std::cout << "Duration: " << duration << " ms" << "\n";
        // std::cout << "MFlops/s: " << complexity/1000./1000./duration*1000. << "\n";
    }
    if (weights_need_reorder) {
        printf("Running weight reorder\n");
        auto op = reorder(conv_user_weights_memory, conv_weights_memory);
        net.clear();
        net.push_back(op);
        auto t1 = Clock::now();
        // Execute
        for (int it = 0; it < ntime; it++)
            stream(stream::kind::eager).submit(net).wait();
        auto t2 = Clock::now();
        float duration = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/ntime;
        std::cout << "Duration: " << duration << " ms" << "\n";
        // std::cout << "MFlops/s: " << complexity/1000./1000./duration*1000. << "\n";
    }

    // compute the output
    net.clear();
    net.push_back(reorder(conv_user_src_memory, conv_src_memory));
    net.push_back(reorder(conv_user_weights_memory, conv_weights_memory));
    net.push_back(conv_op);
    net.push_back(reorder(conv_dst_memory, conv_user_dst_memory));
    stream(stream::kind::eager).submit(net).wait();
    check_result("output", (float *)ref_dst_memory.get_data_handle(), (float *)conv_user_dst_memory.get_data_handle(),
        nbatch * out_channels * out_height * out_width * out_depth, 1e-7);


    printf("Running forward convolution\n");
    net.clear();
    net.push_back(conv_op);
    auto t1 = Clock::now();
    // Execute
    for (int it = 0; it < ntime; it++)
        stream(stream::kind::eager).submit(net).wait();
    auto t2 = Clock::now();
    float duration = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/ntime;
    std::cout << "Duration: " << duration << " ms"  << std::endl;
    std::cout << "FWD: " << nbatch << " " << in_channels << " " << out_channels << " " <<
            in_height << " " << weights_height << " " <<  stride <<
        "   GFlops/s: " << complexity/1000./1000./1000./duration*1000.*nbatch  << std::endl << std::endl;

    return 1;
}

bool test_fwd_conv(const int bs, const int ic, const int oc, const int insize, const int fSize, const int stride) {
    const int ih=insize, iw=insize, id=insize;
    const int kh = fSize, kw = fSize, kd = fSize;
    const int oh=(ih-kh)/stride+1, ow=(iw-kw)/stride+1, od=(id-kd)/stride+1;
    int weights_len = oc*ic*kh*kw*kd;
    std::vector<float> in_weights(weights_len, 0);
    for (int o = 0; o < oc; o++) {
        for (int i = 0; i < ic; i++) {
            for (int d = 0; d < kd; d++) {
                for (int h = 0; h < kh; h++) {
                    for (int w = 0; w < kw; w++) {
                        const int ix = (((o*ic + i)*kd + d)*kh + h)*kw + w;
                        if ((h==1 || w==2) && (i % 4==0)  && (o % 4==0))
                            in_weights[ix] = (w+1) + d - (i/4+1) + (o/4+1);
                    }
                }
            }
        }
    }
    return assert_fwd_convolution(bs, ic, oc, ih, iw, id, kh, kw, kd, oh, ow, od, stride,
                              in_weights);

}

int main(int argc, char **argv) {
    const int cosmoflow_dims[][6] = {
        {1, 1, 16, 128, 3, 1},
        {1, 16, 32, 63, 4, 1},
        {1, 32, 64, 30, 4, 2},
        {1, 64, 64, 14, 3, 1},
        {1, 64, 128, 12, 2, 1},
        {1, 128, 128, 11, 2, 1},
        {0}
    };

    const int (*dims)[6] = cosmoflow_dims;
    for (int i = 0; cosmoflow_dims[i][0]; ++i)
        test_fwd_conv(dims[i][0], dims[i][1], dims[i][2], dims[i][3], dims[i][4], dims[i][5]);
    return 0;
}
