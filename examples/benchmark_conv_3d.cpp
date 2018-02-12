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

#include <cstdio>
#include <ctime>

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

using namespace mkldnn;


bool assert_convolution(const int in_height, const int in_width, const int in_depth,
                        const int weights_height, const int weights_width, const int weights_depth,
                        const int out_height, const int out_width, const int out_depth,
                        std::vector<float>& in_weights){

    auto cpu_engine = engine(engine::cpu, 0);

    // Defining dimensions.
    const int batch = 1;

    const int out_channels = 32;
    const int in_channels = 16;

    // Dimensions of memory to be allocated
    memory::dims conv_src_dims = {batch, in_channels, in_depth, in_height, in_width};
    memory::dims conv_weights_dims = {out_channels, in_channels, weights_depth, weights_height, weights_width};
    memory::dims conv_dst_dims = {batch, out_channels, out_depth, out_height, out_width};
    memory::dims conv_bias_dims = {out_channels};
    memory::dims conv_strides = {1, 1, 1};
    auto conv_padding = {0, 0, 0};


    // User provided memory - in a vector of 1D format.
    // 1D allocations src, dst, weights and biases.
    std::vector<float> net_src(batch * in_channels * in_height * in_width * in_depth);
    std::vector<float> net_dst(batch * out_channels * out_height * out_width * out_depth); 
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

    conv_weights = in_weights;

    // create network array
    std::vector<primitive> net;

    printf("input %dx%dx%d kernel %dx%dx%d in_ch=%d out_ch=%d\n",
           in_height, in_width, in_depth,
           weights_height, weights_width, weights_depth,
           in_channels, out_channels
          );
    float complexity = ((float)in_height)*in_width*in_depth*weights_height*weights_width*weights_depth*in_channels*out_channels;
    std::cout << "flops: " << complexity << "\n";

    const int ntime = 100;
    if (src_needs_reorder) {
        printf("Running src reorder\n");
        auto op = reorder(conv_user_src_memory, conv_src_memory);
        net.clear();
        net.push_back(op);

        auto t1 = Clock::now();
        for (int it = 0; it < ntime; it++) {
            // Execute
            stream(stream::kind::eager).submit(net).wait();
        }
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
        for (int it = 0; it < ntime; it++) {
            // Execute
            stream(stream::kind::eager).submit(net).wait();
        }
        auto t2 = Clock::now();
        float duration = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/ntime;
        std::cout << "Duration: " << duration << " ms" << "\n";
        // std::cout << "MFlops/s: " << complexity/1000./1000./duration*1000. << "\n";
    }

    printf("Running forward convolution\n");
    net.clear();
    net.push_back(conv_op);

    auto t1 = Clock::now();
    for (int it = 0; it < ntime; it++) {
        // Execute
        stream(stream::kind::eager).submit(net).wait();
    }
    auto t2 = Clock::now();
    float duration = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/ntime;
    std::cout << "Duration: " << duration << " ms" << "\n";
    std::cout << "MFlops/s: " << complexity/1000./1000./duration*1000. << "\n";

    return 1;
}

bool test_simple() {
    std::cout << "\nRunning 3D convolution test: simple\n";
    const int ih=5, iw=5, id=4;
    const int oh=3, ow=3, od=2;
    const int kh=3, kw=3, kd=3;
    int out_len = oh*ow*od;
    int weights_len = kh*kw*kd;
    std::vector<float> in_weights(weights_len, 0);
    std::vector<float> correct_output(out_len, 0);
    in_weights = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 1
    };
    return assert_convolution(ih, iw, id, kh, kw, kd, oh, ow, od,
                        in_weights);
}

bool test_asymmetric() {
    std::cout << "\nRunning 3D convolution test: asymmetric\n";
    const int ih=64, iw=64, id=64;
    const int oh=62, ow=62, od=62;
    const int kh=3, kw=3, kd=3;
    int out_len = oh*ow*od;
    int weights_len = kh*kw*kd;
    std::vector<float> in_weights(weights_len, 0);
    std::vector<float> correct_output(out_len, 0);
    in_weights = {
        0, 0, 0,
        0, 4, 0,
        0, 0, 0,
        0, 0, 3,
        0, 0, 1,
        0, 0, 0,
        3, 0, 0,
        1, 2, 0,
        1, 0, 0
    };
    return assert_convolution(ih, iw, id, kh, kw, kd, oh, ow, od,
                        in_weights);

}

int main(int argc, char **argv) {
    try {
        test_asymmetric();
    }
    catch(error& e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
