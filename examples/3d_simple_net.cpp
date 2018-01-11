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

using namespace mkldnn;

void simple_net_3d(){

// 0) Compile current code independently.
// 1) Defining dimensions.
// 2) Figuring out memory sizes and padding/strides.
// 3) Creating memory/data structures for the data.
// 4) Computation.
// 5) Verification.

// Defining parameters of 3D Conv.
// Defining dimensions of the conv in user data.


    auto cpu_engine = engine(engine::cpu, 0);

// Defining dimensions.
    const int batch = 5;
    const int in_height = 7;
    const int in_width = 9;
    const int in_depth = 6;
    
    const int kernel_height = 3;
    const int kernel_width = 3;
    const int kernel_depth = 3;

    const int out_channels = 32;
    const int in_channels = 4;
    
    const int out_height = 5;
    const int out_width = 7;
    const int out_depth = 4;

// Dimensions of memory to be allocated 
    memory::dims conv_src_dims = {batch, in_channels, in_height, in_width, in_depth};
    memory::dims conv_weights_dims = {out_channels, in_channels, kernel_height, kernel_width, kernel_depth};
    memory::dims conv_dst_dims = {batch, out_channels, out_height, out_width, out_depth};
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
    auto conv_user_src_memory = memory({{{conv_src_dims}, memory::data_type::f32, memory::format::nchwd}, cpu_engine}, net_src.data());    
    auto conv_user_weights_memory = memory({{{conv_weights_dims}, memory::data_type::f32, memory::format::oihwd}, cpu_engine}, conv_weights.data());
    auto conv_user_bias_memory = memory({{{conv_bias_dims}, memory::data_type::f32, memory::format::x}, cpu_engine}, conv_bias.data());

    // Metadata- These are only descriptors. Not real allocation of data.
    /* create memory descriptors for convolution data w/ no specified format */
    // src, bias, weights, and dst.
    auto conv_src_md = memory::desc({conv_src_dims}, memory::data_type::f32, memory::format::nchwd);
    auto conv_bias_md = memory::desc({conv_bias_dims}, memory::data_type::f32, memory::format::x);
    auto conv_weights_md = memory::desc({conv_weights_dims}, memory::data_type::f32, memory::format::oihwd);
    auto conv_dst_md = memory::desc({conv_dst_dims}, memory::data_type::f32, memory::format::nchwd);



    /* create a convolution */
    // Pass memory descriptors (metadata) and stride, padding dimensions.
    //
    //Convolution descriptor
    auto conv_desc = convolution_forward::desc(prop_kind::forward,
        convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
        conv_dst_md, conv_strides, conv_padding, conv_padding,
        padding_kind::zero, conv_kind::conv3D);
    // Convolution premitive.
    auto conv_prim_desc =
        convolution_forward::primitive_desc(conv_desc, cpu_engine);


    std::vector<primitive> net;

    /* create reorders between user and data if it is needed and
     *  add it to net before convolution */
    auto conv_src_memory = conv_user_src_memory;
    if (memory::primitive_desc(conv_prim_desc.src_primitive_desc()) !=
        conv_user_src_memory.get_primitive_desc()) {
        conv_src_memory = memory(conv_prim_desc.src_primitive_desc());
        net.push_back(reorder(conv_user_src_memory, conv_src_memory));
    }

    auto conv_weights_memory = conv_user_weights_memory;
    if (memory::primitive_desc(conv_prim_desc.weights_primitive_desc()) !=
        conv_user_weights_memory.get_primitive_desc()) {
        conv_weights_memory = memory(conv_prim_desc.weights_primitive_desc());
        net.push_back(reorder(conv_user_weights_memory, conv_weights_memory));
    }

    auto conv_dst_memory = memory(conv_prim_desc.dst_primitive_desc());

    /* create convolution primitive and add it to net */
    net.push_back(convolution_forward(conv_prim_desc, conv_src_memory,
        conv_weights_memory, conv_user_bias_memory, conv_dst_memory));

    stream(stream::kind::eager).submit(net).wait();

}

int main(int argc, char **argv) {
    std::cout << "\nExecuting 3D ref convolution.\n";
    try {
        simple_net_3d();
    }
    catch(error& e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
