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
                std::cout << std::setw(4) << array[m*l*i + l*j + k];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

bool check_result(float* array, float* correct, const int len) {
    float error = 0;
    for (int i = 0; i < len; i++) {
        error += abs(array[i] - correct[i]);
    }
    bool success =  error < TOLERANCE;
    if (success) {
        std::cout << "SUCCESS" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
        std::cout << "Error: " << error << std::endl;
    }
    return success;
}

bool assert_convolution(const int in_height, const int in_width, const int in_depth,
                        const int weights_height, const int weights_width, const int weights_depth,
                        const int out_height, const int out_width, const int out_depth,
                        std::vector<float>& in_weights,
                        std::vector<float>& in_diff_output_weights,
                        std::vector<float>& in_diff_output_data,
                        std::vector<float>& correct_output,
                        std::vector<float>& correct_diff_weights,
                        std::vector<float>& correct_diff_input){

    auto cpu_engine = engine(engine::cpu, 0);

    // Defining dimensions.
    const int batch = 1;

    const int out_channels = 1;
    const int in_channels = 1;

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
    std::vector<float> conv_diff_src(std::accumulate(conv_src_dims.begin(),
        conv_src_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> conv_diff_weights(std::accumulate(conv_weights_dims.begin(),
        conv_weights_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> conv_diff_bias(std::accumulate(conv_bias_dims.begin(),
        conv_bias_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> conv_diff_dst(std::accumulate(conv_dst_dims.begin(),
        conv_dst_dims.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    // src, weights and bias.
    auto conv_user_src_memory = memory({{{conv_src_dims}, memory::data_type::f32, memory::format::ncdhw}, cpu_engine},
                                       net_src.data());
    auto conv_user_weights_memory = memory({{{conv_weights_dims}, memory::data_type::f32,memory::format::oidhw}, cpu_engine},
                                           conv_weights.data());
    auto conv_user_bias_memory = memory({{{conv_bias_dims}, memory::data_type::f32, memory::format::x}, cpu_engine},
                                        conv_bias.data());

    auto conv_diff_src_memory = memory({{{conv_src_dims}, memory::data_type::f32, memory::format::ncdhw}, cpu_engine},
                                       conv_diff_src.data());
    auto conv_diff_weights_memory = memory({{{conv_weights_dims}, memory::data_type::f32, memory::format::oidhw}, cpu_engine},
                                           conv_diff_weights.data());
    auto conv_diff_bias_memory = memory({{{conv_bias_dims}, memory::data_type::f32, memory::format::x}, cpu_engine},
                                        conv_diff_bias.data());

    // Metadata- These are only descriptors. Not real allocation of data.
    /* create memory descriptors for convolution data w/ no specified format */
    // src, bias, weights, and dst.
    auto conv_src_md = memory::desc({conv_src_dims}, memory::data_type::f32, memory::format::ncdhw);
    auto conv_weights_md = memory::desc({conv_weights_dims}, memory::data_type::f32, memory::format::oidhw);
    auto conv_bias_md = memory::desc({conv_bias_dims}, memory::data_type::f32, memory::format::x);
    auto conv_dst_md = memory::desc({conv_dst_dims}, memory::data_type::f32, memory::format::ncdhw);

    auto conv_diff_src_md = mkldnn::memory::desc({conv_src_dims}, mkldnn::memory::data_type::f32, mkldnn::memory::format::any);
    auto conv_diff_weights_md = mkldnn::memory::desc({conv_weights_dims}, mkldnn::memory::data_type::f32, mkldnn::memory::format::any);
    auto conv_diff_bias_md = mkldnn::memory::desc({conv_bias_dims}, mkldnn::memory::data_type::f32, mkldnn::memory::format::any);
    auto conv_diff_dst_md = mkldnn::memory::desc({conv_dst_dims}, mkldnn::memory::data_type::f32, mkldnn::memory::format::any);

    /* create a convolution */
    // convolution descriptor
    auto conv_fwd_desc = convolution_forward::desc(prop_kind::forward,
        convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
        conv_dst_md, conv_strides, conv_padding, conv_padding,
        padding_kind::zero, conv_kind::conv3D);
    auto conv_bkww_desc = mkldnn::convolution_backward_weights::desc(
        mkldnn::convolution_direct, conv_src_md, conv_diff_weights_md, conv_diff_bias_md,
        conv_diff_dst_md, conv_strides, conv_padding, conv_padding,
        mkldnn::padding_kind::zero, conv_kind::conv3D);
    auto conv_bkwd_desc = mkldnn::convolution_backward_data::desc(
        mkldnn::convolution_direct, conv_diff_src_md, conv_weights_md,
        conv_diff_dst_md, conv_strides, conv_padding, conv_padding,
        mkldnn::padding_kind::zero, conv_kind::conv3D);

    // primitive descriptors
    auto conv_fwd_prim_desc =
        convolution_forward::primitive_desc(conv_fwd_desc, cpu_engine);
    auto conv_bkww_prim_desc =
        convolution_backward_weights::primitive_desc(conv_bkww_desc, cpu_engine, conv_fwd_prim_desc);
    auto conv_bkwd_prim_desc =
        mkldnn::convolution_backward_data::primitive_desc(conv_bkwd_desc, cpu_engine, conv_fwd_prim_desc);

    /* create reorders between user and data if it is needed and
     *  add it to net before convolution */
    auto conv_src_memory = conv_user_src_memory;
    // if (memory::primitive_desc(conv_fwd_prim_desc.src_primitive_desc()) !=
    //     conv_user_src_memory.get_primitive_desc()) {
    //     conv_src_memory = memory(conv_fwd_prim_desc.src_primitive_desc());
    //     net.push_back(reorder(conv_user_src_memory, conv_src_memory));
    // }
    //
    auto conv_weights_memory = conv_user_weights_memory;
    // if (memory::primitive_desc(conv_fwd_prim_desc.weights_primitive_desc()) !=
    //     conv_user_weights_memory.get_primitive_desc()) {
    //     conv_weights_memory = memory(conv_fwd_prim_desc.weights_primitive_desc());
    //     net.push_back(reorder(conv_user_weights_memory, conv_weights_memory));
    // }

    auto conv_dst_memory = memory(conv_fwd_prim_desc.dst_primitive_desc());
    auto conv_diff_dst_memory = memory(conv_bkww_prim_desc.diff_dst_primitive_desc(), conv_diff_dst.data());

    /* create convolution primitive */
    auto conv_op = convolution_forward(conv_fwd_prim_desc, conv_src_memory,
        conv_weights_memory, conv_user_bias_memory, conv_dst_memory);

    // create backward weights convolution obj
    auto conv_bkww_op = convolution_backward_weights(conv_bkww_prim_desc,
        conv_src_memory, conv_diff_dst_memory, conv_diff_weights_memory, conv_diff_bias_memory);

    // create backward data convolution obj
    auto conv_bkwd_op = mkldnn::convolution_backward_data(conv_bkwd_prim_desc,
        conv_diff_dst_memory, conv_weights_memory, conv_diff_src_memory);

    // assign input and weights data
    float *src_data = (float *)conv_src_memory.get_data_handle();
    float *dst_data = (float *)conv_dst_memory.get_data_handle();
    float *w_data = (float *)conv_weights_memory.get_data_handle();

    for (int i = 0; i < in_depth; i++) {
        for (int j = 0; j < in_height; j++) {
            for (int k = 0; k < in_width; k++) {
                src_data[i*in_height*in_width + j*in_width + k] = (i+1)*(j+1)*(k+1);
            }
        }
    }

    conv_weights = in_weights;

    bool success = true;

    // create network array
    std::vector<primitive> net;

    std::cout << "Forward convolution:" << std::endl;
    // push to net
    net.clear();
    net.push_back(conv_op);

    // Execute
    stream(stream::kind::eager).submit(net).wait();

    // Print the output matrix
    print_array_3d("Input", src_data, conv_src_dims[2], conv_src_dims[3], conv_src_dims[4]);
    print_array_3d("Kernel", w_data, conv_weights_dims[2], conv_weights_dims[3], conv_weights_dims[4]);
    print_array_3d("Output", dst_data, conv_dst_dims[2], conv_dst_dims[3], conv_dst_dims[4]);

    // Compute error
    success = success && check_result(dst_data, correct_output.data(), out_height*out_width*out_depth);

    std::cout << "Backward convolution weights:" << std::endl;
    conv_diff_dst = in_diff_output_weights;
    // push to net
    net.clear();
    net.push_back(conv_bkww_op);
    // execute
    stream(stream::kind::eager).submit(net).wait();

    print_array_3d("Input", src_data, conv_src_dims[2], conv_src_dims[3], conv_src_dims[4]);
    print_array_3d("Diff_output", (float*)conv_diff_dst_memory.get_data_handle(), conv_dst_dims[2], conv_dst_dims[3], conv_dst_dims[4]);
    print_array_3d("Diff_weights", conv_diff_weights.data(), conv_weights_dims[2], conv_weights_dims[3], conv_weights_dims[4]);

    success = success && check_result(conv_diff_weights.data(), correct_diff_weights.data(), conv_diff_weights.size());

    std::cout << "Backward convolution data:" << std::endl;
    conv_diff_dst = in_diff_output_data;
    // push to net
    net.clear();
    net.push_back(conv_bkwd_op);
    // execute
    mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

    print_array_3d("Kernel", w_data, conv_weights_dims[2], conv_weights_dims[3], conv_weights_dims[4]);
    print_array_3d("Diff_output", conv_diff_dst.data(), conv_dst_dims[2], conv_dst_dims[3], conv_dst_dims[4]);
    print_array_3d("Diff_input", conv_diff_src.data(), conv_src_dims[2], conv_src_dims[3], conv_src_dims[4]);

    success = success && check_result(conv_diff_src.data(), correct_diff_input.data(), conv_diff_src.size());

    return success;
}

bool test_simple() {
    std::cout << "\nRunning 3D convolution test: simple\n";
    const int ih=5, iw=5, id=4;
    const int oh=3, ow=3, od=2;
    const int kh=3, kw=3, kd=3;
    int in_len = ih*iw*od;
    int out_len = oh*ow*od;
    int weights_len = kh*kw*kd;
    std::vector<float> in_weights(weights_len, 0);
    std::vector<float> in_diff_output_weights(in_len, 0);
    std::vector<float> in_diff_output_data(in_len, 0);
    std::vector<float> correct_output(out_len, 0);
    std::vector<float> correct_diff_weights(weights_len, 0);
    std::vector<float> correct_diff_input(in_len, 0);
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
    in_diff_output_weights = {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 1
    };
    in_diff_output_data = {
         1,  2,  3,
         2,  4,  6,
         3,  6,  9,
         2,  4,  6,
         4,  8, 12,
         6, 12, 18
    };
    correct_output = {
        27,  36,  45,
        36,  48,  60,
        45,  60,  75,
        36,  48,  60,
        48,  64,  80,
        60,  80, 100
    };
    correct_diff_weights = {
        18,  24,  30,
        24,  32,  40,
        30,  40,  50,
        27,  36,  45,
        36,  48,  60,
        45,  60,  75,
        36,  48,  60,
        48,  64,  80,
        60,  80, 100
    };
    correct_diff_input = {
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  1,  2,  3,
        0,  0,  2,  4,  6,
        0,  0,  3,  6,  9,
        0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,
        0,  0,  2,  4,  6,
        0,  0,  4,  8, 12,
        0,  0,  6, 12, 18
    };
    return assert_convolution(ih, iw, id, kh, kw, kd, oh, ow, od,
                        in_weights,
                        in_diff_output_weights,
                        in_diff_output_data,
                        correct_output,
                        correct_diff_weights,
                        correct_diff_input);
}

bool test_asymmetric() {
    std::cout << "\nRunning 3D convolution test: asymmetric\n";
    const int ih=5, iw=5, id=4;
    const int oh=3, ow=3, od=2;
    const int kh=3, kw=3, kd=3;
    int in_len = ih*iw*od;
    int out_len = oh*ow*od;
    int weights_len = kh*kw*kd;
    std::vector<float> in_weights(weights_len, 0);
    std::vector<float> in_diff_output_weights(in_len, 0);
    std::vector<float> in_diff_output_data(in_len, 0);
    std::vector<float> correct_output(out_len, 0);
    std::vector<float> correct_diff_weights(weights_len, 0);
    std::vector<float> correct_diff_input(in_len, 0);
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
    in_diff_output_weights = {
        1, 0, 0,
        0, 0, 1,
        0, 2, 0,
        0, 0, 0,
        0, 1, 0,
        0, 0, 1
    };
    in_diff_output_data = {
        1,  2,  3,
        2,  4,  6,
        3,  6,  9,
        2,  4,  6,
        4,  8, 12,
        6, 12, 18
    };
    correct_output = {
         94, 148, 202,
        153, 240, 327,
        212, 332, 452,
        141, 220, 299,
        229, 356, 483,
        317, 492, 667
    };
    correct_diff_weights = {
         45,  64,  83,
         63,  90, 117,
         81, 116, 151,
         77, 110, 143,
        108, 155, 202,
        139, 200, 261,
        109, 156, 203,
        153, 220, 287,
        197, 284, 371
    };
    correct_diff_input = {
         0,  0,  0,  0,  0,
         0,  4,  8, 12,  0,
         0,  8, 16, 24,  0,
         0, 12, 24, 36,  0,
         0,  0,  0,  0,  0,
         0,  0,  3,  6,  9,
         0,  8, 23, 38, 21,
         0, 16, 43, 70, 33,
         0, 24, 51, 78,  9,
         0,  0,  0,  0,  0,
         3,  6, 15, 12, 18,
         7, 16, 39, 34, 42,
        12, 28, 66, 56, 66,
         5, 16, 33, 30, 18,
         3,  6,  9,  0,  0,
         6, 12, 18,  0,  0,
        14, 32, 50, 12,  0,
        24, 56, 88, 24,  0,
        10, 32, 54, 36,  0,
         6, 12, 18,  0,  0
    };
    return assert_convolution(ih, iw, id, kh, kw, kd, oh, ow, od,
                        in_weights,
                        in_diff_output_weights,
                        in_diff_output_data,
                        correct_output,
                        correct_diff_weights,
                        correct_diff_input);

}

int main(int argc, char **argv) {
    bool success = true;
    try {
        success = success
            && test_simple()
            && test_asymmetric();
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
