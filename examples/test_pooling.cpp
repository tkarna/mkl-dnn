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

/**************************************
 * Compilation:
 *
 * export MKLDNN_ROOT=<path_to_mkldnn_install_dir>
 * export LIBRARY_PATH=$MKLDNN_ROOT/lib/:$LIBRARY_PATH
 * export LD_LIBRARY_PATH=$MKLDNN_ROOT/lib/:$LD_LIBRARY_PATH
 * export CPATH="$MKLDNN_ROOT/include/"
 *
 * g++ -std=c++11 -Wall -Werror=unused-variable test_pooling.cpp -lmkldnn -o test_pooling
 *
 **************************************/

#include <iostream>
#include <numeric>
#include <string>
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

void print_array_2d(std::string name, float* array, int m, int l) {
    std::cout << name << ":" << std::endl;
    for (int j=0; j<m; j++) {
        for (int k=0; k<l; k++) {
            std::cout << std::setw(5) << array[l*j + k];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void compute_fwd_pool(const memory &src_mem,
                         const memory &dst_mem,
                         const memory::dims &strides,
                         const memory::dims &kernel,
                         const memory::dims &padding) {

    auto cpu_engine = engine(engine::cpu, 0);

    auto src_pd = src_mem.get_primitive_desc();
    auto dst_pd = dst_mem.get_primitive_desc();

    auto src_md = src_pd.desc();
    auto dst_md = dst_pd.desc();

    auto pooling_op = pooling_max;

    /* op descriptors */
    auto pool_fwd_desc = pooling_forward::desc(prop_kind::forward,
        pooling_op, src_md, dst_md, strides, kernel, padding, padding,
                                               padding_kind::zero);

    /* primitive op descriptors */
    auto pool_fwd_pd =
        pooling_forward::primitive_desc(pool_fwd_desc, cpu_engine);

    /* test if we need workspace */
    bool with_workspace = pooling_op == pooling_max; // NOTE only for forward op

    auto ws_pd = with_workspace ? pool_fwd_pd.workspace_primitive_desc() : dst_mem.get_primitive_desc();
    auto ws_mem = with_workspace ? memory(ws_pd) : dst_mem;

    /* create forward op primitive */
    auto pool_op = with_workspace ?
       pooling_forward(pool_fwd_pd, src_mem, dst_mem, ws_mem) :
       pooling_forward(pool_fwd_pd, src_mem, dst_mem);

    // create network array
    std::vector<primitive> net;

    // push to net
    net.push_back(pool_op);

    // Execute
    stream(stream::kind::eager).submit(net).wait();
}

bool assert_pooling_2d(const int nbatch, const int in_channels, const int out_channels,
                       const int in_height, const int in_width,
                       const int ker_height,const int ker_width,
                       const int out_height, const int out_width,
                       bool print_arrays = true){

    auto cpu_engine = engine(engine::cpu, 0);

    // Dimensions of memory to be allocated
    memory::dims src_dims = {nbatch, in_channels, in_height, in_width};
    memory::dims dst_dims = {nbatch, out_channels, out_height, out_width};

    auto strides = {1, 1};
    auto padding = {0, 0};
    auto kernel = {ker_height, ker_width};

    std::vector<float> vect_src(std::accumulate(src_dims.begin(),
        src_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_dst(std::accumulate(dst_dims.begin(),
        dst_dims.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    auto src_memory = memory({{{src_dims},
                             memory::data_type::f32, memory::format::nchw},
                             cpu_engine}, vect_src.data());
    auto dst_memory = memory({{{dst_dims},
                             memory::data_type::f32, memory::format::nchw},
                             cpu_engine}, vect_dst.data());

    // assign input and weights data
    for (int mb = 0; mb < nbatch; mb++) {
    for (int c = 0; c < in_channels; c++) {
        for (int j = 0; j < in_height; j++) {
            for (int k = 0; k < in_width; k++) {
                const size_t ix = ((mb*in_channels + c)*in_height + j)*in_width + k;
                vect_src[ix] = (j+1)*(k+1);
            }
        }
    }}

    compute_fwd_pool(src_memory,
                     dst_memory,
                     strides, kernel, padding);

    if (print_arrays) {
        print_array_2d("Input", vect_src.data(), src_dims[2], src_dims[3]);
        print_array_2d("Output", vect_dst.data(), dst_dims[2], dst_dims[3]);
    }

    return true;
}

bool assert_pooling_3d(const int nbatch, const int in_channels, const int out_channels,
                       const int in_height, const int in_width, const int in_depth,
                       const int ker_height,const int ker_width, const int ker_depth,
                       const int out_height, const int out_width, const int out_depth,
                       bool print_arrays = true){

    auto cpu_engine = engine(engine::cpu, 0);

    // Dimensions of memory to be allocated
    memory::dims src_dims = {nbatch, in_channels, in_depth, in_height, in_width};
    memory::dims dst_dims = {nbatch, out_channels, out_depth, out_height, out_width};

    auto strides = {1, 1, 1};
    auto padding = {0, 0, 0};
    auto kernel = {ker_depth, ker_height, ker_width};

    std::vector<float> vect_src(std::accumulate(src_dims.begin(),
        src_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_dst(std::accumulate(dst_dims.begin(),
        dst_dims.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    auto src_memory = memory({{{src_dims},
                             memory::data_type::f32, memory::format::ncdhw},
                             cpu_engine}, vect_src.data());
    auto dst_memory = memory({{{dst_dims},
                             memory::data_type::f32, memory::format::ncdhw},
                             cpu_engine}, vect_dst.data());

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

    compute_fwd_pool(src_memory,
                     dst_memory,
                     strides, kernel, padding);

    if (print_arrays) {
        print_array_3d("Input", vect_src.data(), src_dims[2], src_dims[3], src_dims[4]);
        print_array_3d("Output", vect_dst.data(), dst_dims[2], dst_dims[3], dst_dims[4]);
    }

    return true;
}

bool test_simple_2d(const int ic=1, const int oc=1) {
    printf("\nRunning 2D fwd pooling test: simple IC=%d OC=%d\n", ic, oc);
    const int bs=1;
    const int ih=5, iw=5;
    const int oh=3, ow=3;
    const int kh=3, kw=3;
    return assert_pooling_2d(bs, ic, oc, ih, iw, kh, kw, oh, ow);
}

bool test_simple_3d(const int ic=1, const int oc=1) {
    printf("\nRunning 3D fwd pooling test: simple IC=%d OC=%d\n", ic, oc);
    const int bs=1;
    const int ih=5, iw=5, id=4;
    const int oh=3, ow=3, od=2;
    const int kh=3, kw=3, kd=3;
    return assert_pooling_3d(bs, ic, oc, ih, iw, id, kh, kw, kd, oh, ow, od);
}

int main(int argc, char **argv) {
    bool success = true;
    try {
        success = success
            && test_simple_2d(1, 1)
            && test_simple_3d(1, 1);
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
