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

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

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

void compute_fwd_pool(algorithm pooling_alg,
                      const memory &src_mem,
                      const memory &dst_mem,
                      const memory::dims &strides,
                      const memory::dims &kernel,
                      const memory::dims &padding
                      ) {

    auto cpu_engine = engine(engine::cpu, 0);

    auto src_pd = src_mem.get_primitive_desc();
    auto dst_pd = dst_mem.get_primitive_desc();

    auto src_md = src_pd.desc();
    auto dst_md = dst_pd.desc();

    /* op descriptors */
    auto pool_fwd_desc = pooling_forward::desc(prop_kind::forward,
        pooling_alg, src_md, dst_md, strides, kernel, padding, padding,
                                               padding_kind::zero);

    /* primitive op descriptors */
    auto pool_fwd_pd =
        pooling_forward::primitive_desc(pool_fwd_desc, cpu_engine);

    /* test if we need workspace */
    bool with_workspace = pooling_alg == pooling_max; // NOTE only for forward op

    auto ws_pd = with_workspace ? pool_fwd_pd.workspace_primitive_desc() : dst_mem.get_primitive_desc();
    auto ws_mem = with_workspace ? memory(ws_pd) : dst_mem;

    /* create forward op primitive */
    auto pool_op = with_workspace ?
       pooling_forward(pool_fwd_pd, src_mem, dst_mem, ws_mem) :
       pooling_forward(pool_fwd_pd, src_mem, dst_mem);

    // create network array
    std::vector<primitive> net;

    auto src_dims = src_md.data.dims;
    auto dst_dims = dst_md.data.dims;
    int batch_size = src_dims[0];
    printf("Input %dx%dx%d kernel %dx%dx%d channels=%d bs=%d\n",
           src_dims[2], src_dims[3], src_dims[4],
           kernel[0], kernel[1], kernel[2],
           dst_dims[1], batch_size
          );

    const int ntime = 25;
    float complexity = ((float)dst_dims[2])*dst_dims[3]*dst_dims[4]*kernel[0]*kernel[1]*kernel[2]*dst_dims[1];
    std::cout << "Total flops: " << complexity << "\n";

    for (int it = 0; it < ntime; it++) {
        net.push_back(pool_op);
    }
    // Execute
    auto t1 = Clock::now();
    stream(stream::kind::eager).submit(net).wait();
    auto t2 = Clock::now();

    float duration = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/ntime;
    std::cout << "Duration: " << duration << " ms" << "\n";
    std::cout << "MFlops/s: " << complexity/1000./1000./duration*1000.*batch_size << "\n";

}

void time_pooling_3d(algorithm pooling_alg,
                     const int nbatch, const int in_channels, const int out_channels,
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

    // assign input
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

    compute_fwd_pool(pooling_alg, src_memory, dst_memory,
                     strides, kernel, padding);

}

void test_pool_3d(algorithm pooling_alg, const int insize, const int bs=1) {
    const int ic=16;
    const int ih=insize, iw=insize, id=insize;
    const int kh=2, kw=2, kd=2;
    const int oh=ih-kh+1, ow=iw-kw+1, od=id-kd+1;
    time_pooling_3d(pooling_alg, bs, ic, ic, ih, iw, id, kh, kw, kd, oh, ow, od);
}

int main(int argc, char **argv) {
    try {
        auto pooling_alg = pooling_max;
        std::vector<int> in_sizes = {32, 64, 128};
        std::vector<int> batch_sizes = {1, 4, 8};
        for(std::vector<int>::iterator s = in_sizes.begin(); s != in_sizes.end(); ++s) {
            for(std::vector<int>::iterator mb = batch_sizes.begin(); mb != batch_sizes.end(); ++mb) {
                test_pool_3d(pooling_alg, *s, *mb);
            }
        }
    }
    catch(error& e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
