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

void compute_pool(std::string direction,
                  algorithm pooling_alg,
                  const memory &src_mem,
                  const memory &dst_mem,
                  const memory &diff_src_mem,
                  const memory &diff_dst_mem,
                  const memory::dims &kernel,
                  const memory::dims &strides,
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
    auto pool_fwd_op = with_workspace ?
       pooling_forward(pool_fwd_pd, src_mem, dst_mem, ws_mem) :
       pooling_forward(pool_fwd_pd, src_mem, dst_mem);

    std::vector<primitive> operators;
    operators.push_back(pool_fwd_op);

    if (direction == "both") {
        auto diff_dst_pd = diff_dst_mem.get_primitive_desc();
        auto diff_src_pd = diff_src_mem.get_primitive_desc();

        auto diff_dst_md = diff_dst_pd.desc();
        auto diff_src_md = diff_src_pd.desc();

        /* op descriptors */
        auto pool_bwd_desc = pooling_backward::desc(pooling_alg,
            diff_src_md, diff_dst_md, strides, kernel, padding, padding,
            padding_kind::zero);

        /* primitive op descriptors */
        auto pool_bwd_pd = pooling_backward::primitive_desc(pool_bwd_desc,
                                                            cpu_engine,
                                                            pool_fwd_pd);

        /* create forward op primitive */
        auto pool_bwd_op = pooling_backward(pool_bwd_pd, diff_dst_mem, diff_src_mem);
        operators.push_back(pool_bwd_op);
    }

    // create network array
    std::vector<primitive> net;

    auto src_dims = src_md.data.dims;
    auto dst_dims = dst_md.data.dims;
    int batch_size = src_dims[0];
    std::string alg_str = pooling_alg == pooling_avg ? "avg" : "max";
    printf("Alg:%s Input %dx%dx%d kernel %dx%dx%d channels=%d bs=%d\n",
           alg_str.c_str(),
           src_dims[2], src_dims[3], src_dims[4],
           kernel[0], kernel[1], kernel[2],
           dst_dims[1], batch_size
          );

    const double ntime_target = 1.0e3; // Time in ms
    int nruns = 2;
    const int max_nruns = 1e9;
    const double overshoot = 1.2;
    double elapsed = 0.0;

    while(elapsed < ntime_target) {

        for (int it = 0; it < nruns; it++) {
            net.push_back(operators[0]);
        }

        // Execute
        auto t1 = Clock::now();
        stream(stream::kind::eager).submit(net).wait();
        auto t2 = Clock::now();

        elapsed = (double)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        if(elapsed < ntime_target)
        {
            std::cout << " [took " << elapsed << " discarding....]" << std::endl;
            nruns = std::min(std::max((double)nruns+1.0, nruns*overshoot*ntime_target/elapsed), (double)max_nruns);
        }
    }

    // forward complexity
    float complexity = ((float)dst_dims[2])*dst_dims[3]*dst_dims[4]*kernel[0]*kernel[1]*kernel[2]*dst_dims[1];
    auto speed = complexity/1000./1000./1000./(elapsed/nruns)*1000.*batch_size;
    std::cout << "Total flops: " << complexity << std::endl;

    std::cout << "Total elapsed: " << elapsed << " ms" << std::endl;
    std::cout << "#iter: " << nruns << std::endl;
    std::cout << "Avg duration: " << elapsed/nruns << " ms" << std::endl;
    std::cout << "GFlops/s: " << speed << std::endl;

    printf("CSV,fwd,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%le,%le,%d,%le,%le\n",
           alg_str.c_str(),
           src_dims[2], src_dims[3], src_dims[4],
           kernel[0], kernel[1], kernel[2],
           strides[0], strides[1], strides[2],
           dst_dims[1], batch_size,
           complexity, elapsed, nruns,elapsed/nruns,complexity/1000./1000./(elapsed/nruns)*1000.*batch_size);

    if (direction == "both") {
        const double ntime_target = 1.0e3; // Time in ms
        int nruns = 2;
        const int max_nruns = 1e9;
        const double overshoot = 1.2;
        double elapsed = 0.0;

        while(elapsed < ntime_target) {

            for (int it = 0; it < nruns; it++) {
                net.push_back(operators[1]);
            }

            // Execute
            auto t1 = Clock::now();
            stream(stream::kind::eager).submit(net).wait();
            auto t2 = Clock::now();

            elapsed = (double)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            if(elapsed < ntime_target)
            {
                std::cout << " [took " << elapsed << " discarding....]" << std::endl;
                nruns = std::min(std::max((double)nruns+1.0, nruns*overshoot*ntime_target/elapsed), (double)max_nruns);
            }
        }

        // backward complexity
        float complexity = ((float)dst_dims[2])*dst_dims[3]*dst_dims[4]*kernel[0]*kernel[1]*kernel[2]*dst_dims[1];
        auto speed = complexity/1000./1000./1000./(elapsed/nruns)*1000.*batch_size;
        std::cout << "Total flops: " << complexity << std::endl;

        std::cout << "Total elapsed: " << elapsed << " ms" << std::endl;
        std::cout << "#iter: " << nruns << std::endl;
        std::cout << "Avg duration: " << elapsed/nruns << " ms" << std::endl;
        std::cout << "GFlops/s: " << speed << std::endl;

        printf("CSV,bwd,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%le,%le,%d,%le,%le\n",
            alg_str.c_str(),
            src_dims[2], src_dims[3], src_dims[4],
            kernel[0], kernel[1], kernel[2],
            strides[0], strides[1], strides[2],
            dst_dims[1], batch_size,
            complexity, elapsed, nruns,elapsed/nruns,speed);
    }
}


void time_pooling_3d(std::string direction, algorithm pooling_alg,
                     const int nbatch, const int in_channels,
                     const int in_height, const int in_width, const int in_depth,
                     const int ker_height,const int ker_width, const int ker_depth,
                     const int out_height, const int out_width, const int out_depth,
                     const int stride_height, const int stride_width, const int stride_depth,
                     const int pad_height, const int pad_width, const int pad_depth,
                     bool print_arrays = true){

    auto cpu_engine = engine(engine::cpu, 0);

    // Dimensions of memory to be allocated
    memory::dims src_dims = {nbatch, in_channels, in_depth, in_height, in_width};
    memory::dims dst_dims = {nbatch, in_channels, out_depth, out_height, out_width};

    auto strides = {stride_depth, stride_height, stride_width};
    auto padding = {pad_depth, pad_height, pad_width};
    auto kernel = {ker_depth, ker_height, ker_width};

    std::vector<float> vect_src(std::accumulate(src_dims.begin(),
        src_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_dst(std::accumulate(dst_dims.begin(),
        dst_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_diff_src(std::accumulate(src_dims.begin(),
        src_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_diff_dst(std::accumulate(dst_dims.begin(),
        dst_dims.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    auto src_memory = memory({{{src_dims},
                             memory::data_type::f32, memory::format::ncdhw},
                             cpu_engine}, vect_src.data());
    auto dst_memory = memory({{{dst_dims},
                             memory::data_type::f32, memory::format::ncdhw},
                             cpu_engine}, vect_dst.data());

    auto diff_src_memory = memory({{{src_dims},
                             memory::data_type::f32, memory::format::ncdhw},
                             cpu_engine}, vect_diff_src.data());
    auto diff_dst_memory = memory({{{dst_dims},
                             memory::data_type::f32, memory::format::ncdhw},
                             cpu_engine}, vect_diff_dst.data());

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

    compute_pool(direction, pooling_alg, src_memory, dst_memory,
                 diff_src_memory, diff_dst_memory,
                 kernel, strides, padding);

}

void test_pool_3d(std::string direction, algorithm pooling_alg,
                  std::vector<int> insize, const int channels,
                  std::vector<int> kernel, std::vector<int> strides,
                  std::vector<int> padding, const int bs=1,
                  bool fill_with_floats=true
                 ) {
    const int ih=insize[0], iw=insize[1], id=insize[2];
    const int kh=kernel[0], kw=kernel[1], kd=kernel[2];
    const int sh=strides[0], sw=strides[1], sd=strides[2];
    const int ph=padding[0], pw=padding[1], pd=padding[2];
    const int oh=(ih-kh+2*ph)/sh+1, ow=(iw-kw+2*pw)/sw+1, od=(id-kd+2*pd)/sd+1;
    time_pooling_3d(direction, pooling_alg, bs, channels, ih, iw, id,
                    kh, kw, kd, oh, ow, od,
                    sh, sw, sd, ph, pw, pd);
}

int main(int argc, char **argv) {
    try {
        printf("CSV,op,alg,id,ih,iw,kd,kh,kw,sd,sh,sw,c,bs,exp-flops,elapsed-ms,iters,avg-ms,eff-flops\n");

        // cosmoflow layers
        //                         input size      ch  kernel     stride     padding   batch
        test_pool_3d("both", pooling_avg, {126, 126, 126}, 16, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, 1);
        test_pool_3d("both", pooling_avg, { 60,  60,  60}, 32, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, 1);

        // medical imaging layers
        test_pool_3d("fwd", pooling_max, {334, 300, 396}, 32, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, 1);
        test_pool_3d("fwd", pooling_max, {163, 146, 194}, 32, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, 1);
        test_pool_3d("both", pooling_avg, { 79,  69,  93}, 32, {2, 3, 3}, {1, 1, 1}, {0, 0, 0}, 1);

        // 32, 64 cubes (additional -- not in the applications)
        std::vector<int> in_sizes = {32, 64};
        std::vector<int> kernel_sizes = {2, 3};
        for(std::vector<int>::iterator s = in_sizes.begin(); s != in_sizes.end(); ++s) {
            for(std::vector<int>::iterator k = kernel_sizes.begin(); k != kernel_sizes.end(); ++k) {
                test_pool_3d("both", pooling_avg, {*s ,*s, *s}, 32, {*k, *k, *k}, {1, 1, 1}, {0, 0, 0});
            }
        }
    }
    catch(error& e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
