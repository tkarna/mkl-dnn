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

void time_bkw_weights_convolution(const int nbatch, const int in_channels,
                                  const int out_channels, const int in_height,
                                  const int in_width, const int in_depth,
                                  const int weights_height,
                                  const int weights_width,
                                  const int weights_depth, const int out_height,
                                  const int out_width, const int out_depth,
                                  const int stride_height, const int stride_width, const int stride_depth,
                                  const int pad_height, const int pad_width, const int pad_depth,
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

    auto strides = {stride_depth, stride_height, stride_width};
    auto padding = {pad_depth, pad_height, pad_width};
    auto dilation = {0, 0, 0};

    auto diff_dst_mem = memory({{{dst_dims}, memory::data_type::f32, memory::format::ncdhw}, cpu_engine}, vect_diff_dst.data());
    auto diff_weights_mem = memory({{{weights_dims}, memory::data_type::f32,memory::format::oidhw}, cpu_engine}, vect_diff_weights.data());
    auto diff_bias_mem = memory({{{bias_dims}, memory::data_type::f32, memory::format::x}, cpu_engine}, vect_diff_bias.data());
    auto src_mem = memory({{{src_dims}, memory::data_type::f32, memory::format::ncdhw}, cpu_engine}, vect_src.data());

    auto src_pd = src_mem.get_primitive_desc();
    auto diff_dst_pd = diff_dst_mem.get_primitive_desc();
    auto diff_weights_pd = diff_weights_mem.get_primitive_desc();
    auto diff_bias_pd = diff_bias_mem.get_primitive_desc();

    auto src_md = src_pd.desc();
    auto diff_dst_md = diff_dst_pd.desc();
    auto diff_weights_md = diff_weights_pd.desc();

    // define memory descriptors
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
    // printf("data src format: %d\n", src_fmt);
    // printf("conv src format: %d\n", conv_src_fmt);
    // printf("src format match: %d\n", conv_src_fmt == src_fmt);

    auto weights_fmt = diff_weights_md.data.format;
    auto conv_weights_fmt = bkww_pd.diff_weights_primitive_desc().desc().data.format;
    bool weights_needs_reorder = conv_weights_fmt != weights_fmt;
    // printf("data weights format: %d\n", weights_fmt);
    // printf("conv weights format: %d\n", conv_weights_fmt);
    // printf("weights format match: %d\n", conv_weights_fmt == weights_fmt);

    auto dst_fmt = diff_dst_md.data.format;
    auto conv_dst_fmt = bkww_pd.diff_dst_primitive_desc().desc().data.format;
    bool dst_needs_reorder = conv_dst_fmt != dst_fmt;
    // printf("data dst format: %d\n", dst_fmt);
    // printf("conv dst format: %d\n", conv_dst_fmt);
    // printf("dst format match: %d\n", conv_dst_fmt == dst_fmt);

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

    // create network array
    std::vector<primitive> net;

    float complexity = 2*((float)out_height)*out_width*out_depth*weights_height*weights_width*weights_depth*in_channels*out_channels;

    const int ntime = 10;
    if (src_needs_reorder) {
        auto op = reorder(src_mem, reorder_src_mem);
        net.clear();
        const int ntime = 1;
        for (int it = 0; it < ntime; it++) {
            net.push_back(op);
        }
        // auto t1 = Clock::now();
        // Execute
        stream(stream::kind::eager).submit(net).wait();
        // auto t2 = Clock::now();
        // float duration = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/ntime;
        // std::cout << "src reorder: duration=" << duration << " ms" << "\n";
    }
    if (dst_needs_reorder) {
        auto op = reorder(diff_dst_mem, reorder_diff_dst_mem);
        net.clear();
        const int ntime = 1;
        for (int it = 0; it < ntime; it++) {
            net.push_back(op);
        }
        // auto t1 = Clock::now();
        // Execute
        stream(stream::kind::eager).submit(net).wait();
        // auto t2 = Clock::now();
        // float duration = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/ntime;
        // std::cout << "diff dst reorder: duration=" << duration << " ms" << "\n";
    }

    net.clear();
    for (int it = 0; it < ntime; it++) {
        net.push_back(bkww_op);
    }
    auto t1 = Clock::now();
    // Execute
    stream(stream::kind::eager).submit(net).wait();
    auto t2 = Clock::now();
    float duration = (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/ntime;
    float speed = complexity/1000./1000./1000./duration*1000.*nbatch;

    printf("bs=%2d ic=%3d oc=%3d %2dx%2dx%2d w=%dx%dx%d: %8.2f GFlops/s\n",
        nbatch, in_channels, out_channels, in_height, in_width, in_depth,
        weights_height, weights_width, weights_depth, speed);

}

void test_bkww_conv(const int bs, std::vector<int> insize, std::vector<int> kernel,
                    const int ic, const int oc, std::vector<int> stride,
                    std::vector<int> pad) {
    int ih = insize[0], iw = insize[1], id = insize[2];
    int kh = kernel[0], kw = kernel[1], kd = kernel[2];
    int sh = stride[0], sw = stride[1], sd = stride[2];
    int ph = pad[0], pw = pad[1], pd = pad[2];
    printf("bs=%d %dx%dx%d w=%dx%dx%d st=%dx%dx%d pd=%dx%dx%d ic=%2d oc=%2d ",
           bs, ih, iw, id,
           kh, kw, kd,
           sh, sw, sd,
           ph, pw, pd,
           ic, oc);
    const int oh=(ih-kh+2*ph)/sh+1, ow=(iw-kw+2*pw)/sw+1, od=(id-kd+2*pd)/sd+1;
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
    time_bkw_weights_convolution(bs, ic, oc,
                                 ih, iw, id,
                                 kh, kw, kd,
                                 oh, ow, od,
                                 sh, sw, sd,
                                 ph, pw, pd,
                                 in_diff_dst,
                                 test_bias, print_arrays);
}


int main(int argc, char **argv) {
    try {
        // std::vector<int> in_sizes = {32, 64};
        // std::vector<int> in_channels = {16, 32};
        // std::vector<int> out_channels = {16, 32};
        // std::vector<int> batch_sizes = {1, 8};
        // for(std::vector<int>::iterator s = in_sizes.begin(); s != in_sizes.end(); ++s) {
        // for(std::vector<int>::iterator ic = in_channels.begin(); ic != in_channels.end(); ++ic) {
        // for(std::vector<int>::iterator oc = out_channels.begin(); oc != out_channels.end(); ++oc) {
        //     for(std::vector<int>::iterator mb = batch_sizes.begin(); mb != batch_sizes.end(); ++mb) {
        //         test_bkww_conv(*mb, {*s, *s, *s}, {3, 3, 3}, *ic, *oc, {1, 1 ,1}, {0, 0, 0});
        //
        //     }
        // }}}

        // cosmoflow topology
        test_bkww_conv(1, {128, 128, 128}, {3, 3, 3},   1,  16, {1, 1, 1}, {0, 0, 0});
        test_bkww_conv(1, { 63,  63,  63}, {4, 4, 4},  16,  32, {1, 1, 1}, {0, 0, 0});
        test_bkww_conv(1, { 30,  30,  30}, {4, 4, 4},  32,  64, {2, 2, 2}, {0, 0, 0});
        test_bkww_conv(1, { 14,  14,  14}, {3, 3, 3},  64,  64, {1, 1, 1}, {0, 0, 0});
        test_bkww_conv(1, { 12,  12,  12}, {2, 2, 2},  64, 128, {1, 1, 1}, {0, 0, 0});
        test_bkww_conv(1, { 11,  11,  11}, {2, 2, 2}, 128, 128, {1, 1, 1}, {0, 0, 0});

        // cosmoflow topology (smaller problem size)
        test_bkww_conv(1, {64, 64, 64}, {3, 3, 3},   1,  16, {1, 1, 1}, {0, 0, 0});

        // medical imaging topology
        // test_bkww_conv(1, {336, 304, 400}, {3, 5, 5},  1, 32, {1, 1, 1}, {0, 0, 0});
        // test_bkww_conv(1, {167, 150, 198}, {3, 3, 3}, 32, 32, {1, 1, 1}, {0, 0, 0});
        // test_bkww_conv(1, {165, 148, 196}, {3, 3, 3}, 32, 32, {1, 1, 1}, {0, 0, 0});
        // test_bkww_conv(1, { 81,  73,  97}, {2, 3, 3}, 32, 48, {1, 1, 1}, {0, 0, 0});
        // test_bkww_conv(1, { 80,  71,  95}, {2, 3, 3}, 48, 48, {1, 1, 1}, {0, 0, 0});
        // test_bkww_conv(1, { 78,  67,  91}, {1, 1, 1}, 48,  2, {1, 1, 1}, {0, 0, 0});
    }
    catch(error& e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
