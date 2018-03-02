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

void print_array_3d(std::string name, float* array, int n, int m, int l) {
    std::cout << name << ":" << std::endl;
    for (int i=0; i<n; i++){
        for (int j=0; j<m; j++) {
            for (int k=0; k<l; k++) {
                std::cout << std::setw(8) << std::setprecision(5) << array[m*l*i + l*j + k];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

bool check_result(std::string array_name, float* array, float* correct,
                  const int len, float tolerance, bool verbose=false) {
    /* Computes the average abs relative error in the output array */
    float rel_error = 0;
    for (int i = 0; i < len; i++) {
        float re = (array[i] - correct[i])/correct[i];
        if (verbose && std::abs(re) > tolerance) {
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
        std::cout << "  Relative error: " << rel_error << std::endl;
    }
    return success;
}

inline size_t ptr_off_f(const memory::desc &md, int mb, int ic, int id, int ih, int iw) {
    const int g = 0;
    const int G = 1; // NOTE currently without groups
    return ((((size_t)mb * md.data.dims[1] +
              g * md.data.dims[1]/G + ic) * md.data.dims[2] + id) *
              md.data.dims[3] + ih) * md.data.dims[4] + iw;
}

void compute_reference_fwd_pool(algorithm alg,
                                const memory &src_mem,
                                const memory &dst_mem,
                                const memory::dims &kernel,
                                const memory::dims &strides,
                                const memory::dims &padding) {

    float *src = (float*)src_mem.get_data_handle();
    float *dst = (float*)dst_mem.get_data_handle();

    auto src_pd = src_mem.get_primitive_desc();
    auto dst_pd = dst_mem.get_primitive_desc();

    auto src_md = src_pd.desc();
    auto dst_md = dst_pd.desc();

    const int MB = src_md.data.dims[0];
    const int ID = src_md.data.dims[2];
    const int IH = src_md.data.dims[3];
    const int IW = src_md.data.dims[4];

    const int OC = dst_md.data.dims[1];
    const int OD = dst_md.data.dims[2];
    const int OH = dst_md.data.dims[3];
    const int OW = dst_md.data.dims[4];

    const int KD = kernel[0];
    const int KH = kernel[1];
    const int KW = kernel[2];

    const int SD = strides[0];
    const int SH = strides[1];
    const int SW = strides[2];

    const int padD = padding[0];
    const int padT = padding[1];
    const int padL = padding[2];

    auto apply_offset = [=](int index, int offset) {
        return (index > offset) ? index - offset : 0;
    };

    auto ker_max = [=](float *d, int mb, int oc, int od, int oh, int ow) {
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    const int id = od * SD - padD + kd;
                    const int ih = oh * SH - padT + kh;
                    const int iw = ow * SW - padL + kw;

                    if (id < 0 || id >= ID) continue;
                    if (ih < 0 || ih >= IH) continue;
                    if (iw < 0 || iw >= IW) continue;

                    auto s = src[ptr_off_f(src_md, mb, oc, id, ih, iw)];
                    if (s > d[0]) {
                        d[0] = s;
//                         if (ws) {
//                             size_t off = ws_d.off(mb, oc, od, oh, ow);
//                             if (ws_dt == floatype::u8) {
//                                 ws[off] = kd*KH*KW + kh*KW + kw;
//                             } else {
//                                 assert(ws_dt == floatype::s32);
//                                 ((int *)ws)[off] = kd*KH*KW + kh*KW + kw;
//                             }
//                         }
                    }
                }
            }
        }
    };

    auto ker_avg = [=](float *d, int mb, int oc, int od, int oh, int ow) {
        auto id_start = apply_offset(od*SD, padD);
        auto ih_start = apply_offset(oh*SH, padT);
        auto iw_start = apply_offset(ow*SW, padL);
        auto id_end = std::min(od*SD - padD + KD, ID);
        auto ih_end = std::min(oh*SH - padT + KH, IH);
        auto iw_end = std::min(ow*SW - padL + KW, IW);

        auto num_summands = (alg == pooling_avg_include_padding) ? KD*KW*KH
            : (id_end - id_start)*(ih_end - ih_start)*(iw_end - iw_start);

        float dst = 0;
        for (int id = id_start; id < id_end; ++id) {
            for (int ih = ih_start; ih < ih_end; ++ih) {
                for (int iw = iw_start; iw < iw_end; ++iw) {
                    dst += src[ptr_off_f(src_md, mb, oc, id, ih, iw)];
                }
            }
        }
        // NOTE omitted rounding
        d[0] = ((float)dst / num_summands);
    };

    if (alg == pooling_max) {
#       pragma omp parallel for collapse(5) schedule(static)
        for (int mb = 0; mb < MB; ++mb) {
            for (int oc = 0; oc < OC; ++oc) {
                for (int od = 0; od < OD; ++od) {
                    for (int oh = 0; oh < OH; ++oh) {
                        for (int ow = 0; ow < OW; ++ow) {
                            float *d = &dst[ptr_off_f(dst_md, mb, oc, od, oh, ow)];
                            d[0] =  std::numeric_limits<float>::min();
//                             if (ws) {
//                                 ws[ws_d.off(mb, oc, od, oh, ow)] = 0;
//                             }
                            ker_max(d, mb, oc, od, oh, ow);
                        }
                    }
                }
            }
        }
    } else {
#       pragma omp parallel for collapse(5) schedule(static)
        for (int mb = 0; mb < MB; ++mb) {
            for (int oc = 0; oc < OC; ++oc) {
                for (int od = 0; od < OD; ++od) {
                    for (int oh = 0; oh < OH; ++oh) {
                        for (int ow = 0; ow < OW; ++ow) {
                            float *d = &dst[ptr_off_f(dst_md, mb, oc, od, oh, ow)];
                            d[0] = 0;
                            ker_avg(d, mb, oc, od, oh, ow);
                        }
                    }
                }
            }
        }
    }
}

void compute_reference_bkw_pool(algorithm alg,
                                const memory &diff_dst_mem,
                                const memory &diff_src_mem,
                                const memory::dims &kernel,
                                const memory::dims &strides,
                                const memory::dims &padding) {
    float *diff_src = (float*)diff_src_mem.get_data_handle();
    float *diff_dst = (float*)diff_dst_mem.get_data_handle();

    auto diff_src_pd = diff_src_mem.get_primitive_desc();
    auto diff_dst_pd = diff_dst_mem.get_primitive_desc();

    auto diff_src_md = diff_src_pd.desc();
    auto diff_dst_md = diff_dst_pd.desc();

    const int MB = diff_src_md.data.dims[0];
    const int IH = diff_src_md.data.dims[3];
    const int IW = diff_src_md.data.dims[4];
    const int ID = diff_src_md.data.dims[2];

    const int OC = diff_dst_md.data.dims[1];
    const int OD = diff_dst_md.data.dims[2];
    const int OH = diff_dst_md.data.dims[3];
    const int OW = diff_dst_md.data.dims[4];

    const int KD = kernel[0];
    const int KH = kernel[1];
    const int KW = kernel[2];

    const int SD = strides[0];
    const int SH = strides[1];
    const int SW = strides[2];

    const int padD = padding[0];
    const int padT = padding[1];
    const int padL = padding[2];

    auto ker_zero = [=](int _mb, int _oc) {
        for (int id = 0; id < ID; ++id) {
            for (int ih = 0; ih < IH; ++ih) {
                for (int iw = 0; iw < IW; ++iw) {
                    diff_src[ptr_off_f(diff_src_md, _mb, _oc, id, ih, iw)] = float(0);
                }
            }
        }
    };

    // NOTE omitting max kernel for now
    // auto ker_max = [=](const data_t *d, int mb, int oc, int od, int oh, int ow) {
    //     const size_t ws_off = ws_d.off(mb, oc, od, oh, ow);
    //     const int index = ws_d.data_type() == data_type::u8
    //         ? (int)ws[ws_off] : ((int *)ws)[ws_off];
    //     const int kd = index / (KH*KW);
    //     const int kw = (index % (KH*KW)) % KW;
    //     const int kh = (index % (KH*KW)) / KW;
    //     const int id = od * SD - padD + kd;
    //     const int ih = oh * SH - padT + kh;
    //     const int iw = ow * SW - padL + kw;
    //
    //     diff_src[diff_src_d.off(mb, oc, id, ih, iw)] += d[0];
    // };

    auto ker_avg = [=](const float *d, int mb, int oc, int od, int oh, int ow) {
        int id_start = std::max(od*SD - padD, 0);
        int ih_start = std::max(oh*SH - padT, 0);
        int iw_start = std::max(ow*SW - padL, 0);
        int id_end = std::min(od*SD - padD + KD, ID);
        int ih_end = std::min(oh*SH - padT + KH, IH);
        int iw_end = std::min(ow*SW - padL + KW, IW);

        // NOTE assuming alg == pooling_avg
        int num_summands =  (id_end - id_start)*(ih_end - ih_start)*(iw_end - iw_start);

        for (int id = id_start; id < id_end; ++id) {
            for (int ih = ih_start; ih < ih_end; ++ih) {
                for (int iw = iw_start; iw < iw_end; ++iw) {
                    diff_src[ptr_off_f(diff_src_md, mb, oc, id, ih, iw)] += d[0] / num_summands;
                }
            }
        }
    };

//     if (alg == pooling_max) {
// #       pragma omp parallel for collapse(2) schedule(static)
//         for (int mb = 0; mb < MB; ++mb) {
//             for (int oc = 0; oc < OC; ++oc) {
//                 ker_zero(mb, oc);
//                 for (int od = 0; od < OD; ++od) {
//                     for (int oh = 0; oh < OH; ++oh) {
//                         for (int ow = 0; ow < OW; ++ow) {
//                             const float *d =
//                                 &diff_dst[ptr_off_f(diff_dst_md, mb, oc, od, oh, ow)];
//                             ker_max(d, mb, oc, od, oh, ow);
//                         }
//                     }
//                 }
//             }
//         }
//     }
    if (alg == pooling_avg) {
#       pragma omp parallel for collapse(2) schedule(static)
        for (int mb = 0; mb < MB; ++mb) {
            for (int oc = 0; oc < OC; ++oc) {
                ker_zero(mb, oc);
                for (int od = 0; od < OD; ++od) {
                    for (int oh = 0; oh < OH; ++oh) {
                        for (int ow = 0; ow < OW; ++ow) {
                            const float *d =
                                &diff_dst[ptr_off_f(diff_dst_md, mb, oc, od, oh, ow)];
                            ker_avg(d, mb, oc, od, oh, ow);
                        }
                    }
                }
            }
        }
    } else {
        printf("Reference max pooling has not been implemented\n");
        exit(-1);
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

    // create network array
    std::vector<primitive> net;

    net.push_back(pool_fwd_op);

    if (direction == "both") {
        auto diff_dst_pd = diff_dst_mem.get_primitive_desc();
        auto diff_src_pd = diff_src_mem.get_primitive_desc();

        auto diff_dst_md = diff_dst_pd.desc();
        auto diff_src_md = diff_src_pd.desc();

        /* op descriptors */
        auto pool_bkw_desc = pooling_backward::desc(pooling_alg,
            diff_src_md, diff_dst_md, strides, kernel, padding, padding,
            padding_kind::zero);

        /* primitive op descriptors */
        auto pool_bkw_pd = pooling_backward::primitive_desc(pool_bkw_desc,
                                                            cpu_engine,
                                                            pool_fwd_pd);

        /* create forward op primitive */
        auto pool_bkw_op = pooling_backward(pool_bkw_pd, diff_dst_mem, diff_src_mem);

        net.push_back(pool_bkw_op);
    }

    // Execute
    stream(stream::kind::eager).submit(net).wait();

}

bool assert_pooling_3d(std::string direction, algorithm pooling_alg,
                       const int nbatch, const int channels,
                       const int in_height, const int in_width, const int in_depth,
                       const int ker_height,const int ker_width, const int ker_depth,
                       const int out_height, const int out_width, const int out_depth,
                       const int stride_height, const int stride_width, const int stride_depth,
                       const int pad_height, const int pad_width, const int pad_depth,
                       bool fill_with_floats, float tolerance, bool print_arrays = true){
    auto cpu_engine = engine(engine::cpu, 0);

    if (direction == "both" && pooling_alg == pooling_max) {
        std::cout << "Backward operator for pooling max has not been implemented" << std::endl;
        exit(-1);
    }
    // Dimensions of memory to be allocated
    memory::dims src_dims = {nbatch, channels, in_depth, in_height, in_width};
    memory::dims dst_dims = {nbatch, channels, out_depth, out_height, out_width};

    auto strides = {stride_depth, stride_height, stride_width};
    auto padding = {pad_depth, pad_height, pad_width};
    auto kernel = {ker_depth, ker_height, ker_width};

    std::vector<float> vect_src(std::accumulate(src_dims.begin(),
        src_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_dst(std::accumulate(dst_dims.begin(),
        dst_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_ref_dst(std::accumulate(dst_dims.begin(),
        dst_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_diff_src(std::accumulate(src_dims.begin(),
        src_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_diff_dst(std::accumulate(dst_dims.begin(),
        dst_dims.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> vect_ref_diff_src(std::accumulate(src_dims.begin(),
        src_dims.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    auto src_memory = memory({{{src_dims},
                             memory::data_type::f32, memory::format::ncdhw},
                             cpu_engine}, vect_src.data());
    auto dst_memory = memory({{{dst_dims},
                             memory::data_type::f32, memory::format::ncdhw},
                             cpu_engine}, vect_dst.data());
    auto ref_dst_memory = memory({{{dst_dims},
                                 memory::data_type::f32, memory::format::ncdhw},
                                 cpu_engine}, vect_ref_dst.data());

    auto diff_src_memory = memory({{{src_dims},
                             memory::data_type::f32, memory::format::ncdhw},
                             cpu_engine}, vect_diff_src.data());
    auto diff_dst_memory = memory({{{dst_dims},
                             memory::data_type::f32, memory::format::ncdhw},
                             cpu_engine}, vect_diff_dst.data());
    auto ref_diff_src_memory = memory({{{src_dims},
                             memory::data_type::f32, memory::format::ncdhw},
                             cpu_engine}, vect_ref_diff_src.data());

    // fill input array with random numbers
    if (fill_with_floats) {
        // floats between 1 and 25
        for (size_t i = 0; i < vect_src.size(); i++)
            vect_src[i] = 1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(25-1)));
        for (size_t i = 0; i < vect_diff_dst.size(); i++)
            vect_diff_dst[i] = 1 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(25-1)));
    } else {
        // ints between 1 and 25
        for (size_t i = 0; i < vect_src.size(); i++)
            vect_src[i] = rand() % 25 + 1.0;
        for (size_t i = 0; i < vect_diff_dst.size(); i++)
            vect_diff_dst[i] = rand() % 25 + 1.0;
    }

    bool success = true;

    auto alg_str = pooling_alg == pooling_max ? "max" : "avg";
    auto float_str = fill_with_floats ? "float" : "int";
    printf("%dx%dx%d k=%dx%dx%d %s ch=%d bs=%d (%s)\n",
           in_height, in_width, in_depth,
           ker_height, ker_width, ker_depth, alg_str,
           channels, nbatch, float_str
          );

    /* Compute reference solution */
    compute_reference_fwd_pool(pooling_alg, src_memory, ref_dst_memory,
                               kernel, strides, padding);

    // Print the output matrix
    if (print_arrays) {
        print_array_3d("Input", vect_src.data(), src_dims[2], src_dims[3], src_dims[4]);
        print_array_3d("Reference output", vect_ref_dst.data(), dst_dims[2], dst_dims[3], dst_dims[4]);
    }

    compute_pool(direction, pooling_alg, src_memory, dst_memory,
                 diff_src_memory, diff_dst_memory,
                 kernel, strides, padding);

    if (print_arrays) {
        // Print the output matrix
        print_array_3d("Output", vect_dst.data(), dst_dims[2], dst_dims[3], dst_dims[4]);
    }
    // Compute error
    success = success && check_result("fwd output", vect_dst.data(), vect_ref_dst.data(), vect_ref_dst.size(), tolerance);

    if (direction == "both") {
        compute_reference_bkw_pool(pooling_alg, diff_dst_memory, ref_diff_src_memory,
                                   kernel, strides, padding);

        // Print the output matrix
        if (print_arrays) {
            print_array_3d("Diff output", vect_diff_dst.data(), dst_dims[2], dst_dims[3], dst_dims[4]);
            print_array_3d("Reference diff input", vect_ref_diff_src.data(), src_dims[2], src_dims[3], src_dims[4]);
        }

        if (print_arrays) {
            // Print the output matrix
            print_array_3d("Diff input", vect_diff_src.data(), src_dims[2], src_dims[3], src_dims[4]);
        }
        // Compute error
        success = success && check_result("bkw output", vect_diff_src.data(), vect_ref_diff_src.data(), vect_ref_diff_src.size(), tolerance);
    }
    return success;
}

bool test_pool_simple_3d(std::string direction, algorithm pooling_alg, const int insize) {
    const int bs=1;
    const int ic=1;
    const int ih=insize, iw=insize, id=insize;
    const int kh=2, kw=2, kd=2;
    const int oh=ih-kh+1, ow=iw-kw+1, od=id-kd+1;
//     return assert_pooling_3d(direction, pooling_alg, bs, ic, ih, iw, id, kh, kw, kd, oh, ow, od, false, 1e-16);
    return assert_pooling_3d(direction, pooling_alg, bs, ic, ih, iw, id,
                    kh, kw, kd, oh, ow, od,
                    1, 1, 1, 0, 0, 0, false, 1e-16);
}

bool test_pool_3d(std::string direction, algorithm pooling_alg,
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
    float tol = fill_with_floats ? 1e-5 : 1e-25;
    return assert_pooling_3d(direction, pooling_alg, bs, channels, ih, iw, id,
                    kh, kw, kd, oh, ow, od,
                    sh, sw, sd, ph, pw, pd, fill_with_floats, tol, false);
}

int main(int argc, char **argv) {
    bool success = true;
    try {
        test_pool_simple_3d("both", pooling_avg, 4);

        // 32, 64 cubes (additional -- not in the applications)
        std::vector<int> in_sizes = {32, 64};
        std::vector<int> kernel_sizes = {2, 3};
        for(std::vector<int>::iterator s = in_sizes.begin(); s != in_sizes.end(); ++s) {
            for(std::vector<int>::iterator k = kernel_sizes.begin(); k != kernel_sizes.end(); ++k) {
                success = success && test_pool_3d("fwd", pooling_max, {*s ,*s, *s}, 32, {*k, *k, *k}, {1, 1, 1}, {0, 0, 0}, 1, false);
                success = success && test_pool_3d("fwd", pooling_max, {*s ,*s, *s}, 32, {*k, *k, *k}, {1, 1, 1}, {0, 0, 0}, 1, true);
                success = success && test_pool_3d("both", pooling_avg, {*s ,*s, *s}, 32, {*k, *k, *k}, {1, 1, 1}, {0, 0, 0}, 1, false);
                success = success && test_pool_3d("both", pooling_avg, {*s ,*s, *s}, 32, {*k, *k, *k}, {1, 1, 1}, {0, 0, 0}, 1, true);
            }
        }

        // cosmoflow layers
        //                         input size      ch  kernel     stride     padding   batch
        success = success && test_pool_3d("both", pooling_avg, {126, 126, 126}, 16, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, 1);
        success = success && test_pool_3d("both", pooling_avg, { 60,  60,  60}, 32, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, 1);

        // medical imaging layers
        // success = success && test_pool_3d("fwd", pooling_max, {334, 300, 396}, 32, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, 1);
        // success = success && test_pool_3d("fwd", pooling_max, {163, 146, 194}, 32, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, 1);
        success = success && test_pool_3d("both", pooling_avg, { 79,  69,  93}, 32, {2, 3, 3}, {1, 1, 1}, {0, 0, 0}, 1, false);
        success = success && test_pool_3d("both", pooling_avg, { 79,  69,  93}, 32, {2, 3, 3}, {1, 1, 1}, {0, 0, 0}, 1, true);

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
