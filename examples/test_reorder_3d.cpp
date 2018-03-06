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
#include <cmath>
#include <stdlib.h>

using namespace mkldnn;

const float TOLERANCE=1e-16;

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
        std::cout << "OK" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
        std::cout << "  Len: " << len << "  Nerrs: " << nerr;
        std::cout << "  Relative error: " << rel_error << std::endl;
    }
    return success;
}

/* function to fill input arrays */
float fill_generator(const int n, const int c, const int d, const int h, const int w) {
    return 1 + (n % 12) + (c % 7) + (d % 10) + (h + 4) + (w % 25);
}

void fill_ncdhw(float* array, const int N, const int C, const int D, const int H, const int W) {
    for (int n=0; n < N; n++) {
    for (int c=0; c < C; c++) {
    for (int d=0; d < D; d++) {
    for (int h=0; h < H; h++) {
    for (int w=0; w < W; w++) {
        const int offset = (((n*C + c)*D + d)*H + h)*W + w;
        array[offset] = fill_generator(n, c, d, h, w);
    } } } } }
}

void fill_ndhwc(float* array, const int N, const int C, const int D, const int H, const int W) {
    for (int n=0; n < N; n++) {
    for (int d=0; d < D; d++) {
    for (int h=0; h < H; h++) {
    for (int w=0; w < W; w++) {
    for (int c=0; c < C; c++) {
        const int offset = (((n*D + d)*H + h)*W + w)*C + c;
        array[offset] = fill_generator(n, c, d, h, w);
    } } } } }
}

void fill_nCdhw16c(float* array, const int N, const int C, const int D, const int H, const int W) {
    const int NBLOCK = 16;
    const int CB = C/NBLOCK;
    for (int n=0; n < N; n++) {
    for (int cb=0; cb < CB; cb++) {
    for (int d=0; d < D; d++) {
    for (int h=0; h < H; h++) {
    for (int w=0; w < W; w++) {
    for (int c=0; c < NBLOCK; c++) {
        const int offset = ((((n*CB + cb)*D + d)*H + h)*W + w)*NBLOCK + c;
        array[offset] = fill_generator(n, cb*NBLOCK + c, d, h, w);
    } } } } } }
}

void fill_array(memory::format fmt, float* array, const int N, const int C, const int D, const int H, const int W) {
    if (fmt == memory::format::ncdhw) {
        fill_ncdhw(array, N, C, D, H, W);
    } else if (fmt == memory::format::ndhwc) {
        fill_ndhwc(array, N, C, D, H, W);
    } else if (fmt == memory::format::nCdhw16c) {
        fill_nCdhw16c(array, N, C, D, H, W);
    } else {
        printf("Unknown memory format %d\n", (int)fmt);
        exit(-1);
    }
}

bool test_reorder(memory::format ifmt, memory::format ofmt){

    auto cpu_engine = engine(engine::cpu, 0);

    const int MB = 1;
    const int IC = 16;
    const int ID = 12;
    const int IH = 15;
    const int IW = 8;

    memory::dims dims_ncdhw = {MB, IC, ID, IH, IW};

    const size_t array_len = MB * IC * ID * IH * IW;
    std::vector<float> vect_src(array_len);
    std::vector<float> vect_dst(array_len);
    std::vector<float> vect_ref(array_len);

    auto src_memory = memory({{{dims_ncdhw}, memory::data_type::f32, ifmt}, cpu_engine}, vect_src.data());
    auto dst_memory = memory({{{dims_ncdhw}, memory::data_type::f32, ofmt}, cpu_engine}, vect_dst.data());

    bool success = true;
    std::vector<primitive> net;

    /* forward */
    std::cout << "Testing reorder " << ifmt << " -> " << ofmt << " ";
    fill_array(ifmt, vect_src.data(), MB, IC, ID, IH, IW);
    fill_array(ofmt, vect_ref.data(), MB, IC, ID, IH, IW);
    vect_dst = {0};
    net.clear();
    net.push_back(reorder(src_memory, dst_memory));
    stream(stream::kind::eager).submit(net).wait();
    success = success && check_result("", vect_dst.data(), vect_ref.data(), vect_ref.size(), 1e-16);

    /* backward */
    std::cout << "Testing reorder " << ofmt << " -> " << ifmt << " ";
    fill_array(ofmt, vect_dst.data(), MB, IC, ID, IH, IW);
    fill_array(ifmt, vect_ref.data(), MB, IC, ID, IH, IW);
    vect_src = {0};
    net.clear();
    net.push_back(reorder(dst_memory, src_memory));
    stream(stream::kind::eager).submit(net).wait();
    success = success && check_result("", vect_src.data(), vect_ref.data(), vect_ref.size(), 1e-16);

    return success;
}

int main(int argc, char **argv) {
    bool success = true;
    try {
        success = success && test_reorder(memory::format::ncdhw, memory::format::ndhwc);
        success = success && test_reorder(memory::format::ncdhw, memory::format::nCdhw16c);
        success = success && test_reorder(memory::format::ndhwc, memory::format::nCdhw16c);

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


