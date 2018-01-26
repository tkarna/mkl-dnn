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
#define TEST_NCDHW_TO_NDHWC

#define COMMON_CODE_PRE() \
    for (int a=0; a<A; a++){ \
        for (int b=0; b<B; b++) { \
            for (int c=0; c<C; c++) { \             
              for ( int d=0; d<D; d++) {  \
                for ( int e=0; e<E; e++ ) {  \
                    int indx = a*B*C*D*E + b*C*D*E + c*D*E + d*E + e;

#define COMMON_CODE_POST() \
                } \
              } \
            } \
        } \
    } 

#define NCDHW()   batch, channels, depth, height, width
#define NDHWC()   batch, depth, height, width, channels


void copy_array_5d( float* src_array, float* dst_array, int A, int B, int C, int D, int E) { 
  COMMON_CODE_PRE()
  dst_array[indx] = src_array[indx];
  COMMON_CODE_POST()
}

void compare_array_5d( float* src_array, float* dst_array, int A, int B, int C, int D, int E) { 
  float residual = 0;
  COMMON_CODE_PRE()
  residual += dst_array[indx] - src_array[indx];
  COMMON_CODE_POST()
  std::cout << "\nError Residual value = " << residual << "\n";
}

void assign_array_5d(std::string name, float* array, int A, int B, int C, int D, int E, int offset) {
  float count = 0;
  COMMON_CODE_PRE()
  array[indx] = count++ + offset;
  COMMON_CODE_POST()
}

void print_array_5d(std::string name, float* array, int A, int B, int C, int D, int E) {
    std::cout << name << ":" << std::endl;
    std::cout << "\nDims: " << A << " " << B << " " << C << " " << D << " " << E << "\n";
    COMMON_CODE_PRE()

                    std::cout << std::setw(6) << array[a*B*C*D*E + b*C*D*E + c*D*E + d*E + e];
                }
                std::cout << std::endl;
              }
              std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}


void compare_ncdhw_ndhwc( float * dt_ncdhw, float * dt_ndhwc, int A, int B, int C, int D, int E ) {
  float residual = 0.0;
  COMMON_CODE_PRE()
  int indx_ncdhw = indx;
  int indx_ndhwc = a*C*D*E*B +  c*D*E*B + d*E*B + e*B + b;
  residual += std::abs(dt_ncdhw[indx_ncdhw] - dt_ndhwc[indx_ndhwc] );   // should be correct
//  residual += std::abs(dt_ncdhw[indx_ncdhw] - dt_ndhwc[indx_ncdhw] );  // wrong

  COMMON_CODE_POST()
  std::cout<< "\n Comparing final data: " << residual; 
}

void simple_net_3d(){

    auto cpu_engine = engine(engine::cpu, 0);

// Defining dimensions.
    const int batch = 1;
    const int channels = 3;
    const int depth = 2;
    const int height = 5;
    const int width = 4;
  
    std::cout << "\n Dimensions are : batch = " << batch << ", channels = " << channels << ", depth = " << depth << " height = " << height << " width = " <<width;

// Dimensions of memory to be allocated 
    memory::dims dims_ncdhw = {NCDHW()}; 

    // User provided memory - in a vector of 1D format.
    // 1D allocations src, dst, weights and biases.
    std::vector<float> net_src(batch * channels * depth * height * width );
    std::vector<float> dst_src(batch * channels * depth * height * width );

    /* create memory for user data */
    auto dnn_ncdhw_memory = memory({{{dims_ncdhw}, memory::data_type::f32, memory::format::ncdhw}, cpu_engine}, net_src.data());    
    auto dnn_ndhwc_memory = memory({{{dims_ncdhw}, memory::data_type::f32, memory::format::ndhwc}, cpu_engine}, dst_src.data());   
    float *ncdhw_data = (float *)dnn_ncdhw_memory.get_data_handle();
    float *ndhwc_data = (float *)dnn_ndhwc_memory.get_data_handle();
    float* tmp_data = new float[ batch*depth*height*width*channels  ];

    std::vector<primitive> net;

    std::cout << "\nOriginal src array: \n"; 
#ifdef TEST_NCDHW_TO_NDHWC
    std::cout << "\nTesting NCDHW to NDHWC reorder: ";
    assign_array_5d("ncdhw_data", ncdhw_data, NCDHW(), 0  );
    print_array_5d("ncdhw_mem", ncdhw_data, NCDHW() );
    copy_array_5d(ncdhw_data, tmp_data, NCDHW() );
    // ncdhw to ndhwc and ndhwc to ncdhw again.
    net.push_back(reorder(dnn_ncdhw_memory, dnn_ndhwc_memory));  
    net.push_back(reorder(dnn_ndhwc_memory, dnn_ncdhw_memory));  
    stream(stream::kind::eager).submit(net).wait();
    print_array_5d("ndhwc_mem", ndhwc_data, NDHWC() );
    compare_array_5d(ncdhw_data, tmp_data, NCDHW() );
#else
    std::cout << "\nTesting NDHWC to NCDHW reorder: ";
    assign_array_5d("ndhwc_data", ndhwc_data, NDHWC(), 0  );
    print_array_5d("ndhwc_mem", ndhwc_data, NDHWC() );
    copy_array_5d(ndhwc_data, tmp_data, NDHWC() );
    net.push_back(reorder(dnn_ndhwc_memory, dnn_ncdhw_memory));  
    net.push_back(reorder(dnn_ncdhw_memory, dnn_ndhwc_memory));  
    stream(stream::kind::eager).submit(net).wait();
    print_array_5d("ncdhw_mem", ncdhw_data, NCDHW() );
    compare_array_5d(ndhwc_data, tmp_data, NDHWC() );

#endif    

    compare_ncdhw_ndhwc ( ncdhw_data, ndhwc_data, NCDHW()  );

    delete tmp_data;
}

int main(int argc, char **argv) {
    std::cout << "\nExecuting 3D reorder\n";
    try {
        simple_net_3d();
    }
    catch(error& e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}


