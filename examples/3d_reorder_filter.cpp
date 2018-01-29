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

#define oidhw()   out_channels, in_channels, depth, height, width
#define dhwio()   depth, height, width, in_channels, out_channels 


void copy_array_5d( float* src_array, float* dst_array, int A, int B, int C, int D, int E) { 
  COMMON_CODE_PRE()
  dst_array[indx] = src_array[indx];
  COMMON_CODE_POST()
}

bool compare_array_5d( float* src_array, float* dst_array, int A, int B, int C, int D, int E) { 
  float residual = 0; 
  COMMON_CODE_PRE()
  residual += dst_array[indx] - src_array[indx];
  COMMON_CODE_POST()
  std::cout << "\nError Residual value = " << residual << "\n";
  if ( residual < TOLERANCE)
    return true;
  else 
    return false;
}

void reset_array_5d( float* array, int A, int B, int C, int D, int E) {
  COMMON_CODE_PRE()
  array[indx] = 0; 
  COMMON_CODE_POST()
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


bool compare_oidhw_dhwio( float * dt_oidhw, float * dt_dhwio, int A, int B, int C, int D, int E ) {
  float residual = 0.0;
  COMMON_CODE_PRE()
  // 01234
  int indx_oidhw = indx;
  // 23410
  int indx_dhwio = a + A*(b + B*( e + E * ( d + D * ( c )  )   )) ;
  residual += std::abs(dt_oidhw[indx_oidhw] - dt_dhwio[indx_dhwio] );   // should be correct
//  residual += std::abs(dt_oidhw[indx_oidhw] - dt_dhwio[indx_oidhw] );  // wrong

  COMMON_CODE_POST()
  std::cout<< "\n Comparing final data: " << residual; 
  if ( residual < TOLERANCE)
    return true;
  else 
    return false;
}

void simple_net_3d(){

    auto cpu_engine = engine(engine::cpu, 0);

// Defining dimensions.
    const int out_channels = 4;
    const int in_channels = 3;
    const int depth = 2;
    const int height = 2;
    const int width = 2;
  
    std::cout << "\n Dimensions are : out_channels = " << out_channels << ", in_channels = " << in_channels << ", depth = " << depth << " height = " << height << " width = " <<width;

// Dimensions of memory to be allocated 
    memory::dims dims_oidhw = {oidhw()}; 

    // User provided memory - in a vector of 1D format.
    // 1D allocations src, dst, weights and biases.
    std::vector<float> net_src(out_channels * in_channels * depth * height * width );
    std::vector<float> dst_src(out_channels * in_channels * depth * height * width );

    /* create memory for user data */
    auto dnn_oidhw_memory = memory({{{dims_oidhw}, memory::data_type::f32, memory::format::oidhw}, cpu_engine}, net_src.data());    
    auto dnn_dhwio_memory = memory({{{dims_oidhw}, memory::data_type::f32, memory::format::dhwio}, cpu_engine}, dst_src.data());   
    float *oidhw_data = (float *)dnn_oidhw_memory.get_data_handle();
    float *dhwio_data = (float *)dnn_dhwio_memory.get_data_handle();
    float* tmp_data = new float[ out_channels*depth*height*width*in_channels  ];

    bool status1 = true;
    bool status2 = true;

    {
      std::vector<primitive> net;
      std::cout << "\nOriginal src array: \n"; 
      std::cout << "\nTesting oidhw to dhwio reorder: ";
      reset_array_5d( oidhw_data, oidhw()  );
      reset_array_5d( tmp_data, oidhw() );
      reset_array_5d( dhwio_data, dhwio()  );
      assign_array_5d("oidhw_data", oidhw_data, oidhw(), 0  );
      print_array_5d("oidhw_mem", oidhw_data, oidhw() );
      copy_array_5d(oidhw_data, tmp_data, oidhw() );
      // oidhw to dhwio and dhwio to oidhw again.
      net.push_back(reorder(dnn_oidhw_memory, dnn_dhwio_memory));  
      net.push_back(reorder(dnn_dhwio_memory, dnn_oidhw_memory));  
      stream(stream::kind::eager).submit(net).wait();
      print_array_5d("dhwio_mem", dhwio_data, dhwio() );
      status1 = status1 && compare_array_5d(oidhw_data, tmp_data, oidhw() );
      status1 = status1 && compare_oidhw_dhwio ( oidhw_data, dhwio_data, oidhw()  );
    }

    {
      reset_array_5d( oidhw_data, oidhw()  );
      reset_array_5d( tmp_data, dhwio() );
      reset_array_5d( dhwio_data, dhwio()  );

      std::vector<primitive> net;
      std::cout << "\nTesting dhwio to oidhw reorder: ";
      assign_array_5d("dhwio_data", dhwio_data, dhwio(), 0  );
      print_array_5d("dhwio_mem", dhwio_data, dhwio() );
      copy_array_5d(dhwio_data, tmp_data, dhwio() );
      net.push_back(reorder(dnn_dhwio_memory, dnn_oidhw_memory));  
      net.push_back(reorder(dnn_oidhw_memory, dnn_dhwio_memory));  
      stream(stream::kind::eager).submit(net).wait();
      print_array_5d("oidhw_mem", oidhw_data, oidhw() );
      status2 = status2 && compare_array_5d(dhwio_data, tmp_data, dhwio() );
      status2 = status2 && compare_oidhw_dhwio ( oidhw_data, dhwio_data, oidhw()  );
    }

    if ( status1 && status2 )
      std::cout << "\nTests passed.\n";
    else
      std::cout << "\nEither one or both tests failed.\n";

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


