#===============================================================================
# Copyright 2016-2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(simple-net-c simple_net.c)
set_property(TARGET simple-net-c PROPERTY C_STANDARD 99)
target_link_libraries(simple-net-c ${LIB_NAME})
add_test(simple-net-c simple-net-c)
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
        simple-net-c.vcxproj.user @ONLY)
    set_property(TEST simple-net-c PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()

add_executable(simple-net-cpp simple_net.cpp)
set_property(TARGET simple-net-cpp PROPERTY CXX_STANDARD 11)
target_link_libraries(simple-net-cpp ${LIB_NAME})
add_test(simple-net-cpp simple-net-cpp)
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
        simple-net-cpp.vcxproj.user @ONLY)
    set_property(TEST simple-net-cpp PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()

add_executable(3d-simple-net-cpp 3d_simple_net.cpp)
set_property(TARGET 3d-simple-net-cpp PROPERTY CXX_STANDARD 11)
target_link_libraries(3d-simple-net-cpp ${LIB_NAME})
add_test(3d-simple-net-cpp 3d-simple-net-cpp)
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
      3d-simple-net-cpp.vcxproj.user @ONLY)
    set_property(TEST 3d-simple-net-cpp PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()

add_executable(3d_reorder_cpp 3d_reorder.cpp)
set_property(TARGET 3d_reorder_cpp PROPERTY CXX_STANDARD 11)
target_link_libraries(3d_reorder_cpp ${LIB_NAME})
add_test(3d_reorder_cpp 3d_reorder_cpp)
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
      3d_reorder_cpp.vcxproj.user @ONLY)
    set_property(TEST 3d_reorder_cpp PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()

add_executable(3d_reorder_filter_cpp 3d_reorder_filter.cpp)
set_property(TARGET 3d_reorder_filter_cpp PROPERTY CXX_STANDARD 11)
target_link_libraries(3d_reorder_filter_cpp ${LIB_NAME})
add_test(3d_reorder_filter_cpp 3d_reorder_filter_cpp)
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
      3d_reorder_filter_cpp.vcxproj.user @ONLY)
    set_property(TEST 3d_reorder_filter_cpp PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()

add_executable(test-convolution-3d-cpp test_convolution_3d.cpp)
set_property(TARGET test-convolution-3d-cpp PROPERTY CXX_STANDARD 11)
target_link_libraries(test-convolution-3d-cpp ${LIB_NAME})
add_test(test-convolution-3d-cpp test-convolution-3d-cpp)
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
      test-convolution-3d-cpp.vcxproj.user @ONLY)
    set_property(TEST test-convolution-3d-cpp PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()

add_executable(test-convolution-3d-fwd-ref-cpp test_convolution_3d_fwd_ref.cpp)
set_property(TARGET test-convolution-3d-fwd-ref-cpp PROPERTY CXX_STANDARD 11)
target_link_libraries(test-convolution-3d-fwd-ref-cpp ${LIB_NAME})
add_test(test-convolution-3d-fwd-ref-cpp test-convolution-3d-fwd-ref-cpp)
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
      test-convolution-3d-fwd-ref-cpp.vcxproj.user @ONLY)
    set_property(TEST test-convolution-3d-fwd-ref-cpp PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()

add_executable(test-convolution-3d-bkw-weights-ref-cpp test_convolution_3d_bkw_weights_ref.cpp)
set_property(TARGET test-convolution-3d-bkw-weights-ref-cpp PROPERTY CXX_STANDARD 11)
target_link_libraries(test-convolution-3d-bkw-weights-ref-cpp ${LIB_NAME})
add_test(test-convolution-3d-bkw-weights-ref-cpp test-convolution-3d-bkw-weights-ref-cpp)
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
      test-convolution-3d-bkw-weights-ref-cpp.vcxproj.user @ONLY)
    set_property(TEST test-convolution-3d-bkw-weights-ref-cpp PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()

add_executable(test-convolution-3d-bkw-data-ref-cpp test_convolution_3d_bkw_data_ref.cpp)
set_property(TARGET test-convolution-3d-bkw-data-ref-cpp PROPERTY CXX_STANDARD 11)
target_link_libraries(test-convolution-3d-bkw-data-ref-cpp ${LIB_NAME})
add_test(test-convolution-3d-bkw-data-ref-cpp test-convolution-3d-bkw-data-ref-cpp)
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
      test-convolution-3d-bkw-data-ref-cpp.vcxproj.user @ONLY)
    set_property(TEST test-convolution-3d-bkw-data-ref-cpp PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()

add_executable(simple-training-net-c simple_training_net.c)
set_property(TARGET simple-training-net-c PROPERTY C_STANDARD 99)
if(WIN32)
target_link_libraries(simple-training-net-c ${LIB_NAME})
else()
target_link_libraries(simple-training-net-c ${LIB_NAME} m)
endif()
add_test(simple-training-net-c simple-training-net-c)
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
        simple-training-net-c.vcxproj.user @ONLY)
    set_property(TEST simple-training-net-c PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()

add_executable(simple-training-net-cpp simple_training_net.cpp)
set_property(TARGET simple-training-net-cpp PROPERTY CXX_STANDARD 11)
if(WIN32)
target_link_libraries(simple-training-net-cpp ${LIB_NAME})
else()
target_link_libraries(simple-training-net-cpp ${LIB_NAME} m)
endif()
add_test(simple-training-net-cpp simple-training-net-cpp)
if(WIN32)
    configure_file(${CMAKE_SOURCE_DIR}/config_template.vcxproj.user
        simple-training-net-cpp.vcxproj.user @ONLY)
    set_property(TEST simple-training-net-cpp PROPERTY ENVIRONMENT "PATH=${CTESTCONFIG_PATH}")
endif()
