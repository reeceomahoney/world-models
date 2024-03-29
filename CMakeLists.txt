cmake_minimum_required(VERSION 3.10)

project(raisim_gym)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_PREFIX_PATH 
        $ENV{LOCAL_INSTALL}
        ~/.virtualenvs/wm/lib/python3.8/site-packages/pybind11/share/cmake/pybind11)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/dreamer/raisim_gym/build)

####################
### dependencies ###
####################
set(Dependencies)

find_package(pybind11)

find_package(Eigen3 CONFIG REQUIRED)

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

find_package(raisim CONFIG REQUIRED)

include_directories(
        ${EIGEN3_INCLUDE_DIRS}
        dreamer/raisim_gym/dependencies/actuation_dynamics_inference/include
        dreamer/raisim_gym/dependencies/networks_minimal/include
        dreamer/raisim_gym/include)

#######################
### src and linking ###
#######################
set(ENV_DIR ${CMAKE_SOURCE_DIR}/dreamer/raisim_gym)

######## Env ##########
pybind11_add_module(${PROJECT_NAME}
        dreamer/raisim_gym/src/raisim_gym.cpp
        dreamer/raisim_gym/src/Yaml.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE raisim::raisim)
target_include_directories(${PROJECT_NAME} PUBLIC ${ENV_DIR})
target_compile_options(${PROJECT_NAME} PRIVATE -fPIC -O3 -flto -mno-avx2)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DRAISIMGYM_TORCH_ENV_NAME=${PROJECT_NAME}")
target_compile_definitions(${PROJECT_NAME} PRIVATE "$<$<CONFIG:Release>:EIGEN_NO_DEBUG>")
target_compile_definitions(${PROJECT_NAME} PRIVATE EIGEN_DONT_PARALLELIZE)
