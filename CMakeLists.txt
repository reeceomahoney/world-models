cmake_minimum_required(VERSION 3.10)

project(world_model)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_PREFIX_PATH $ENV{LOCAL_INSTALL})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/env/gym_envs/${PROJECT_NAME}/build)


######################
### help functions ###
######################
MACRO(SUBDIRLIST result curdir)
    FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
    SET(dirlist "")
    FOREACH(child ${children})
        IF(IS_DIRECTORY ${curdir}/${child})
            LIST(APPEND dirlist ${child})
        ENDIF()
    ENDFOREACH()
    SET(${result} ${dirlist})
ENDMACRO()

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
        env/dependencies/actuation_dynamics_inference/include
        env/dependencies/networks_minimal/include
        env/raisim_gym_torch/env)

#######################
### src and linking ###
#######################
set(RAISIMGYM_ENV_DIR ${CMAKE_SOURCE_DIR}/env/gym_envs)
#SUBDIRLIST(SUBDIRS ${RAISIMGYM_ENV_DIR})
set(SUBDIRS ${PROJECT_NAME})

######## Env ##########
FOREACH(subdir ${SUBDIRS})
    pybind11_add_module(${subdir} env/raisim_gym_torch/env/raisim_gym.cpp env/raisim_gym_torch/env/Yaml.cpp)
    target_link_libraries(${subdir} PRIVATE raisim::raisim)
    target_include_directories(${subdir} PUBLIC 
        ${RAISIMGYM_ENV_DIR}/${subdir})
    target_compile_options(${subdir} PRIVATE -fPIC -O3 -flto -mno-avx2)
    target_compile_definitions(${subdir} PRIVATE "-DRAISIMGYM_TORCH_ENV_NAME=${subdir}")
    target_compile_definitions(${subdir} PRIVATE "$<$<CONFIG:Release>:EIGEN_NO_DEBUG>")
    target_compile_definitions(${subdir} PRIVATE EIGEN_DONT_PARALLELIZE)

    if (CMAKE_BUILD_TYPE STREQUAL "DEBUG" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("[RAISIM_GYM] BUILDING THE DEBUG APP for ${subdir}")
        add_executable(${subdir}_debug_app env/raisim_gym_torch/env/debug_app.cpp env/raisim_gym_torch/env/Yaml.cpp)
        target_link_libraries(${subdir}_debug_app PRIVATE raisim::raisim)
        target_include_directories(${subdir}_debug_app PUBLIC env/gym_envs/${subdir} ${EIGEN3_INCLUDE_DIRS} env/raisim_gym_torch/env)
    endif()
ENDFOREACH()
