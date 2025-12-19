cmake_minimum_required( VERSION 3.5 FATAL_ERROR)

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/executor/c_api/thinker_define.h" THINKER_DEFINE)
string(REGEX MATCH "THINKER_VERSION_MAJOR [0-9]+" THINKER_VERSION_MAJOR_DEFINE ${THINKER_DEFINE})
string(REGEX MATCH "[0-9]+" THINKER_VERSION_MAJOR ${THINKER_VERSION_MAJOR_DEFINE})
string(REGEX MATCH "THINKER_VERSION_MINOR [0-9]+" THINKER_VERSION_MINOR_DEFINE ${THINKER_DEFINE})
string(REGEX MATCH "[0-9]+" THINKER_VERSION_MINOR ${THINKER_VERSION_MINOR_DEFINE})
string(REGEX MATCH "THINKER_VERSION_PATCH [0-9]+" THINKER_VERSION_PATCH_DEFINE ${THINKER_DEFINE})
string(REGEX MATCH "[0-9]+" THINKER_VERSION_PATCH ${THINKER_VERSION_PATCH_DEFINE})
set(THINKER_VERSION ${THINKER_VERSION_MAJOR}.${THINKER_VERSION_MINOR}.${THINKER_VERSION_PATCH})

if (POLICY CMP0048)
  CMAKE_POLICY(SET CMP0048 NEW)
endif()

if (POLICY CMP0053)
  CMAKE_POLICY(SET CMP0053 NEW)
endif()

project(thinker VERSION ${THINKER_VERSION} LANGUAGES C CXX ASM)

set(CMAKE_C_STANDARD 99)
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(CMAKE_USE_WIN32_THREADS_INIT 0)
	set(CMAKE_USE_PTHREADS_INIT 1)
elseif (MSVC)
	set(CMAKE_USE_WIN32_THREADS_INIT 1)
	set(CMAKE_USE_PTHREADS_INIT 0)
endif ()
set(THREADS_PREFER_PTHREAD_FLAG ON)

option(THINKER_SHARED_LIB         "share library support(dynamic library)."           ON)
option(THINKER_USE_OPENMP         "thinker use openmp."                               OFF)
option(THINKER_PROFILE            "Porfiling each layer's performance."               OFF)
option(THINKER_RESULT_DUMP        "If dump middle layeres' output."                   OFF)
option(DTHINKER_RESULT_CRC_PRINT  "If print crc32_calc of middle layeres' output."    OFF)
option(DTHINKER_RESOUCR_CRC_CHECK "check crc of model resource."                      ON)
option(THINKER_USE_VENUS          "Using simulator of venus.  "                       ON)
option(THINKER_USE_ARCS           "Using simulator of arcs.  "                        OFF)
option(THINKER_USE_VENUSA         "Using simulator of venusa.  "                      OFF)
option(THINKER_AUTO_TEST          "thinker auto test"                                 OFF)
option(THINKER_DYNAMIC            "support dynamic shape"                             OFF)
option(THINKER_CHECK_PLATFORM     "check resources compatible with the target platform" ON)

include( "./cmake/config.cmake" )

IF(THINKER_USE_VENUS)
  ADD_DEFINITIONS(-DTHINKER_USE_VENUS=1)
endif()

IF(THINKER_USE_ARCS)
  ADD_DEFINITIONS(-DTHINKER_USE_ARCS=1)
endif()

IF(THINKER_USE_VENUSA)
  ADD_DEFINITIONS(-DTHINKER_USE_VENUSA=1)
endif()

if(THINKER_USE_OPENMP)
    ADD_DEFINITIONS(-DTHINKER_USE_OPENMP=1)
endif()

if(THINKER_PROFILE)
  ADD_DEFINITIONS(-DTHINKER_PROFILE=1)
endif()

if(THINKER_RESULT_DUMP)
  ADD_DEFINITIONS(-DTHINKER_DUMP=1)
endif()

if(THINKER_RESULT_CRC_PRINT)
  ADD_DEFINITIONS(-DTHINKER_RESULT_CRC_PRINT=1)
endif()

if(THINKER_RESOUCR_CRC_CHECK)
  ADD_DEFINITIONS(-DTHINKER_CHECK_CRC=1)
endif()

if(THINKER_DYNAMIC)
  ADD_DEFINITIONS(-DTHINKER_DYNAMIC=1)
endif()

if(THINKER_CHECK_PLATFORM)
  ADD_DEFINITIONS(-DTHINKER_CHECK_PLATFORM=1)
endif()

ADD_SUBDIRECTORY(executor)
ADD_SUBDIRECTORY(demo/test_thinker)
ADD_SUBDIRECTORY(demo/test_dynamic)
