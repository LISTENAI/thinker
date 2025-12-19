CMAKE_MINIMUM_REQUIRED( VERSION 2.8 FATAL_ERROR )

if (POLICY CMP0023)
	cmake_policy(SET CMP0023 NEW)
endif()

set(TOTAL_SRC_LIST "")
set(TOTAL_LINK_LIBS "")
set(TOTAL_LINK_DIRS "")

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	if (CMAKE_SIZEOF_VOID_P EQUAL 8)
			set (in_lib_dir "win64")
    elseif (CMAKE_SIZEOF_VOID_P EQUAL 4)
			set (in_lib_dir "win32")
    endif ()
else()
	if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		set(DYN_SYMBOL "${CMAKE_CURRENT_SOURCE_DIR}/include/thinker/libthinker.dynsym")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--version-script=${DYN_SYMBOL}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--version-script=${DYN_SYMBOL}")
		
		if (CMAKE_SIZEOF_VOID_P EQUAL 8)
			set (in_lib_dir "linux64")
		elseif (CMAKE_SIZEOF_VOID_P EQUAL 4)
			set (in_lib_dir "linux32")
		endif ()
		
	endif()

endif(CMAKE_SYSTEM_NAME MATCHES "Windows")

set (TOTAL_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
						${CMAKE_CURRENT_SOURCE_DIR}/include
						${CMAKE_CURRENT_SOURCE_DIR}/include/thinker
						${CMAKE_CURRENT_SOURCE_DIR}/core
						${CMAKE_CURRENT_SOURCE_DIR}/core/comm
						if(THINKER_USE_VENUS)
							${CMAKE_CURRENT_SOURCE_DIR}/core/ops/venus/luna
							${CMAKE_CURRENT_SOURCE_DIR}/core/ops/venus/hifi
						elseif(THINKER_USE_ARCS)
							${CMAKE_CURRENT_SOURCE_DIR}/core/ops/arcs/luna
						elseif(THINKER_USE_VENUSA)
							${CMAKE_CURRENT_SOURCE_DIR}/core/ops/venusa/luna
						endif()
                        ${CMAKE_CURRENT_SOURCE_DIR}/c_api
                        )

aux_source_directory(c_api SRC_LIST)
set(TOTAL_SRC_LIST ${TOTAL_SRC_LIST} ${SRC_LIST})
file(GLOB SRC_LIST "core/*.c*")
set(TOTAL_SRC_LIST ${TOTAL_SRC_LIST} ${SRC_LIST})
file(GLOB SRC_LIST "core/ops/*.c*")
set(TOTAL_SRC_LIST ${TOTAL_SRC_LIST} ${SRC_LIST})
file(GLOB SRC_LIST "core/comm/*.c*")
set(TOTAL_SRC_LIST ${TOTAL_SRC_LIST} ${SRC_LIST})
file(GLOB SRC_LIST "core/shape_infer/*.c*")
set(TOTAL_SRC_LIST ${TOTAL_SRC_LIST} ${SRC_LIST})

if(THINKER_USE_VENUS)
	file(GLOB SRC_LIST "core/ops/venus/*.c")
	set(TOTAL_SRC_LIST ${TOTAL_SRC_LIST} ${SRC_LIST})
	set(TOTAL_LINK_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/libs/venus/linux64")
	if(THINKER_USE_NNBLAS)
		set(TOTAL_LINK_LIBS ${TOTAL_LINK_LIBS} "nnblas")
	else()
		set(TOTAL_LINK_LIBS ${TOTAL_LINK_LIBS} "luna")
	endif()
	set(TOTAL_LINK_LIBS ${TOTAL_LINK_LIBS} "hifi")
elseif(THINKER_USE_ARCS)
	file(GLOB SRC_LIST "core/ops/arcs/*.c")
	set(TOTAL_SRC_LIST ${TOTAL_SRC_LIST} ${SRC_LIST})
	set(TOTAL_LINK_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/libs/arcs/linux64")
	if(THINKER_USE_NNBLAS)
		set(TOTAL_LINK_LIBS ${TOTAL_LINK_LIBS} "nnblas")
	else()
		set(TOTAL_LINK_LIBS ${TOTAL_LINK_LIBS} "luna")
		set(TOTAL_LINK_LIBS ${TOTAL_LINK_LIBS} "opi_cpy")
	endif()
elseif(THINKER_USE_VENUSA)
	file(GLOB SRC_LIST "core/ops/venusa/*.c")
	set(TOTAL_SRC_LIST ${TOTAL_SRC_LIST} ${SRC_LIST})
	set(TOTAL_LINK_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/libs/venusa/linux64")
	if(THINKER_USE_NNBLAS)
		set(TOTAL_LINK_LIBS ${TOTAL_LINK_LIBS} "nnblas")
	else()
		set(TOTAL_LINK_LIBS ${TOTAL_LINK_LIBS} "luna")
	endif()
endif()
LINK_DIRECTORIES(${TOTAL_LINK_DIRS})

if(THINKER_SHARED_LIB)
    add_library (thinker SHARED ${TOTAL_SRC_LIST})
else()
    add_library (thinker STATIC ${TOTAL_SRC_LIST})
endif()


TARGET_INCLUDE_DIRECTORIES(thinker PRIVATE  ${TOTAL_INCLUDE_DIRS})

SET_TARGET_PROPERTIES(thinker PROPERTIES LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/bin)
SET_TARGET_PROPERTIES(thinker PROPERTIES ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# copy to root dir bin 
add_custom_command(TARGET thinker POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
				   ${CMAKE_CURRENT_SOURCE_DIR}/bin/  ${PROJECT_SOURCE_DIR}/bin)

message(cmake source dir:${CMAKE_CURRENT_SOURCE_DIR})
message(cmake source dir:${PROJECT_SOURCE_DIR}/)
find_package(Threads)
if (Threads_FOUND)
	TARGET_LINK_LIBRARIES (${PROJECT_NAME} PRIVATE ${TOTAL_LINK_LIBS} ${CMAKE_THREAD_LIBS_INIT} -Wno-dev )
else()
	TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE ${TOTAL_LINK_LIBS} -Wno-dev)
	message(WARNING "Threads is not found in compile")
endif()
