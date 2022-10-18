CMAKE_MINIMUM_REQUIRED( VERSION 2.8 FATAL_ERROR )

if (POLICY CMP0023)
	cmake_policy(SET CMP0023 NEW)
endif()

set(TOTAL_SRC_LIST "")
set(TOTAL_LINK_LIBS "")
set(TOTAL_LINK_DIRS "")

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
else()
	if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		set(DYN_SYMBOL "${CMAKE_CURRENT_SOURCE_DIR}/include/thinker/libthinker.dynsym")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--version-script=${DYN_SYMBOL}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--version-script=${DYN_SYMBOL}")
	endif()

endif(CMAKE_SYSTEM_NAME MATCHES "Windows")

set (TOTAL_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
						${CMAKE_CURRENT_SOURCE_DIR}/include
						${CMAKE_CURRENT_SOURCE_DIR}/include/thinker
						${CMAKE_CURRENT_SOURCE_DIR}/core
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

if(THINKER_USE_VENUS)
	file(GLOB SRC_LIST "core/ops/venus/*.c")
	set(TOTAL_SRC_LIST ${TOTAL_SRC_LIST} ${SRC_LIST})
	set(TOTAL_LINK_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/libs/venus/linux64")
	set(TOTAL_LINK_LIBS ${TOTAL_LINK_LIBS} "luna")
	set(TOTAL_LINK_LIBS ${TOTAL_LINK_LIBS} "hifi")
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

find_package(Threads)
if (Threads_FOUND)
	TARGET_LINK_LIBRARIES (${PROJECT_NAME} PRIVATE ${TOTAL_LINK_LIBS} ${CMAKE_THREAD_LIBS_INIT} -Wno-dev )
else()
	TARGET_LINK_LIBRARIES(thinker PRIVATE ${TOTAL_LINK_LIBS} -Wno-dev)
	message(WARNING "Threads is not found in compile")
endif()
