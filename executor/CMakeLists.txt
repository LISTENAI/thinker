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

set(TOTAL_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/thinker
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/core/comm
    ${CMAKE_CURRENT_SOURCE_DIR}/c_api)

if(THINKER_USE_MOSS)
	list(APPEND TOTAL_INCLUDE_DIRS
	    ${CMAKE_CURRENT_SOURCE_DIR}/include/moss
	    ${CMAKE_CURRENT_SOURCE_DIR}/moss_runtime/include
	    ${CMAKE_CURRENT_SOURCE_DIR}/moss_runtime/lib
	    ${PROJECT_SOURCE_DIR}/moss_runtime/include)
endif()

if(THINKER_USE_VENUS)
	list(APPEND TOTAL_INCLUDE_DIRS
	    ${CMAKE_CURRENT_SOURCE_DIR}/core/ops/venus/luna
	    ${CMAKE_CURRENT_SOURCE_DIR}/core/ops/venus/hifi)
elseif(THINKER_USE_ARCS)
	list(APPEND TOTAL_INCLUDE_DIRS
	    ${CMAKE_CURRENT_SOURCE_DIR}/core/ops/arcs/luna)
elseif(THINKER_USE_VENUSA)
	list(APPEND TOTAL_INCLUDE_DIRS
	    ${CMAKE_CURRENT_SOURCE_DIR}/core/ops/venusA/luna)
endif()

aux_source_directory(c_api SRC_LIST)
if(THINKER_USE_MOSS)
	list(REMOVE_ITEM SRC_LIST
	    c_api/thinker_api.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/c_api/thinker_api.c)
else()
	list(REMOVE_ITEM SRC_LIST
	    c_api/thinker_api_moss.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/c_api/thinker_api_moss.c)
endif()
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
	file(GLOB SRC_LIST "core/ops/venusA/*.c")
	set(TOTAL_SRC_LIST ${TOTAL_SRC_LIST} ${SRC_LIST})
	set(TOTAL_LINK_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/libs/venusa/linux64")
	if(THINKER_USE_NNBLAS)
		set(TOTAL_LINK_LIBS ${TOTAL_LINK_LIBS} "nnblas")
	else()
		set(TOTAL_LINK_LIBS ${TOTAL_LINK_LIBS} "luna")
	endif()
endif()

if(THINKER_USE_MOSS)
	set(THINKER_MOSS_RUNTIME_LIB "${TOTAL_LINK_DIRS}/libmossruntime.so")
	set(THINKER_MOSS_NNBLAS_LIB "${TOTAL_LINK_DIRS}/libnnblaslinux.a")
	foreach(THINKER_MOSS_LIB ${THINKER_MOSS_RUNTIME_LIB} ${THINKER_MOSS_NNBLAS_LIB})
		if(NOT EXISTS "${THINKER_MOSS_LIB}")
			message(FATAL_ERROR "Missing MOSS dependency: ${THINKER_MOSS_LIB}")
		endif()
	endforeach()
	list(APPEND TOTAL_LINK_LIBS
	    "${THINKER_MOSS_RUNTIME_LIB}"
	    "${THINKER_MOSS_NNBLAS_LIB}")
endif()
link_directories(${TOTAL_LINK_DIRS})

if(THINKER_SHARED_LIB)
    add_library (thinker SHARED ${TOTAL_SRC_LIST})
else()
    add_library (thinker STATIC ${TOTAL_SRC_LIST})
endif()


target_include_directories(thinker PRIVATE ${TOTAL_INCLUDE_DIRS})

if(THINKER_USE_MOSS AND CMAKE_C_COMPILER_ID STREQUAL "GNU")
	set_property(TARGET thinker APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--allow-multiple-definition")
endif()

if(THINKER_USE_MOSS)
	set(THINKER_MOSS_HOST_SOURCE "" CACHE FILEPATH "Generated MOSS host.c compiled into thinker")
	set(THINKER_MOSS_HOST_HEADER "" CACHE FILEPATH "Optional generated MOSS host header; its directory is added to include paths")
	set(THINKER_MOSS_HOST_INCLUDE_DIRS "" CACHE STRING "Extra include directories for generated MOSS host source")
	set(THINKER_MOSS_MODEL_GETTER "" CACHE STRING "Generated mGetModel_* symbol stored in the MOSS resource")
	set(THINKER_MOSS_MODEL_NAME "" CACHE STRING "Logical MOSS model name stored in the resource")
	set(THINKER_MOSS_HOST_SOURCES "" CACHE STRING "Semicolon-separated generated MOSS host.c sources compiled into thinker")
	set(THINKER_MOSS_MODEL_GETTERS "" CACHE STRING "Semicolon-separated generated mGetModel_* symbols stored in MOSS resources")
	set(THINKER_MOSS_MODEL_NAMES "" CACHE STRING "Semicolon-separated logical model names stored in MOSS resources")

	if(NOT THINKER_MOSS_HOST_SOURCE AND NOT THINKER_MOSS_HOST_SOURCES AND THINKER_USE_VENUSA AND EXISTS "${PROJECT_SOURCE_DIR}/demo/test_moss/VenusAint8_HandDetect_host.c")
		set(THINKER_MOSS_HOST_SOURCE "${PROJECT_SOURCE_DIR}/demo/test_moss/VenusAint8_HandDetect_host.c")
		set(THINKER_MOSS_MODEL_GETTER "mGetModel_VenusAint8_HandDetect")
		set(THINKER_MOSS_MODEL_NAME "VenusAint8_HandDetect")
	endif()

	set(_THINKER_MOSS_HOST_SOURCES ${THINKER_MOSS_HOST_SOURCES})
	set(_THINKER_MOSS_MODEL_GETTERS ${THINKER_MOSS_MODEL_GETTERS})
	set(_THINKER_MOSS_MODEL_NAMES ${THINKER_MOSS_MODEL_NAMES})
	if(THINKER_MOSS_HOST_SOURCE)
		list(APPEND _THINKER_MOSS_HOST_SOURCES "${THINKER_MOSS_HOST_SOURCE}")
		list(APPEND _THINKER_MOSS_MODEL_GETTERS "${THINKER_MOSS_MODEL_GETTER}")
		if(THINKER_MOSS_MODEL_NAME)
			list(APPEND _THINKER_MOSS_MODEL_NAMES "${THINKER_MOSS_MODEL_NAME}")
		endif()
	endif()

	if(_THINKER_MOSS_HOST_SOURCES)
		list(LENGTH _THINKER_MOSS_HOST_SOURCES _THINKER_MOSS_HOST_COUNT)
		list(LENGTH _THINKER_MOSS_MODEL_GETTERS _THINKER_MOSS_GETTER_COUNT)
		list(LENGTH _THINKER_MOSS_MODEL_NAMES _THINKER_MOSS_NAME_COUNT)
		if(NOT _THINKER_MOSS_GETTER_COUNT EQUAL _THINKER_MOSS_HOST_COUNT)
			message(FATAL_ERROR "THINKER_MOSS_MODEL_GETTERS must have one getter per MOSS host source")
		endif()
		if(_THINKER_MOSS_NAME_COUNT GREATER 0 AND NOT _THINKER_MOSS_NAME_COUNT EQUAL _THINKER_MOSS_HOST_COUNT)
			message(FATAL_ERROR "THINKER_MOSS_MODEL_NAMES must be empty or have one name per MOSS host source")
		endif()

		set(_THINKER_MOSS_ABS_HOST_SOURCES "")
		set(_THINKER_MOSS_HOST_INCLUDE_DIRS "")
		set(_THINKER_MOSS_REGISTRY_TEXT "#include <string.h>\n\n")
		math(EXPR _THINKER_MOSS_LAST_INDEX "${_THINKER_MOSS_HOST_COUNT} - 1")
		foreach(_THINKER_MOSS_INDEX RANGE 0 ${_THINKER_MOSS_LAST_INDEX})
			list(GET _THINKER_MOSS_HOST_SOURCES ${_THINKER_MOSS_INDEX} _THINKER_MOSS_HOST)
			list(GET _THINKER_MOSS_MODEL_GETTERS ${_THINKER_MOSS_INDEX} _THINKER_MOSS_GETTER)
			if(_THINKER_MOSS_NAME_COUNT GREATER 0)
				list(GET _THINKER_MOSS_MODEL_NAMES ${_THINKER_MOSS_INDEX} _THINKER_MOSS_NAME)
			else()
				set(_THINKER_MOSS_NAME "${_THINKER_MOSS_GETTER}")
			endif()
			if(NOT _THINKER_MOSS_GETTER)
				message(FATAL_ERROR "empty MOSS getter at index ${_THINKER_MOSS_INDEX}")
			endif()
			if(NOT IS_ABSOLUTE "${_THINKER_MOSS_HOST}")
				set(_THINKER_MOSS_HOST "${PROJECT_SOURCE_DIR}/${_THINKER_MOSS_HOST}")
			endif()
			list(APPEND _THINKER_MOSS_ABS_HOST_SOURCES "${_THINKER_MOSS_HOST}")
			get_filename_component(_THINKER_MOSS_HOST_SOURCE_DIR "${_THINKER_MOSS_HOST}" DIRECTORY)
			list(APPEND _THINKER_MOSS_HOST_INCLUDE_DIRS "${_THINKER_MOSS_HOST_SOURCE_DIR}")
			string(APPEND _THINKER_MOSS_REGISTRY_TEXT "extern void *${_THINKER_MOSS_GETTER}(void);\n")
		endforeach()
		string(APPEND _THINKER_MOSS_REGISTRY_TEXT "\nvoid *thinker_moss_get_registered_model(const char *getter_name, const char *model_name) {\n")
		foreach(_THINKER_MOSS_INDEX RANGE 0 ${_THINKER_MOSS_LAST_INDEX})
			list(GET _THINKER_MOSS_MODEL_GETTERS ${_THINKER_MOSS_INDEX} _THINKER_MOSS_GETTER)
			if(_THINKER_MOSS_NAME_COUNT GREATER 0)
				list(GET _THINKER_MOSS_MODEL_NAMES ${_THINKER_MOSS_INDEX} _THINKER_MOSS_NAME)
			else()
				set(_THINKER_MOSS_NAME "${_THINKER_MOSS_GETTER}")
			endif()
			string(APPEND _THINKER_MOSS_REGISTRY_TEXT "  if ((getter_name != 0 && strcmp(getter_name, \"${_THINKER_MOSS_GETTER}\") == 0) ||\n")
			string(APPEND _THINKER_MOSS_REGISTRY_TEXT "      (model_name != 0 && strcmp(model_name, \"${_THINKER_MOSS_NAME}\") == 0)) {\n")
			string(APPEND _THINKER_MOSS_REGISTRY_TEXT "    return ${_THINKER_MOSS_GETTER}();\n")
			string(APPEND _THINKER_MOSS_REGISTRY_TEXT "  }\n")
		endforeach()
		string(APPEND _THINKER_MOSS_REGISTRY_TEXT "  return 0;\n}\n")

		set(THINKER_MOSS_REGISTRY_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/thinker_moss_registry.c")
		file(WRITE "${THINKER_MOSS_REGISTRY_SOURCE}" "${_THINKER_MOSS_REGISTRY_TEXT}")

		target_sources(thinker PRIVATE
			${_THINKER_MOSS_ABS_HOST_SOURCES}
			${THINKER_MOSS_REGISTRY_SOURCE})
		target_include_directories(thinker PRIVATE
			${_THINKER_MOSS_HOST_INCLUDE_DIRS}
			${THINKER_MOSS_HOST_INCLUDE_DIRS})
		if(THINKER_MOSS_HOST_HEADER)
			get_filename_component(THINKER_MOSS_HOST_HEADER_DIR "${THINKER_MOSS_HOST_HEADER}" DIRECTORY)
			target_include_directories(thinker PRIVATE ${THINKER_MOSS_HOST_HEADER_DIR})
		endif()
	endif()
endif()

set_target_properties(thinker PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set_target_properties(thinker PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# copy to root dir bin
add_custom_command(TARGET thinker POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
				   ${CMAKE_CURRENT_SOURCE_DIR}/bin/  ${PROJECT_SOURCE_DIR}/bin)

message(cmake source dir:${CMAKE_CURRENT_SOURCE_DIR})
message(cmake source dir:${PROJECT_SOURCE_DIR}/)
find_package(Threads)
if (Threads_FOUND)
	target_link_libraries (${PROJECT_NAME} PRIVATE ${TOTAL_LINK_LIBS} ${CMAKE_THREAD_LIBS_INIT} -Wno-dev )
else()
	target_link_libraries(${PROJECT_NAME} PRIVATE ${TOTAL_LINK_LIBS} -Wno-dev)
	message(WARNING "Threads is not found in compile")
endif()

if(THINKER_BUILD_MOSS_RUNTIME)
	set(MOSS_RUNTIME_SRC_LIST
	    ${CMAKE_CURRENT_SOURCE_DIR}/moss_runtime/lib/moss_runtime.c
	    ${CMAKE_CURRENT_SOURCE_DIR}/moss_runtime/lib/moss_model_compat.c)
	add_library(mossruntime STATIC ${MOSS_RUNTIME_SRC_LIST})
	target_include_directories(mossruntime PUBLIC
	    ${CMAKE_CURRENT_SOURCE_DIR}/moss_runtime/include
	    ${CMAKE_CURRENT_SOURCE_DIR}/moss_runtime/lib)
	set_target_properties(mossruntime PROPERTIES
	    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
endif()
