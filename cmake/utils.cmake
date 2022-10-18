MACRO (GET_CPP_SOURCE SRC_DIRS CPP_SOURCE)
	message("=dir=" ${SRC_DIRS})
	FOREACH(SRC_DIR ${SRC_DIRS})
		FILE(GLOB ${CPP_SOURCE}_TMP "${SRC_DIR}/*.c" "${SRC_DIR}/*.cc" "${SRC_DIR}/*.cpp")
		LIST (APPEND ${CPP_SOURCE} ${${CPP_SOURCE}_TMP})
	ENDFOREACH()
ENDMACRO()

MACRO (GET_C_SOURCE SRC_DIRS C_SOURCE)
	message("=dir=" ${SRC_DIRS})
	FOREACH(SRC_DIR ${SRC_DIRS})
		FILE(GLOB_RECURSE ${C_SOURCE}_TMP "${SRC_DIR}/*.c")
		SET (${C_SOURCE} ${${C_SOURCE}} ${${C_SOURCE}_TMP})
	ENDFOREACH()
ENDMACRO()

MACRO (ADD_STATIC_LIBRARY TRAGET_NAME SRC_DIRS DST_DIR )
	GET_CPP_SOURCE("${SRC_DIRS}" BUILD_${TRAGET_NAME}_CPP_SOURCE)
	ADD_LIBRARY( ${TRAGET_NAME} STATIC ${BUILD_${TRAGET_NAME}_CPP_SOURCE})
	SET_TARGET_PROPERTIES( ${TRAGET_NAME} PROPERTIES 
					      	ARCHIVE_OUTPUT_DIRECTORY ${DST_DIR})
ENDMACRO ( ADD_STATIC_LIBRARY )

MACRO (ADD_C_STATIC_LIBRARY TRAGET_NAME SRC_DIRS DST_DIR )
	GET_C_SOURCE(${SRC_DIRS} BUILD_${TRAGET_NAME}_C_SOURCE)
	ADD_LIBRARY( ${TRAGET_NAME} STATIC ${BUILD_${TRAGET_NAME}_C_SOURCE})
	SET_TARGET_PROPERTIES( ${TRAGET_NAME} PROPERTIES 
					      	ARCHIVE_OUTPUT_DIRECTORY ${DST_DIR})
ENDMACRO ( ADD_C_STATIC_LIBRARY )

function (ADD_SHARED_LIBRARY TRAGET_NAME SRC_DIRS DST_DIR)
	GET_CPP_SOURCE("${SRC_DIRS}" BUILD_${TRAGET_NAME}_CPP_SOURCE)
	ADD_LIBRARY( ${TRAGET_NAME} SHARED ${BUILD_${TRAGET_NAME}_CPP_SOURCE})
	TARGET_LINK_LIBRARIES( ${TRAGET_NAME} ${ARGN} )
	SET_TARGET_PROPERTIES( ${TRAGET_NAME} PROPERTIES 
					      	LIBRARY_OUTPUT_DIRECTORY ${DST_DIR})
endfunction ( ADD_SHARED_LIBRARY )

MACRO (ADD_C_SHARED_LIBRARY TRAGET_NAME SRC_DIRS DST_DIR)
	GET_C_SOURCE(${SRC_DIRS} BUILD_${TRAGET_NAME}_C_SOURCE)
	ADD_LIBRARY( ${TRAGET_NAME} SHARED ${BUILD_${TRAGET_NAME}_C_SOURCE})
	TARGET_LINK_LIBRARIES( ${TRAGET_NAME} ${ARGN} )
	SET_TARGET_PROPERTIES( ${TRAGET_NAME} PROPERTIES 
					      	LIBRARY_OUTPUT_DIRECTORY ${DST_DIR})
ENDMACRO ( ADD_C_SHARED_LIBRARY )

MACRO (ADD_EXECTION TRAGET_NAME SRC_DIRS DST_DIR)
	GET_CPP_SOURCE("${SRC_DIRS}" BUILD_${TRAGET_NAME}_CPP_SOURCE)
	ADD_EXECUTABLE( ${TRAGET_NAME} ${BUILD_${TRAGET_NAME}_CPP_SOURCE})
	TARGET_LINK_LIBRARIES( ${TRAGET_NAME} ${ARGN} )
	SET_TARGET_PROPERTIES( ${TRAGET_NAME} PROPERTIES 
					      	RUNTIME_OUTPUT_DIRECTORY ${DST_DIR})
ENDMACRO ( ADD_EXECTION )

MACRO (ADD_C_EXECTION TRAGET_NAME SRC_DIRS DST_DIR)
	GET_C_SOURCE(${SRC_DIRS} BUILD_${TRAGET_NAME}_C_SOURCE)
	ADD_EXECUTABLE( ${TRAGET_NAME} ${BUILD_${TRAGET_NAME}_C_SOURCE})
	# ADD_EXECUTABLE( ${TRAGET_NAME} EXCLUDE_FROM_ALL ${BUILD_${TRAGET_NAME}_C_SOURCE})
	TARGET_LINK_LIBRARIES( ${TRAGET_NAME} ${ARGN} )
	SET_TARGET_PROPERTIES( ${TRAGET_NAME} PROPERTIES 
					      	RUNTIME_OUTPUT_DIRECTORY ${DST_DIR})
ENDMACRO ( ADD_C_EXECTION )

MACRO (GROUP_SOURCE_TREE SOURCE_FILES SOURCE_ROOT)
  IF (MSVC)
	GET_FILENAME_COMPONENT (GST_ROOT ${SOURCE_ROOT} ABSOLUTE )
    FOREACH (GST_FILE IN LISTS ${${SOURCE_FILES}})
	  GET_FILENAME_COMPONENT (GST_FILE ${GST_FILE} ABSOLUTE )
      STRING (REGEX REPLACE "${GST_ROOT}/(.*)" \\1 SGT_FPATH ${SOURCE_FILE})
      STRING (REGEX REPLACE "(.*)/.*" \\1 GROUP_NAME ${SGT_FPATH})
      STRING (COMPARE EQUAL ${SGT_FPATH} ${GROUP_NAME} NOGROUP)
      STRING (REPLACE "/" "\\" GROUP_NAME ${GROUP_NAME})
      IF (NOGROUP)
          set(GROUP_NAME "\\")
      ENDIF (NOGROUP)
      SOURCE_GROUP(${GROUP_NAME} FILES ${GST_FILE})
    ENDFOREACH (GST_FILE)
  ENDIF(MSVC)
ENDMACRO (GROUP_SOURCE_TREE)

