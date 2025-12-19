list(APPEND    THINKER_EXECUTOR_SOURCES_DIRS  "${PROJECT_SOURCE_DIR}/executor/c_api"
                                              "${PROJECT_SOURCE_DIR}/executor/core"
                                              "${PROJECT_SOURCE_DIR}/executor/core/comm"
                                              "${PROJECT_SOURCE_DIR}/executor/core/ops")

if(THINKER_USE_VENUS)
  list(APPEND THINKER_EXECUTOR_SOURCES_DIRS "${PROJECT_SOURCE_DIR}/executor/core/ops/venus")
endif()

if(THINKER_USE_ARCS)
  list(APPEND THINKER_EXECUTOR_SOURCES_DIRS "${PROJECT_SOURCE_DIR}/executor/core/ops/arcs")
endif()
