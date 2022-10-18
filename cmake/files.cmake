list(APPEND    THINKER_EXECUTOR_SOURCES_DIRS  "${PROJECT_SOURCE_DIR}/thinker/executor/c_api"
                                              "${PROJECT_SOURCE_DIR}/thinker/executor/core"
                                              "${PROJECT_SOURCE_DIR}/thinker/executor/core/comm"
                                              "${PROJECT_SOURCE_DIR}/thinker/executor/core/ops")

if(THINKER_USE_VENUS)
  list(APPEND THINKER_EXECUTOR_SOURCES_DIRS "${PROJECT_SOURCE_DIR}/thinker/executor/core/ops/venus")
endif()
