
if (DEFINED OP_LIST)
  message(STATUS "Reading ops from ${OP_LIST}...")
  file(STRINGS ${OP_LIST} ENABLE_OP_LIST)
  foreach (ENABLE_OP ${ENABLE_OP_LIST}) 
    message("Enable Op ${ENABLE_OP}")
    add_definitions("-D__${ENABLE_OP}__")
  endforeach()
else()
  message("Enable All Op")
endif()