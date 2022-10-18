#include "core/operator_register.h"
#include "thinker_log.h"

// all
#define DATA_TYPE_SWITCH_ALL(type, Type, ...)         \
  switch (type) {                                     \
    case Float16: {                                   \
      THINKER_LOG_FATAL("do not support Float16!");   \
    } break;                                          \
    case Float32: {                                   \
      typedef float Type;                             \
      { __VA_ARGS__ }                                 \
    } break;                                          \
    case Int8: {                                      \
      typedef int8_t Type;                            \
      { __VA_ARGS__ }                                 \
    } break;                                          \
    case Int16: {                                     \
      typedef int16_t Type;                           \
      { __VA_ARGS__ }                                 \
    } break;                                          \
    case Int32: {                                     \
      typedef int32_t Type;                           \
      { __VA_ARGS__ }                                 \
    } break;                                          \
    case Int64: {                                     \
      typedef int64_t Type;                           \
      { __VA_ARGS__ }                                 \
    } break;                                          \
    case Uint8: {                                     \
      typedef uint8_t Type;                           \
      { __VA_ARGS__ }                                 \
    } break;                                          \
    case Uint16: {                                    \
      typedef uint16_t Type;                          \
      { __VA_ARGS__ }                                 \
    } break;                                          \
    case Uint32: {                                    \
      typedef uint32_t Type;                          \
      { __VA_ARGS__ }                                 \
    } break;                                          \
    case Uint64: {                                    \
      typedef uint64_t Type;                          \
      { __VA_ARGS__ }                                 \
    } break;                                          \
    default:                                          \
      THINKER_LOG_FATAL("do not support this type!"); \
      break;                                          \
  }

// uint32_t
#define DATA_TYPE_SWITCH_UNSIGNEDINT(type, Type, ...)   \
  switch (type) {                                       \
    case Uint8: {                                       \
      typedef uint8_t Type;                             \
      { __VA_ARGS__ }                                   \
    } break;                                            \
    case Uint16: {                                      \
      typedef uint16_t Type;                            \
      { __VA_ARGS__ }                                   \
    } break;                                            \
    case Uint32: {                                      \
      typedef uint32_t Type;                            \
      { __VA_ARGS__ }                                   \
    } break;                                            \
    case Uint64: {                                      \
      typedef uint64_t Type;                            \
      { __VA_ARGS__ }                                   \
    } break;                                            \
    default:                                            \
      THINKER_LOG_FATAL("only support uint32_t type!"); \
      break;                                            \
  }

// signed int32_t, float
#define DATA_TYPE_SWITCH_SIGNED(type, Type, ...)      \
  switch (type) {                                     \
    case Int8: {                                      \
      typedef int8_t Type;                            \
      { __VA_ARGS__ }                                 \
    } break;                                          \
    case Int16: {                                     \
      typedef int16_t Type;                           \
      { __VA_ARGS__ }                                 \
    } break;                                          \
    case Int32: {                                     \
      typedef int32_t Type;                           \
      { __VA_ARGS__ }                                 \
    } break;                                          \
    case Int64: {                                     \
      typedef int64_t Type;                           \
      { __VA_ARGS__ }                                 \
    } break;                                          \
    case Float32: {                                   \
      typedef float Type;                             \
      { __VA_ARGS__ }                                 \
    } break;                                          \
    default:                                          \
      THINKER_LOG_FATAL("only support signed type!"); \
      break;                                          \
  }

// 用于cumsum
#define DATA_TYPE_SWITCH_BIGTYPE(type, Type, ...) \
  switch (type) {                                 \
    case Float32: {                               \
      typedef float Type;                         \
      { __VA_ARGS__ }                             \
    } break;                                      \
    case Uint32: {                                \
      typedef uint32_t Type;                      \
      { __VA_ARGS__ }                             \
    } break;                                      \
    case Int32: {                                 \
      typedef int32_t Type;                       \
      { __VA_ARGS__ }                             \
    } break;                                      \
    case Uint64: {                                \
      typedef uint64_t Type;                      \
      { __VA_ARGS__ }                             \
    } break;                                      \
    case Int64: {                                 \
      typedef int64_t Type;                       \
      { __VA_ARGS__ }                             \
    } break;                                      \
    default:                                      \
      THINKER_LOG_FATAL(" do not support !");     \
      break;                                      \
  }
