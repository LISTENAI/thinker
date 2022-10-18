/** @file */

// thinker_status.h - THINKER API interface and data structures visible to user
// code.

// Copyright (c) 2021-2022 LISTENAI, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#ifndef _THINKER_STATUS_H_
#define _THINKER_STATUS_H_

typedef enum _thinker_StatusCode_ {
  T_SUCCESS = 0,  // run success
  T_ERR_FAIL = -1,

  T_ERR_INVALID_PLATFROM = 10002,

  T_ERR_RES_MISSING = 20000,
  T_ERR_RES_INCOMPLETE = 20001,
  T_ERR_RES_CRC_CHECK = 20002,

  T_ERR_INVALID_PARA = 30000,
  T_ERR_INVALID_INST = 30001,
  T_ERR_INVALID_DATA = 30002,

  T_ERR_NO_IMPLEMENTED = 40000,
  T_ERR_INDEX_OF_BOUND = 40001,
  T_ERR_INVALID_DATATYPE = 40002,

  T_ERR_NO_SUPPORT_OP = 50000,
} tStatus;

#endif  // _THINKER_STATUS_H_
