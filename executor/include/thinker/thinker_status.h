/** @file */
#ifndef _THINKER_STATUS_H_
#define _THINKER_STATUS_H_

/**
 * Enumeration of status codes returned by THINKER API functions.
 */
typedef enum _thinker_StatusCode_ {
    T_SUCCESS = 0,              // Operation completed successfully
    T_ERR_FAIL = -1,            // General failure
    
    T_ERR_INVALID_PLATFROM = 10002, // Invalid platform error
    
    T_ERR_RES_MISSING = 20000,   // Resource missing error
    T_ERR_RES_INCOMPLETE = 20001, // Resource incomplete error
    T_ERR_RES_CRC_CHECK = 20002,  // Resource CRC check failed
    
    T_ERR_INVALID_PARA = 30000,   // Invalid parameter error
    T_ERR_INVALID_INST = 30001,   // Invalid instruction error
    T_ERR_INVALID_DATA = 30002,   // Invalid data error
    
    T_ERR_NO_IMPLEMENTED = 40000, // Not implemented error
    T_ERR_INDEX_OF_BOUND = 40001, // Index out of bounds error
    T_ERR_INVALID_DATATYPE = 40002, // Invalid data type error
    
    T_ERR_NO_WORKSPACE  = 4003,   // No workspace available error
    
    T_ERR_NO_SUPPORT_OP = 50000,  // Unsupported operation error
} tStatus;

#endif  // _THINKER_STATUS_H_