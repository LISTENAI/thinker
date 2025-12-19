#include "operator_register.h"
#include <stdlib.h>
#include <string.h>

// Static array to store registered operators
static tOperatorAPI op_list[128] = {{0}};
// Counter for number of registered operators
static int32_t op_count = 0;

/**
 * Register a new operator API.
 * @param api: The operator API structure to register.
 * @return: Operator ID on success, -1 if already exists.
 */
int32_t RegistryOperatorAPI(tOperatorAPI api) {
    // Check if operator with same name already exists
    for (int32_t i = 0; i < op_count; ++i) {
        if (strcmp(api.name(), op_list[i].name()) == 0) {
            return -1;
        }
    }

    // Add the new operator to the list
    int32_t op_id = op_count;
    op_list[op_id] = api;
    ++op_count;
    return op_id;
}

/**
 * Retrieve an operator API by its name.
 * @param op_name: Name of the operator to retrieve.
 * @return: Pointer to the operator API or NULL if not found.
 */
tOperatorAPI *GetOperatorAPI(const char *op_name) {
    for (int32_t i = 0; i < op_count; ++i) {
        if (strcmp(op_name, op_list[i].name()) == 0) {
            return &(op_list[i]);
        }
    }
    return NULL;
}

/**
 * Get total count of registered operators.
 * @return: Number of registered operators.
 */
int32_t GetOperatorCount() { return op_count; }