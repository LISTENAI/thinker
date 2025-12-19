#ifndef __LUNA_MTQ_H_
#define __LUNA_MTQ_H_

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>

#define LUNA_MTQ_COST_CYCLE_RETURN_EN (1)  //TODO: 临时测试添加后续标准化

// common_define
#define MTQ_MAX_CHANNELS                    (4)
#define MTQ_MAX_PRIORITY_LEVEL              (4)
#define MTQ_MAX_PRIORITY_WEIGHT             (16)

#define MTQ_CHANNEL_0                       (0)
#define MTQ_CHANNEL_1                       (1)
#define MTQ_CHANNEL_2                       (2)
#define MTQ_CHANNEL_3                       (3)

#define MTQ_PRIORITY_LEVEL_0                (0)
#define MTQ_PRIORITY_LEVEL_1                (1)
#define MTQ_PRIORITY_LEVEL_2                (2)
#define MTQ_PRIORITY_LEVEL_3                (3)

#define MTQ_PRIORITY_WEIGHT_1               (1)
#define MTQ_PRIORITY_WEIGHT_2               (2)
#define MTQ_PRIORITY_WEIGHT_3               (3)
#define MTQ_PRIORITY_WEIGHT_4               (4)
#define MTQ_PRIORITY_WEIGHT_5               (5)
#define MTQ_PRIORITY_WEIGHT_6               (6)
#define MTQ_PRIORITY_WEIGHT_7               (7)
#define MTQ_PRIORITY_WEIGHT_8               (8)
#define MTQ_PRIORITY_WEIGHT_9               (9)
#define MTQ_PRIORITY_WEIGHT_10              (10)
#define MTQ_PRIORITY_WEIGHT_11              (11)
#define MTQ_PRIORITY_WEIGHT_12              (12)
#define MTQ_PRIORITY_WEIGHT_13              (13)
#define MTQ_PRIORITY_WEIGHT_14              (14)
#define MTQ_PRIORITY_WEIGHT_15              (15)

// task_type
#define MTQ_TASK_TYPE_LUNA_TASK             (0x01)
#define MTQ_TASK_TYPE_WAIT_OP               (0x02)

// mark_idx
#define MTQ_MARK_IDX_IOWR_OVER              (0x00)
#define MTQ_MARK_IDX_SLAVE0_OVER            (0x01)
#define MTQ_MARK_IDX_DMA_DONE0              (0x02)
#define MTQ_MARK_IDX_DMA_DONE1              (0x03)

// blocking_type
#define MTQ_BLOCKING_TYPE_BLOCKING_TASK     (0x00)
#define MTQ_BLOCKING_TYPE_NON_BLOCKING_TASK (0x01)

// return_cq_bypass
#define MTQ_RETURE_CQ_BYPASS_ENABLE       (0x01)
#define MTQ_RETURE_CQ_BYPASS_DISABLE      (0x00)

// op_interrupt_enable
#define MTQ_OP_INTERRUPT_ENABLE           (0x01)
#define MTQ_OP_INTERRUPT_DISABLE          (0x00)

// op_status
#define MTQ_OP_STATUS_OKEY                      (0x1)
#define MTQ_OP_STATUS_DONE_BUT_TIMEOUT          (0x2)
#define MTQ_OP_STATUS_NOT_DONE_BUT_TIMEOUT      (0x3)

// event mask
#define MTQ_OP_DONE_EVENT_MASK               (1<<24)
#define MTQ_OP_TIMEOUT_EVENT_MASK            (1<<25)
#define MTQ_WAIT_OP_TIMEOUT_EVENT_MASK       (1<<26)
#define MTQ_SQ_EMPTY_EVENT_MASK              (1<<27)
#define MTQ_SQ_ALMOST_EMPTY_EVENT_MASK       (1<<28)
#define MTQ_CQ_FULL_EVENT_MASK               (1<<29)
#define MTQ_TASK_INVALID_EVENT_MASK          (1<<30) 
#define MTQ_IN_MARK_DONE_ERROR_EVENT_MASK    (1<<31)

// event type
#define MTQ_OP_DONE_EVENT               (0x0)
#define MTQ_OP_TIMEOUT_EVENT            (0x1)
#define MTQ_WAIT_OP_TIMEOUT_EVENT       (0x2)
#define MTQ_SQ_EMPTY_EVENT              (0x3)
#define MTQ_SQ_ALMOST_EMPTY_EVENT       (0x4)
#define MTQ_CQ_FULL_EVENT               (0x5)
#define MTQ_TASK_INVALID_EVENT          (0x6)
#define MTQ_IN_MARK_DONE_ERROR_EVENT    (0x7)

#ifdef CONFIG_EXT_RAM
#define _MTQ_EXT_RAM __attribute__((section (".ramcode")))
#else
#define _MTQ_EXT_RAM
#endif

/**
 * @brief Event callback function prototype.
 *
 * This function prototype defines the signature of the event callback function
 * that can be registered with luna_mtq_register_event_handler().
 *
 * @param event_type   Type of the event (e.g., MTQ_OP_DONE_EVENT).
 * @param event_value  Additional value associated with the event.
 * @param userdata     User-defined data provided during registration.
 */
typedef void (*luna_mtq_event_handler_t)(uint32_t event_type, uint32_t event_value, void* userdata);

typedef struct luna_mqt_done_event_info_t
{
    uint32_t task_info : 8;  // bypass sq_elem[0:7]
    uint32_t op_id : 16; // bypass sq_elem op_id
    uint32_t op_status : 2; // 1-okay, 2-done but timeout, 3-not done and timeout
    uint32_t reserved : 3; 
    uint32_t arb_cur_id : 3; // chanel id
} luna_mqt_done_event_info_t;

/**
 * @brief Submit Queue (SQ) element structure.
 *
 * This structure defines the format of an element in the Submit Queue (SQ).
 * Each element contains information about a task or operation to be executed.
 */
typedef struct luna_mtq_sq_elem_t 
{
    uint32_t    task_type : 2;  // b01:luna task, b10:wait op;
    uint32_t    mark_idx : 2;  // 0-iowr_over，1-slave0_over，2-dma_done[0]，3-dma_done[1]
    uint32_t    blocking_type : 2; // b00:blocking task，b01: non-blocking task
    uint32_t    reture_cq_bypass : 1; // 1-bypass, 0-not
    uint32_t    op_interrupt_enable : 1; // 1-enable, 0-disable
    uint32_t    reserved : 8;
    uint32_t    op_id : 16; 
    union {
        uint32_t  task_base_addr; // define task base addr when task_type is luna task
        uint32_t  mark_timeout_wl; // define mark timeout when task_type is wait op
    } task_base_addr;
    uint32_t task_param; // define task param when task_type is luna task
} luna_mtq_sq_elem_t;

/**
 * @brief Completion Queue (CQ) element structure.
 *
 * This structure defines the format of an element in the Completion Queue (CQ).
 * Each element contains information about the completion of a task or operation.
 */
typedef struct luna_mtq_cq_elem_t 
{
    uint32_t    task_type : 2;  // b01-luna task, b10-wait op;
    uint32_t    mark_idx : 2;  // 0-iowr_over，1-slave0_over，2-dma_done[0]，3-dma_done[1]
    uint32_t    blocking_type : 2; // b00-blocking task，b01-non-blocking task
    uint32_t    reture_cq_bypass : 1; // b00-write return，b01-not write return
    uint32_t    op_interrupt_enable : 1;
    uint32_t    op_status : 2; //1-okay, 2-done but timeout, 3-not done and timeout
    uint32_t    reserved : 6;
    uint32_t    op_id : 16; 
#if LUNA_MTQ_COST_CYCLE_RETURN_EN
    uint32_t    op_cost_cycle;  // define op cost cycle when task_type is luna task
#endif //LUNA_MTQ_COST_CYCLE_RETURN_EN
} luna_mtq_cq_elem_t;

/**
 * @brief Configuration structure for an MTQ (Multi-Task Queue) channel.
 *
 * This structure holds the configuration parameters required to set up and initialize
 * an MTQ channel, including memory addresses and sizes for the Submit Queue (SQ)
 * and Completion Queue (CQ), priority settings, timeout threshold, and almost-empty threshold.
 *
 * @field sq_addr              Physical address of the Submit Queue (SQ) memory.
 * @field cq_addr              Physical address of the Completion Queue (CQ) memory.
 * @field sq_length            Length of the Submit Queue in elements.
 * @field cq_length            Length of the Completion Queue in elements.
 * @field priority_level       Priority level of the channel (must be < MTQ_MAX_PRIORITY_LEVEL).
 * @field priority_weight      Weight of the priority (used for weighted arbitration, must be < MTQ_MAX_PRIORITY_WEIGHT).
 * @field timeout_threshold    Timeout value for operations on this channel.
 * @field almost_empty_threshold Threshold to trigger almost-empty events for SQ.
 */
typedef struct luna_mtq_config_t
{
    uint32_t sq_addr; 
    uint32_t sq_length; 
    uint32_t cq_addr;
    uint32_t cq_length;
    uint32_t priority_level; 
    uint32_t priority_weight; 
    uint32_t timeout_threshold;
    uint32_t almost_empty_threshold; 
} luna_mtq_config_t;

typedef struct luna_mtq_status_t
{
    uint32_t sq_addr; 
    uint32_t sq_length; 
    uint32_t sq_head;
    uint32_t sq_tail;
    uint32_t sq_size;
    uint32_t cq_addr;
    uint32_t cq_length;
    uint32_t cq_head;
    uint32_t cq_tail;
    uint32_t cq_size;
    uint32_t priority_level; 
    uint32_t priority_weight; 
    uint32_t timeout_threshold;
    uint32_t almost_empty_threshold; 
} luna_mtq_status_t;

int luna_mtq_init();
void luna_mtq_deinit(void);

/**
 * @brief Configures and initializes a specific MTQ channel with given parameters.
 *
 * This function sets up the Submit Queue (SQ) and Completion Queue (CQ) for a specified MTQ channel,
 * including their memory addresses, lengths, priority settings, timeout thresholds, and empty thresholds.
 * It must be called after luna_mtq_init() and before enabling the channel via luna_mtq_enable().
 *
 * @param ch      The channel index (0-based, must be less than MTQ_MAX_CHANNELS).
 * @param config  A pointer to a luna_mtq_config_t structure containing the configuration parameters.
 *
 * @return int                Returns 0 on success, or a negative error code on failure.
 */
int luna_mtq_setup(uint32_t ch, luna_mtq_config_t *config);


/**
 * @brief Registers an event callback function for a specific MTQ channel.
 *
 * This function sets the event handler and associated user data for a given MTQ channel.
 * The registered callback will be invoked when certain events (e.g., completion, timeout)
 * occur on that channel.
 *
 * @param ch           Channel index (0-based, must be < MTQ_MAX_CHANNELS).
 * @param func         Pointer to the event handling function. NULL can be used to unregister.
 * @param userdata     User-defined pointer passed back to the event handler.
 *
 * @return int         Returns 0 on success, or a negative error code on failure.
 */
int luna_mtq_set_event(uint32_t ch, luna_mtq_event_handler_t func, void* userdata);

/**
 * @brief Enables a specific MTQ channel.
 *
 * This function enables a specific MTQ channel, allowing it to process operations.
 *
 * @param ch           Channel index (0-based, must be < MTQ_MAX_CHANNELS).
 *
 * @return int         Returns 0 on success, or a negative error code on failure.
 */
int luna_mtq_enable(uint32_t ch);

/**
 * @brief Disables a specific MTQ channel.
 *
 * This function disables a specific MTQ channel, preventing it from processing operations.
 *
 * @param ch           Channel index (0-based, must be < MTQ_MAX_CHANNELS).
 *
 * @return int         Returns 0 on success, or a negative error code on failure.
 */
int luna_mtq_disable(uint32_t ch);

/**
 * @brief Resets a specific MTQ channel.
 *
 * This function clear a specific MTQ channel, clearing its internal state and queues.
 *
 * @param ch           Channel index (0-based, must be < MTQ_MAX_CHANNELS).
 *
 * @return int         Returns 0 on success, or a negative error code on failure.
 */
int luna_mtq_clear(uint32_t ch);

/**
 * @brief Pauses a specific MTQ channel.
 *
 * This function pauses a specific MTQ channel, preventing it from processing operations.
 *
 * @param ch           Channel index (0-based, must be < MTQ_MAX_CHANNELS).
 *
 * @return int         Returns 0 on success, or a negative error code on failure.
 */
int luna_mtq_pause(uint32_t ch);

/**
 * @brief Continues a specific MTQ channel.
 *
 * This function continues a specific MTQ channel, allowing it to process operations again.
 *
 * @param ch           Channel index (0-based, must be < MTQ_MAX_CHANNELS).
 *
 * @return int         Returns 0 on success, or a negative error code on failure.
 */
int luna_mtq_continue(uint32_t ch);

/**
 * @brief Checks if a specific MTQ channel is enabled.
 *
 * This function checks if a specific MTQ channel is currently enabled.
 *
 * @param ch           Channel index (0-based, must be < MTQ_MAX_CHANNELS).
 *
 * @return int         Returns 0 if the channel is enabled, or a negative error code if it's disabled.
 */
int luna_mtq_is_enabled(uint32_t ch);

/**
 * @brief Checks if a specific MTQ channel is paused.
 *
 * This function checks if a specific MTQ channel is currently paused.
 *
 * @param ch           Channel index (0-based, must be < MTQ_MAX_CHANNELS).
 *
 * @return int         Returns 0 if the channel is paused, or a negative error code if it's not paused.
 */
int luna_mtq_is_paused(uint32_t ch);

/**
 * @brief Pushes elements to the Submit Queue (SQ) for a specific MTQ channel.
 *
 * This function pushes elements to the Submit Queue (SQ) for a specific MTQ channel.
 *
 * @param ch           Channel index (0-based, must be < MTQ_MAX_CHANNELS).
 * @param sq_elem      Pointer to an array of luna_mtq_sq_elem_t elements to be pushed.
 * @param sq_elem_num  Number of elements to be pushed.
 *
 * @return int         Returns the number of elements actually pushed, or a negative error code on failure.
 */
int luna_mtq_sq_push_doorbell(uint32_t ch, luna_mtq_sq_elem_t *sq_elem, uint32_t sq_elem_num);

/**
 * @brief Pops elements from the Completion Queue (CQ) for a specific MTQ channel.
 *
 * This function pops elements from the Completion Queue (CQ) for a specific MTQ channel.
 *
 * @param ch           Channel index (0-based, must be < MTQ_MAX_CHANNELS).
 * @param cq_elem      Pointer to an array of luna_mtq_cq_elem_t elements to be popped.
 * @param cq_elem_num  Number of elements to be popped.
 *
 * @return int         Returns the number of elements actually popped, or a negative error code on failure.
 */
int luna_mtq_cq_pop_doorbell(uint32_t ch, luna_mtq_cq_elem_t *cq_elem, uint32_t cq_elem_num);

/**
 * @brief Retrieves the current status of a specific MTQ (Multi-Task Queue) channel.
 *
 * This function fetches the current state of the specified MTQ channel, including
 * the head and tail pointers of both the Submit Queue (SQ) and Completion Queue (CQ),
 * as well as their respective sizes.
 *
 * @param ch     The channel index (0-based, must be less than MTQ_MAX_CHANNELS).
 * @param status A pointer to a luna_mtq_status_t structure where the status will be stored.
 *
 * @return int   Returns 0 on success, or a negative error code on failure.
 */
int luna_mtq_get_status(uint32_t ch, luna_mtq_status_t *status);

int luna_mtq_wait(uint32_t ch);

#endif //__LUNA_MTQ_H_