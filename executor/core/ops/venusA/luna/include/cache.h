/*
 * cache.h
 *
 *  Created on: Jul 10, 2020
 *
 */

#ifndef INCLUDE_CACHE_H_
#define INCLUDE_CACHE_H_

#define HAL_ICACHE_VALID                            1
#define HAL_ICACHE_CFG_LINE_SIZE                    32
#define HAL_ICACHE_CFG_WAYS                         2
#define HAL_ICACHE_CFG_SET                          512
#if HAL_ICACHE_VALID
#define HAL_ICACHE_RAM_SIZE                         (32*1024)
#else
#define HAL_ICACHE_RAM_SIZE                         0
#endif

#define HAL_DCACHE_VALID                            1
#define HAL_DCACHE_CFG_LINE_SIZE                    32
#define HAL_DCACHE_CFG_WAYS                         2
#define HAL_DCACHE_CFG_SET                          256
#if HAL_DCACHE_VALID
#define HAL_DCACHE_RAM_SIZE                         (16*1024)
#else
#define HAL_DCACHE_RAM_SIZE                         0
#endif

#include "stdint.h"

/**
 * @brief Enables the CPU instruction cache.
 *
 * This function activates the internal instruction cache of the CPU
 * to enhance the execution speed of programs. It is part of the Hardware
 * Abstraction Layer (HAL), providing an interface to manage hardware-specific
 * features such as caching operations. Enabling the instruction cache allows
 * for faster access to frequently executed instructions, which is crucial
 * for performance-critical applications.
 */
void HAL_EnableICache(void);

/**
 * @brief Disables the CPU instruction cache.
 *
 * This function deactivates the internal instruction cache of the CPU
 * to potentially aid in debugging or to meet specific system requirements
 * where caching of instructions needs to be prevented. It is part of the Hardware
 * Abstraction Layer (HAL), providing an interface to manage hardware-specific
 * features such as caching operations. Disabling the instruction cache may be
 * necessary in scenarios where precise control over instruction execution is required.
 */
void HAL_DisableICache(void);

/**
 * @brief Invalidates the CPU instruction cache.
 *
 * This function clears the contents of the internal instruction cache of the CPU.
 * Invalidating the cache is useful to ensure that no stale or corrupted data is used
 * by the CPU, which is particularly important after direct memory access (DMA) operations
 * or after loading new programs into memory. It is part of the Hardware Abstraction Layer (HAL),
 * providing an interface to manage hardware-specific features such as caching operations.
 * This operation helps in maintaining data coherency and consistency across the system.
 */
void HAL_InvalidateICache(void);

/**
 * @brief Enables the CPU data cache.
 *
 * This function activates the internal data cache of the CPU
 * to enhance the execution speed and efficiency of data access and processing.
 * It is part of the Hardware Abstraction Layer (HAL), providing an interface to manage
 * hardware-specific features such as caching operations. Enabling the data cache helps
 * improve system performance by reducing memory access times and minimizing CPU idle time
 * during data fetches from main memory.
 */
void HAL_EnableDCache(void);

/**
 * @brief Disables the CPU data cache.
 *
 * This function deactivates the internal data cache of the CPU.
 * Disabling the data cache can be useful in scenarios where data caching may lead
 * to consistency issues, such as during non-cache-coherent DMA operations or when
 * the predictability of every data access is required. It is part of the Hardware
 * Abstraction Layer (HAL), providing an interface to manage hardware-specific features.
 * Disabling the data cache ensures that all data reads and writes are directly made to
 * and from the main memory, which can be crucial for real-time and safety-critical applications.
 */
void HAL_DisableDCache(void);


/**
 * @brief Invalidates the CPU data cache.
 *
 * This function clears the contents of the internal data cache of the CPU.
 * Invalidating the cache is essential to prevent the use of stale or incorrect data
 * that might remain after changes in memory. It is commonly used after direct memory
 * access (DMA) operations or when hardware devices modify memory outside of the CPU's control.
 * It is part of the Hardware Abstraction Layer (HAL), providing an interface to manage
 * hardware-specific features such as caching operations. This operation ensures data coherency
 * and consistency across the system, particularly in systems where memory is shared between the CPU
 * and other hardware components.
 */
void HAL_InvalidateDCache(void);

/**
 * @brief Flushes the CPU data cache.
 *
 * This function ensures that all modified data within the CPU's internal data cache
 * are written back to the main memory. Flushing the data cache is crucial before
 * any operations that require up-to-date data from other processors or hardware
 * that do not have cache coherency mechanisms. It is part of the Hardware
 * Abstraction Layer (HAL), providing an interface to manage hardware-specific features
 * such as caching operations. This operation helps maintain data coherency and
 * consistency across different parts of the system, particularly in multi-core
 * or multi-processor environments.
 */
void HAL_FlushDCache(void);

/**
 * @brief Flushes and invalidates the CPU data cache.
 *
 * This function ensures that all modified data within the CPU's internal data cache
 * are written back to the main memory, and then invalidates the cache to remove all entries.
 * This is particularly useful in scenarios where data coherence and consistency are critical,
 * such as before DMA operations where peripheral devices need to access the latest data,
 * or after updating firmware that changes the memory layout. It is part of the Hardware
 * Abstraction Layer (HAL), providing an interface to manage hardware-specific features
 * such as caching operations. Flushing and invalidating the data cache ensures that
 * no stale data is used and all future data reads are done directly from the main memory.
 */
void HAL_FlushInvalidateDCache(void);

/**
 * @brief Invalidates a range of the CPU data cache based on address and size.
 *
 * This function clears a specific portion of the CPU's internal data cache. By providing
 * an address and the size of the area, this function ensures that any modifications in
 * memory in this specified range do not use stale or outdated cache entries. This is particularly
 * useful for systems where memory regions are dynamically altered or when devices not supporting
 * cache coherency modify the memory. It helps in maintaining data integrity and coherency
 * especially in systems involving direct memory access (DMA) operations.
 *
 * @param addr Pointer to the start address of the memory region to invalidate.
 * @param dsize Size of the memory region to invalidate, in bytes.
 */
void HAL_InvalidateDCache_by_Addr(uint32_t *addr, uint32_t dsize);

/**
 * @brief Flushes a range of the CPU data cache based on address and size.
 *
 * This function writes back all modified data within a specified range of the CPU's internal data cache
 * to the main memory. This operation is crucial for ensuring data coherence in systems where other processors
 * or hardware devices access the same memory region but do not share a cache coherency mechanism. It is typically
 * used prior to DMA operations or when processors in a multi-processor system access shared data.
 *
 * @param addr Pointer to the start address of the memory region to flush.
 * @param dsize Size of the memory region to flush, in bytes.
 */
void HAL_FlushDCache_by_Addr(uint32_t *addr, uint32_t dsize);

/**
 * @brief Flushes and invalidates a range of the CPU data cache based on address and size.
 *
 * This function combines the actions of writing back all modified data within a specified range
 * of the CPU's internal data cache to the main memory and then invalidating the cache entries.
 * This ensures that no stale data remains and all future accesses to this memory range will be
 * fetched directly from the main memory. This operation is particularly vital in systems with
 * non-cache-coherent DMA operations or in multi-core systems where processors need to share
 * up-to-date data without any inconsistencies.
 *
 * @param addr Pointer to the start address of the memory region to flush and invalidate.
 * @param dsize Size of the memory region to flush and invalidate, in bytes.
 */
void HAL_FlushInvalidateDCache_by_Addr(uint32_t *addr, uint32_t dsize);


// This function will be abandon
int range_is_cacheable(unsigned long start, unsigned long size);
void dcache_clean_range(unsigned long start, unsigned long end);
void dcache_invalidate_range(unsigned long start, unsigned long end);
void dcache_flush_range(unsigned long start, unsigned long end);
void cache_dma_fast_inv_stage1(unsigned long start, unsigned long end);
void cache_dma_fast_inv_stage2(unsigned long start, unsigned long end);

#endif /* INCLUDE_CACHE_H_ */
