// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_MEMORY_POOLS_H_
#define IREE_HAL_DRIVERS_HIP_MEMORY_POOLS_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/api.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/hip_headers.h"

// WARNING: hipMemPool API's are marked as beta in HIP library meaning
// that while the feature is complete, it is still open to changes and may
// have outstanding issues.
// The following API's are used in this module: hipMallocFromPoolAsync,
// hipMemPoolCreate, hipMemPoolDestroy, hipMemPoolGetAttribute,
// hipMemPoolSetAttribute, hipMemPoolTrimTo

// Retained HIP memory pools for various allocation types.
typedef struct iree_hal_hip_memory_pools_t {
  // Used exclusively for DEVICE_LOCAL allocations.
  hipMemPool_t device_local;
  // Used for any host-visible/host-local memory types.
  hipMemPool_t other;

  iree_hal_device_t* parent_device;
  const iree_hal_hip_dynamic_symbols_t* hip_symbols;
  iree_allocator_t host_allocator;

  IREE_STATISTICS(struct {
    iree_atomic_int64_t device_bytes_allocated;
    iree_atomic_int64_t device_bytes_freed;
    iree_atomic_int64_t host_bytes_allocated;
    iree_atomic_int64_t host_bytes_freed;
  } statistics;)
} iree_hal_hip_memory_pools_t;

// Initializes |out_pools| by configuring new HIP memory pools.
iree_status_t iree_hal_hip_memory_pools_initialize(
    iree_hal_device_t* parent_device,
    const iree_hal_hip_dynamic_symbols_t* hip_symbols, hipDevice_t hip_device,
    const iree_hal_hip_memory_pooling_params_t* pooling_params,
    iree_allocator_t host_allocator,
    iree_hal_hip_memory_pools_t* IREE_RESTRICT out_pools);

// Deinitializes the |pools| and releases the underlying HIP resources.
void iree_hal_hip_memory_pools_deinitialize(iree_hal_hip_memory_pools_t* pools);

// Merges statistics information from |pools| into |statistics|.
void iree_hal_hip_memory_pools_merge_statistics(
    iree_hal_hip_memory_pools_t* pools,
    iree_hal_allocator_statistics_t* statistics);

// Trims all memory pools by releasing resources back to the system.
iree_status_t iree_hal_hip_memory_pools_trim(
    iree_hal_hip_memory_pools_t* pools,
    const iree_hal_hip_memory_pooling_params_t* pooling_params);

iree_status_t iree_hal_hip_memory_pools_allocate_pointer(
    iree_hal_hip_memory_pools_t* pools, iree_hal_buffer_t* buffer,
    hipStream_t stream, iree_device_size_t allocation_size);

// Prepares a buffer to be allocated from the given pool.
// It does no actual allocations, they must happen on another thread.
// Any calls to get the device_ptr from the buffer will
// result in a wait until the allocation is available.
iree_status_t iree_hal_hip_memory_pools_prepare_buffer(
    iree_hal_hip_memory_pools_t* pools, hipStream_t stream,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer);

// Asynchronously deallocates a buffer from its pool.
// The deallocation will be stream-ordered on |stream|.
iree_status_t iree_hal_hip_memory_pools_deallocate(
    iree_hal_hip_memory_pools_t* pools, hipStream_t stream,
    iree_hal_buffer_t* buffer);

#endif  // IREE_HAL_DRIVERS_HIP_MEMORY_POOLS_H_
