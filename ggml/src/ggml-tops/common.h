#pragma once

#include <ggml-tops.h>

#include <array>
#include <cassert>
#include <cfloat>
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <cstring>
#include <chrono>

#include <tops_runtime_api.h>
#include <driver_types.h>

#include "ggml-impl.h"

[[noreturn]] void ggml_tops_error(const char* stmt, const char* func,
    const char* file, int line, const char* msg);

#define ACL_TOPS_GEN(stmt, success, error_fn)                                \
    do {                                                                      \
        topsError_t err_code = (stmt);                                                \
        if (err_code != (success)) {                                          \
            ggml_tops_error(#stmt, __func__, __FILE__, __LINE__, error_fn(err_code)); \
        }                                                                     \
    } while (0);

#define TOPS_CHECK(stmt) ACL_TOPS_GEN(stmt, 0, topsGetErrorString)

#define TOPS_ATEN_CHECK(call)                                        \
  {                                                                  \
    const topsatenStatus_t error = call;                             \
    if (error != TOPSATEN_STATUS_SUCCESS) {                          \
      printf("Error: %s:%d, error=%d\n", __FILE__, __LINE__, error); \
      exit(1);                                                       \
    }                                                                \
  }

#define UNUSED_PTR(ptr) \
{   \
    free(ptr);\
    ptr = nullptr;\
}

#define GGML_TOPS_MAX_DEVICES 16
#define GGML_TOPS_MAX_STREAMS 8

struct ggml_tops_device_info {
    int device_count;

    struct tops_device_info {
        int     cc;                 // compute capability
        int     nsm;                // number of streaming multiprocessors
        size_t  smpb;               // max. shared memory per block
        size_t  smpbo;              // max. shared memory per block (with opt-in)
        bool    integrated;         // Device is integrated as opposed to discrete
        bool    vmm;                // virtual memory support
        size_t  vmm_granularity;    // granularity of virtual memory
        size_t  total_vram;
        int     warp_size;          // Number of threads in a dispatch
    };

    tops_device_info devices[GGML_TOPS_MAX_DEVICES] = {};

    std::array<float, GGML_TOPS_MAX_DEVICES> default_tensor_split = {};
};

void ggml_tops_set_device(const int32_t device);

struct ggml_tops_pool {
    virtual ~ggml_tops_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void free(void * ptr, size_t size) = 0;
};

/**
 * @brief A pool of TOPS buffers(segment buffer).
 *
 * This class manages a pool of TOPS buffers for a specific device.
 */
struct ggml_tops_pool_buf : public ggml_tops_pool {
    /**
     * @brief The maximum reuse margin for a buffer.
     */
    static const size_t max_reuse_margin = 1ull << 22;  // 4MB

    /**
     * @brief The minimum free margin for a buffer.
     */
    static const size_t min_free_margin = 1ull << 20;   // 1MB

    /**
     * @brief The alignment for buffer allocation.
     */
    static const size_t alignment = 128;

    /**
     * @brief The maximum number of buffers in the pool.
     */
    static const int MAX_BUFFERS = 256;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    int device;

    /**
     * @brief Whether to disable clean during buffer allocation.
     */
    bool disable_clean = false;

    /**
     * @brief Structure representing a CANN buffer.
     */
    struct ggml_tops_buffer {
        void* ptr = nullptr;  ///< Pointer to the buffer memory.
        size_t size = 0;      ///< Size of the buffer.
        bool used = false;    ///< Whether the buffer is currently in use.
        std::chrono::steady_clock::time_point last_used;  ///< Last used time.
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    ggml_tops_buffer buffer_pool[MAX_BUFFERS] = {};

    /**
     * @brief Total size of all buffers in the pool.
     */
    size_t pool_size = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    explicit ggml_tops_pool_buf(int device) : device(device) {
        disable_clean = true;
    }

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_tops_pool_buf() {
        ggml_tops_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_tops_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
                TOPS_CHECK(topsFree(b.ptr));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    void* alloc(size_t size, size_t* actual_size) override {
        size = GGML_PAD(size, alignment);
        if (size == 0) {
            size = alignment;
        }

        void* ptr = nullptr;
        auto now = std::chrono::steady_clock::now();

        int i = 0;
        for (; i < MAX_BUFFERS; ++i) {
            ggml_tops_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                break;
            }
            if (b.used) {
                continue;
            }
            if (b.size >= size) {
                // reuse the buffer if the size is enough
                const size_t margin = b.size - size;
                if (margin <= max_reuse_margin) {
                    *actual_size = b.size;
                    b.used = true;
                    ptr = b.ptr;
                    GGML_LOG_INFO(
                        "tops pool[%d]: reused   %p, "
                        "pool_size = %5u MB, "
                        "size = %5u MB, "
                        "margin = %5u MB\n",
                        device, b.ptr,
                        (uint32_t)(GGML_PAD(pool_size, 1048576) / 1048576),
                        (uint32_t)(GGML_PAD(size, 1048576) / 1048576),
                        (uint32_t)(GGML_PAD(margin, 1048576) / 1048576));
                    break;
                }
            }

            bool should_clean = !disable_clean &&
                                b.size > min_free_margin &&
                                std::chrono::duration_cast<std::chrono::milliseconds>(now - b.last_used).count() > 100;
            if (should_clean) {
                // free the buffer if the size is needed to be freed
                TOPS_CHECK(topsFree(b.ptr));
                pool_size -= b.size;
                GGML_LOG_INFO(
                    "tops pool[%d]: clean    %p, "
                    "pool_size = %5u MB, "
                    "size = %5u MB\n",
                    device, b.ptr,
                    (uint32_t)(GGML_PAD(pool_size, 1048576) / 1048576),
                    (uint32_t)(GGML_PAD(b.size, 1048576) / 1048576));
                b.ptr = nullptr;
            }
        }
        if (ptr != nullptr) {
            return ptr;
        }

        if (i < MAX_BUFFERS) {
            // allocate a new buffer if no buffer can be reused
            ggml_tops_buffer& b = buffer_pool[i];
            ggml_tops_set_device(device);
            TOPS_CHECK(topsMalloc(&b.ptr, size));
            pool_size += size;
            *actual_size = size;
            b.size = size;
            b.used = true;
            if (i >= MAX_BUFFERS - 8) {
                GGML_LOG_INFO("tops pool[%d]: slots almost full\n", device);
            }
            GGML_LOG_INFO(
                "tops pool[%d]: allocate %p, "
                "pool_size = %5u MB, "
                "size = %5u MB\n",
                device, b.ptr,
                (uint32_t)(GGML_PAD(pool_size, 1048576) / 1048576),
                (uint32_t)(GGML_PAD(b.size, 1048576) / 1048576));
            return b.ptr;
        }

        GGML_ABORT("cann pool[%d]: slots full\n", device);
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    void free(void* ptr, size_t size) override {
        GGML_UNUSED(size);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_tops_buffer& b = buffer_pool[i];
            if (b.ptr != ptr) {
                continue;
            }
            b.used = false;
            b.last_used = std::chrono::steady_clock::now();
            GGML_LOG_INFO(
                "tops pool[%d]: return   %p, "
                "pool_size = %5u MB\n",
                device, b.ptr,
                (uint32_t)(GGML_PAD(pool_size, 1048576) / 1048576));
            return;
        }
        GGML_ABORT("tops pool[%d]: slots full\n", device);
    }
};

/**
 * @brief RAII wrapper for managing memory allocations from a TOPS memory pool.
 */
struct ggml_tops_pool_alloc {
    ggml_tops_pool* pool = nullptr; /**< Pointer to the memory pool. */
    void* ptr = nullptr;    /**< Pointer to the allocated memory block. */
    size_t actual_size = 0; /**< Actual size of the allocated memory block. */

    /**
     * @brief Default constructor.
     */
    ggml_tops_pool_alloc() = default;

    /**
     * @brief Constructor that initializes the memory pool.
     * @param pool Reference to the memory pool.
     */
    explicit ggml_tops_pool_alloc(ggml_tops_pool& pool) : pool(&pool) {}

    /**
     * @brief Constructor that initializes the memory pool and allocates memory.
     * @param pool Reference to the memory pool.
     * @param size Size of the memory block to allocate.
     */
    ggml_tops_pool_alloc(ggml_tops_pool& pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    /**
     * @brief Destructor that frees the allocated memory block.
     */
    ~ggml_tops_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    /**
     * @brief Allocates memory from the pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    void* alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = pool->alloc(size, &this->actual_size);
        return ptr;
    }

    /**
     * @brief Allocates memory from a specific memory pool.
     * @param pool Reference to the memory pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    void* alloc(ggml_tops_pool& pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    /**
     * @brief Gets the pointer to the allocated memory block.
     * @return Pointer to the allocated memory block.
     */
    void* get() { return ptr; }

    // Deleted copy constructor
    ggml_tops_pool_alloc(const ggml_tops_pool_alloc&) = delete;

    // Deleted move constructor
    ggml_tops_pool_alloc(ggml_tops_pool_alloc&&) = delete;

    // Deleted copy assignment operator
    ggml_tops_pool_alloc& operator=(const ggml_tops_pool_alloc&) = delete;

    // Deleted move assignment operator
    ggml_tops_pool_alloc& operator=(ggml_tops_pool_alloc&&) = delete;
};


struct ggml_backend_tops_context {
    int32_t device;
    std::string name;
    topsEvent_t copy_event = nullptr;

    topsStream_t streams[GGML_TOPS_MAX_DEVICES][GGML_TOPS_MAX_STREAMS] = { { nullptr } };

    explicit ggml_backend_tops_context(int device) :
        device(device),
        name(GGML_TOPS_NAME + std::to_string(device)) {
    }

    ~ggml_backend_tops_context() {
        ggml_tops_set_device(device);
        if (copy_event != nullptr) {
            TOPS_CHECK(topsEventDestroy(copy_event));
        }
        for (int i = 0; i < GGML_TOPS_MAX_STREAMS; ++i) {
            if (streams[device][i] != nullptr) {
                TOPS_CHECK(topsStreamDestroy(streams[device][i]));
            }
        }
    }

    topsStream_t stream(int device, int stream) {
        if (streams[device][stream] == nullptr) {
            ggml_tops_set_device(device);
            TOPS_CHECK(topsStreamCreateWithFlags(&streams[device][stream], topsStreamNonBlocking));
        }
        return streams[device][stream];
    }

    topsStream_t stream() {
        return stream(device, 0);
    }

    // pool
    std::unique_ptr<ggml_tops_pool> pools[GGML_TOPS_MAX_DEVICES];

    static std::unique_ptr<ggml_tops_pool> new_pool_for_device(int device);

    ggml_tops_pool & pool(int device) {
        if (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(device);
        }
        return *pools[device];
    }

    ggml_tops_pool & pool() {
        return pool(device);
    }
};