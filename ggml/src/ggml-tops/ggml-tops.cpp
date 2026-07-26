#include "ggml-tops.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-tops/common.h"
#include "ggml-tops/ggml-tops_ops.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <float.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

[[noreturn]] void ggml_tops_error(const char* stmt, const char* func,
    const char* file, int line, const char* msg) {
    int id = -1;
    topsGetDevice(&id);

    GGML_LOG_ERROR("TOPS error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func,
    file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ABORT("TOPS error");
}

void ggml_tops_set_device(int device) {
    int current_device;
    TOPS_CHECK(topsGetDevice(&current_device));

    if (device == current_device) {
        return;
    }

    TOPS_CHECK(topsSetDevice(device));
}

// backend device

struct ggml_backend_tops_device_context {
    int device;
    std::string name;
    std::string description;
};

struct ggml_backend_tops_buffer_type_context {
    int32_t device; /**< Device identifier associated with the buffer context. */
    std::string name; /**< Name associated with the buffer context. */
};

static ggml_tops_device_info ggml_tops_init() {
    ggml_tops_device_info info = {};
    //TODO
    TOPS_CHECK(topsGetDeviceCount(&info.device_count));

    GGML_ASSERT(info.device_count <= GGML_TOPS_MAX_DEVICES);
    for (int id = 0; id < info.device_count; ++id) {
        //TODO
    }

    return info;
}

const ggml_tops_device_info & ggml_tops_info() {
    static ggml_tops_device_info info = ggml_tops_init();
    return info;
}

static enum ggml_backend_dev_type ggml_backend_tops_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}


static const char * ggml_backend_tops_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_tops_device_context * ctx = (ggml_backend_tops_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_tops_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_tops_device_context * ctx = (ggml_backend_tops_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_tops_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_tops_device_context * ctx = (ggml_backend_tops_device_context *)dev->context;

    int device = ctx->device;
    ggml_tops_set_device(device);
    TOPS_CHECK(topsMemGetInfo(free,total))

    return;
}

static bool ggml_backend_tops_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_tops_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    //TODO need to replace with sip 
    return 128;

    GGML_UNUSED(buft);
}

struct ggml_backend_tops_buffer_context {
    int32_t device;  ///< The device ID associated with this buffer context.
    void* dev_ptr = nullptr;  ///< Pointer to the device memory allocated for the buffer.

    /**
     * @brief Constructor to initialize the TOPS buffer context.
     *
     * @param device The device ID associated with this buffer context.
     * @param dev_ptr Pointer to the device memory allocated for the buffer.
     */
    ggml_backend_tops_buffer_context(int32_t device, void* dev_ptr)
        : device(device),
          dev_ptr(dev_ptr) {}

    /**
     * @brief Destructor to free the device memory allocated for the buffer.
     */
    //TODO use host memory ,need to be repalced with device memory management api 
    ~ggml_backend_tops_buffer_context() {
        if (dev_ptr != nullptr) {
            topsFree(dev_ptr);
        dev_ptr = nullptr;
        }
    }
};

/**
 * @brief Retrieve the base pointer of a TOPS buffer.
 *
 * This function returns the base pointer of a TOPS buffer, which points to the
 * device memory allocated for the buffer.
 *
 * @param buffer The TOPS buffer whose base pointer is to be retrieved.
 * @return A pointer to the base of the device memory allocated for the buffer.
 */
static void* ggml_backend_tops_buffer_get_base(
    ggml_backend_buffer_t buffer) {
    ggml_backend_tops_buffer_context* ctx =
        (ggml_backend_tops_buffer_context*)buffer->context;
    return ctx->dev_ptr;
}

static void ggml_backend_cann_buffer_clear(
    ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_tops_buffer_context* ctx =
        (ggml_backend_tops_buffer_context*)buffer->context;

    ggml_tops_set_device(ctx->device);
    TOPS_CHECK(topsMemset(ctx->dev_ptr, buffer->size, value));
}

static bool need_transform(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

static void ggml_backend_tops_transform_back(
    const ggml_tensor* tensor, void* src, void* dst) {
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            //only float ,do not need added currently
            break;
        case GGML_TYPE_Q8_0:
            //only float ,do not need added currently
            break;
        default:
            break;
    }
}

static void ggml_backend_tops_transform(ggml_tensor* tensor,
                                        const void* src, void* dst) {
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            //only float ,do not need added currently
            break;
        case GGML_TYPE_Q8_0:
            //only float ,do not need added currently
            break;
        default:
            break;
    }
}

static void ggml_backend_tops_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_tops_buffer_context * ctx = (ggml_backend_tops_buffer_context *)buffer->context;

    ggml_tops_set_device(ctx->device);
    // // TODO: refer to cann(#6017), it use thread's default stream.
    // // For acl, synchronous functions use this default stream.
    // // Why aclrtSynchronizeDevice?

    if (!need_transform(tensor->type)) {
        TOPS_CHECK(topsMemcpy((char *)tensor->data + offset, data, size,
                              topsMemcpyHostToDevice));
    } else {
        void *transform_buffer = malloc(size);
        ggml_backend_tops_transform(tensor, data, transform_buffer);

        TOPS_CHECK(topsMemcpy((char *)tensor->data + offset, 
                              transform_buffer, size,
                              topsMemcpyHostToDevice));
        free(transform_buffer);
    }
    // memcpy((char *)tensor->data + offset, data, size);
}

static void ggml_backend_tops_buffer_get_tensor(
    ggml_backend_buffer_t buffer, const ggml_tensor* tensor, void* data,
    size_t offset, size_t size) {
    ggml_backend_tops_buffer_context* ctx =
        (ggml_backend_tops_buffer_context*)buffer->context;

    ggml_tops_set_device(ctx->device);

    if (!need_transform(tensor->type)) {
        TOPS_CHECK(topsMemcpy(data, (char*)tensor->data + offset, size,
                              topsMemcpyDeviceToHost));
    } else {
        void* transform_buffer = malloc(size);
        TOPS_CHECK(topsMemcpy(transform_buffer, 
                              (char*)tensor->data + offset, size,
                              topsMemcpyDeviceToHost));
        ggml_backend_tops_transform_back(tensor, transform_buffer, data);
        free(transform_buffer);
    }
}
static void ggml_backend_tops_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_tops_buffer_context * ctx = (ggml_backend_tops_buffer_context *)buffer->context;
    delete ctx;
}

static const ggml_backend_buffer_i ggml_backend_tops_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_tops_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_tops_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_tops_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_tops_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_cann_buffer_clear,
    /* .reset           = */ NULL,
};

static ggml_backend_buffer_t ggml_backend_tops_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_tops_buffer_type_context* buft_ctx =
        (ggml_backend_tops_buffer_type_context*)buft->context;
    //TODO alloc memory from device

    const size_t alignment = ggml_backend_tops_buffer_type_get_alignment(buft);
    size = GGML_PAD(size, alignment);
    if (size == 0) {
        size = alignment;
    }
    //TODO use host memory ,need to be repalced with device memory management api
    void* dev_ptr = nullptr;
    TOPS_CHECK(topsMalloc(&dev_ptr, size));

    ggml_backend_tops_buffer_context* ctx =
        new ggml_backend_tops_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_tops_buffer_interface,
                                    ctx, size);
}

static const char* ggml_backend_tops_buffer_type_name(
    ggml_backend_buffer_type_t buft) {
    ggml_backend_tops_buffer_type_context* buft_ctx =
        (ggml_backend_tops_buffer_type_context*)buft->context;

    return buft_ctx->name.c_str();
}

static const ggml_backend_buffer_type_i ggml_backend_tops_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_tops_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_tops_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_tops_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ NULL,
    /* .is_host          = */ ggml_backend_tops_buffer_type_is_host,
};

int ggml_backend_tops_get_device_count() {
    return ggml_tops_info().device_count;
}

static ggml_backend_buffer_type_t ggml_backend_tops_buffer_type(int32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_tops_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type
        ggml_backend_tops_buffer_types[GGML_TOPS_MAX_DEVICES];

    static bool ggml_backend_tops_buffer_type_initialized = false;

    if (!ggml_backend_tops_buffer_type_initialized) {
        for (int32_t i = 0; i < ggml_backend_tops_get_device_count(); i++) {
            ggml_backend_tops_buffer_types[i] = {
                /* .iface    = */ ggml_backend_tops_buffer_type_interface,
                /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_tops_reg(), i),
                /* .context  = */
                 new ggml_backend_tops_buffer_type_context{
                    i, "tops" + std::to_string(i)},
            };
        }
        ggml_backend_tops_buffer_type_initialized = true;
    }

    return &ggml_backend_tops_buffer_types[device];
}

static ggml_backend_buffer_type_t ggml_backend_tops_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_tops_device_context * ctx = (ggml_backend_tops_device_context *)dev->context;
    return ggml_backend_tops_buffer_type(ctx->device);
}

static bool ggml_backend_tops_supports_op(ggml_backend_dev_t dev, const ggml_tensor* op) {
    //TODO need recheck every operator according to the spec, now the data type is float(fp32 or fp 16)
    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_STEP:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT: {
            switch (op->src[0]->type) {
                case GGML_TYPE_F16:
                case GGML_TYPE_F32:
                    return true;
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_0:
#ifdef ASCEND_310P
                    // Q4 && Q8 per group is not suppor on 310p device
                    return false;
#endif
                    // only support contiguous for quantized types.
                    return ggml_is_contiguous(op->src[0]) &&
                            ggml_is_contiguous(op->src[1]);
                default:
                    return false;
            }
        }
        case GGML_OP_MUL_MAT_ID:
            switch (op->src[0]->type) {
                case GGML_TYPE_F16:
                case GGML_TYPE_F32:
                    return true;
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_0:
#ifdef ASCEND_310P
                    // Q4 && Q8 per group is not suppor on 310p device
                    return false;
#endif
                    // only support contiguous for quantized types.
                    return ggml_is_contiguous(op->src[0]) &&
                            ggml_is_contiguous(op->src[1]);
                default:
                    return false;
            }
        // embedding
        case GGML_OP_GET_ROWS: {
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q8_0:
                    return true;
                default:
                    return false;
            }
        } break;
        case GGML_OP_SET_ROWS:
            {
                // TODO: add support
                // ref: https://github.com/ggml-org/llama.cpp/pull/14274
#pragma message("TODO: implement F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL support (https://github.com/ggml-org/llama.cpp/pull/14661)")
                return false;
            } break;
        case GGML_OP_CPY: {
            ggml_tensor *src = op->src[0];
            if ((op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) ||
                  (src->type != GGML_TYPE_F32 &&
                    src->type != GGML_TYPE_F16)) {
                // only support F32 and F16.
                return false;
            }

            if (!ggml_are_same_shape(op, src) && !ggml_is_contiguous(op)) {
                // unsupport dst is not contiguous.
                return false;
            }

            return true;
        } break;
        case GGML_OP_CONT: {
            // TODO: support GGML_TYPE_BF16
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                    return true;
                default:
                    return false;
            }
        }
        case GGML_OP_ROPE: {
            // TODO: with ops-test v == 1
            // float ext_factor = 0.0f;
            // memcpy(&ext_factor, (const float *) op->op_params + 7, sizeof(float));
            // // TODO: n_dims <= ne0
            // if (op->src[0]->ne[0] != op->op_params[1]) {
            //     return false;
            // }
            // // TODO: ext_factor != 0
            // if (ext_factor != 0) {
            //     return false;
            // }

            // const int mode = ((const int32_t *) op->op_params)[2];
            // if (mode & GGML_ROPE_TYPE_MROPE) {
            //     return false;
            // }
            // if (mode & GGML_ROPE_TYPE_VISION) {
            //     return false;
            // }

            // if(!ggml_is_contiguous(op->src[0])){
            //     return false;
            // }
            return true;
        }
        case GGML_OP_UPSCALE: {
            // aclnnUpsampleNearest2dGetWorkspaceSize not support
            // selfDimN[2]/outDimN[2] or selfDimC[3]/outDimC[3] not equal
            if (op->src[0]->ne[2] * op->ne[3] != op->src[0]->ne[3] * op->ne[2]) {
                return false;
            }
            if (op->op_params[0] != GGML_SCALE_MODE_NEAREST && op->op_params[0] != GGML_SCALE_MODE_BILINEAR) {
                return false;
            }
            return true;
        }
        case GGML_OP_POOL_2D: {
            const int32_t * opts = (const int32_t *) op->op_params;
#ifdef ASCEND_310P
            enum ggml_op_pool opt = static_cast<ggml_op_pool>(opts[0]);
            if(opt == GGML_OP_POOL_MAX){
                return false;
            }
#endif
            const int       k0   = opts[1];
            const int       k1   = opts[2];
            const int       p0   = opts[5];
            const int       p1   = opts[6];
            // value of paddingH should be at most half of kernelH
            // value of paddingW should be at most half of kernelW
            return (p0 <= (k0 / 2)) && (p1 <= (k1 / 2));
        }
        case GGML_OP_SUM:
        case GGML_OP_DUP:
        case GGML_OP_IM2COL:
        case GGML_OP_CONCAT:
        case GGML_OP_REPEAT:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NORM:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_CLAMP:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT:
        case GGML_OP_ACC:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_PAD:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_ARGMAX:
        case GGML_OP_COS:
        case GGML_OP_SIN:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_LOG:
        case GGML_OP_MEAN:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_COUNT_EQUAL:
            return true;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
        case GGML_OP_SCALE:
            float bias;
            memcpy(&bias, (float*)op->op_params + 1, sizeof(float));
            return bias == 0.0f; // TODO: support bias != 0.0f
        case GGML_OP_SOFT_MAX:
            // TODO: support broadcast
            // ref: https://github.com/ggml-org/llama.cpp/pull/14435
            return !op->src[1] || (op->src[1]->ne[2] == 1 && op->src[1]->ne[3] == 1);
        case GGML_OP_FLASH_ATTN_EXT:{
            // derived from [ggml-cuda.cu]
            if(op->src[1]->type != GGML_TYPE_F16 || op->src[2]->type != GGML_TYPE_F16){
                return false;
            }
            if(op->src[1]->type != GGML_TYPE_F16 && op->src[1]->type != GGML_TYPE_F32 && op->src[1]->type != GGML_TYPE_BF16){
                return false;
            }
            if(op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_BF16){
                return false;
            }
            if (op->src[1]->ne[0] != op->src[2]->ne[0]) {
                // different head sizes of K and V are not supported yet
                return false;
            }
            if (op->src[0]->ne[0] == 192) {
                return false;
            }
            if (op->src[0]->ne[0] == 576) {
                // DeepSeek MLA
                return false;
            }
            // TODO: support broadcast
            // ref: https://github.com/ggml-org/llama.cpp/pull/14435
            if (op->src[0]->ne[3] != 1) {
                return false;
            }
            float logitSoftcap = 0.0f;
            memcpy(&logitSoftcap,  (float*)op->op_params + 2, sizeof(float));
            if(logitSoftcap != 0.0f) {
                return false;
            }
            return true;
        }
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static void ggml_backend_tops_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    //TODO
    props->name        = ggml_backend_tops_device_get_name(dev);
    props->description = ggml_backend_tops_device_get_description(dev);
    props->type        = ggml_backend_tops_device_get_type(dev);
    ggml_backend_tops_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_TOPS_NO_PINNED") == nullptr;//TODO need add later for pinned me

    props->caps = {
        /* .async                 = */ false,//TODO need check if sip support async mode
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,//TODO need check 
    };
}

static ggml_guid_t ggml_backend_tops_guid() {
    //TODO need replacd with tops
    static ggml_guid guid = {0xa1, 0x94, 0xaf, 0xac, 0xbd, 0x4f, 0x47, 0x34,
                             0xbe, 0x1a, 0x9e, 0x71, 0x1f, 0x9e, 0xed, 0x64};
    return &guid;
}

static const char* ggml_backend_tops_name(ggml_backend_t backend) {
    ggml_backend_tops_context* tops_ctx =
        (ggml_backend_tops_context*)backend->context;

    return tops_ctx->name.c_str();
}

std::unique_ptr<ggml_tops_pool> ggml_backend_tops_context::new_pool_for_device(int device) {
    
    return std::unique_ptr<ggml_tops_pool>(new ggml_tops_pool_buf(device));
}

static bool ggml_tops_compute_forward(ggml_backend_tops_context& ctx,
                                      struct ggml_tensor* dst) {
    switch (dst->op) {
        case GGML_OP_GLU:
            printf("GGML_OP_GLU\n");
            ggml_tops_op_glu(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            printf("GGML_OP_REPEAT\n");
            ggml_tops_op_repeat(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:

            break;
        case GGML_OP_DUP:
            printf("GGML_OP_DUP\n");
            ggml_tops_op_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
            printf("GGML_OP_ADD\n");
            ggml_tops_op_add(ctx, dst);
            break;
        case GGML_OP_SUB:
            printf("GGML_OP_SUB\n");
            ggml_tops_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            printf("GGML_OP_ACC\n");
            ggml_tops_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            printf("GGML_OP_MUL\n");
            ggml_tops_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            printf("GGML_OP_DIV\n");
            ggml_tops_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    printf("GGML_UNARY_OP_ABS\n");
                    ggml_tops_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    printf("GGML_UNARY_OP_NEG\n");
                    ggml_tops_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    printf("GGML_UNARY_OP_GELU\n");
                    ggml_tops_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    printf("GGML_UNARY_OP_SILU\n");
                    ggml_tops_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    printf("GGML_UNARY_OP_GELU_QUICK\n");
                    ggml_tops_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    printf("GGML_UNARY_OP_TANH\n");
                    ggml_tops_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    printf("GGML_UNARY_OP_RELU\n");
                    ggml_tops_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    printf("GGML_UNARY_OP_SIGMOID\n");
                    ggml_tops_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    printf("GGML_UNARY_OP_HARDSIGMOID\n");
                    ggml_tops_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    printf("GGML_UNARY_OP_HARDSWISH\n");
                    ggml_tops_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    printf("GGML_UNARY_OP_EXP\n");
                    ggml_tops_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    printf("GGML_UNARY_OP_ELU\n");
                    ggml_tops_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    printf("GGML_UNARY_OP_SGN\n");
                    ggml_tops_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            printf("GGML_OP_NORM\n");
            ggml_tops_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            printf("GGML_OP_GROUP_NORM\n");
            ggml_tops_op_groupnorm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            printf("GGML_OP_CONCAT\n");
            ggml_tops_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            printf("GGML_OP_UPSCALE\n");
            ggml_tops_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            printf("GGML_OP_PAD\n");
            ggml_tops_op_pad(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            printf("GGML_OP_ARANGE\n");
            ggml_tops_op_arange(ctx, dst);
            break;

        case GGML_OP_TIMESTEP_EMBEDDING:
            break;
        case GGML_OP_LEAKY_RELU:
            printf("GGML_OP_LEAKY_RELU\n");
            ggml_tops_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            printf("GGML_OP_RMS_NORM\n");
            ggml_tops_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            printf("GGML_OP_MUL_MAT\n");
            ggml_tops_op_mul_mat(ctx, dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            printf("GGML_OP_MUL_MAT_ID\n");
            ggml_tops_op_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_SCALE:
            printf("GGML_OP_SCALE\n");
            ggml_tops_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            GGML_ASSERT(dst->src[1] == nullptr);
            dst->src[1] = dst->src[0];
            printf("GGML_OP_SQR\n");
            ggml_tops_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            printf("GGML_OP_SQRT\n");
            ggml_tops_op_sqrt(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            printf("GGML_OP_CLAMP\n");
            ggml_tops_op_clamp(ctx, dst);
            break;
        case GGML_OP_CPY:
            printf("GGML_OP_CPY\n");
            ggml_tops_op_cpy(ctx, dst);
            break;
        case GGML_OP_CONT:
            printf("GGML_OP_CONT\n");
            ggml_tops_op_dup(ctx, dst);
            break;
        case GGML_OP_NONE:

        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            printf("GGML_OP_TRANSPOSE\n");
            ggml_tops_op_transpose(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            break;
        case GGML_OP_SOFT_MAX:
            printf("GGML_OP_SOFT_MAX\n");
            ggml_tops_op_softmax(ctx, dst);
            break;
        case GGML_OP_ROPE:
            printf("GGML_OP_ROPE\n");
            ggml_tops_op_rope(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            break;
        case GGML_OP_POOL_2D:
            printf("GGML_OP_POOL_2D\n");
            ggml_tops_op_pool_2d(ctx, dst);
            break;
        case GGML_OP_SUM:
            printf("GGML_OP_SUM\n");
            ggml_tops_op_sum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            printf("GGML_OP_SUM_ROWS\n");
            ggml_tops_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            printf("GGML_OP_ARGSORT\n");
            ggml_tops_op_argsort(ctx, dst);
            break;
        case GGML_OP_ARGMAX:
            printf("GGML_OP_ARGMAX\n");
            ggml_tops_op_argmax(ctx, dst);
            break;
        case GGML_OP_COS:
            printf("GGML_OP_COS\n");
            ggml_tops_op_cos(ctx, dst);
            break;
        case GGML_OP_SIN:
            printf("GGML_OP_SIN\n");
            ggml_tops_op_sin(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            printf("GGML_OP_CONV_TRANSPOSE_1D\n");
            ggml_tops_op_transpose_1d(ctx, dst);
            break;
        case GGML_OP_LOG:
            printf("GGML_OP_LOG\n");
            ggml_tops_op_log(ctx, dst);
            break;
        case GGML_OP_MEAN:
            printf("GGML_OP_MEAN\n");
            ggml_tops_op_mean(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            printf("GGML_OP_PAD_REFLECT_1D\n");
            ggml_tops_op_reflect_1d(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            printf("GGML_OP_COUNT_EQUAL\n");
            ggml_tops_op_count_equal(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            printf("GGML_OP_FLASH_ATTN_EXT\n");
            ggml_tops_op_flash_atten_ext(ctx, dst);
            break;
        default:
            return false;
    }

    return true;
}

static enum ggml_status ggml_backend_tops_graph_compute(
    ggml_backend_t backend, ggml_cgraph* cgraph) {
    ggml_backend_tops_context* tops_ctx =
        (ggml_backend_tops_context*)backend->context;

    ggml_tops_set_device(tops_ctx->device);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor* node = cgraph->nodes[i];

        if (ggml_is_empty(node) || node->op == GGML_OP_NONE) {
            continue;
        }

        bool ok = ggml_tops_compute_forward(*tops_ctx, node);

        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__,
                    node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_tops_free(ggml_backend_t backend) {
    ggml_backend_tops_context* tops_ctx =
        (ggml_backend_tops_context*)backend->context;

    //TODO releases resources associated with the tops backend context and resets the device associated with the backend to its initial state.
    
    delete tops_ctx;
    delete backend;
}


static bool ggml_backend_buffer_is_tops(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_tops_buffer_free_buffer;
}

static void ggml_backend_tops_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_tops_context * tops_ctx = (ggml_backend_tops_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_tops_buffer_type(tops_ctx->device) && "unsupported buffer type");

    TOPS_CHECK(topsMemcpyAsync((char *)tensor->data + offset, data, size, topsMemcpyHostToDevice, tops_ctx->stream()));
}

static void ggml_backend_tops_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_tops_context * tops_ctx = (ggml_backend_tops_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_tops_buffer_type(tops_ctx->device) && "unsupported buffer type");

    TOPS_CHECK(topsMemcpyAsync(data, (const char *)tensor->data + offset, size, topsMemcpyDeviceToHost, tops_ctx->stream()));
}

bool ggml_backend_is_tops(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_tops_guid());
}

static bool ggml_backend_tops_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_buffer_is_tops(src->buffer) || !ggml_backend_buffer_is_tops(dst->buffer)) {
        return false;
    }

    if (!ggml_backend_is_tops(backend_src) || !ggml_backend_is_tops(backend_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_tops_context * tops_ctx_src = (ggml_backend_tops_context *)backend_src->context;
    ggml_backend_tops_context * tops_ctx_dst = (ggml_backend_tops_context *)backend_dst->context;

    ggml_backend_tops_buffer_context * buf_ctx_src = (ggml_backend_tops_buffer_context *)buf_src->context;
    ggml_backend_tops_buffer_context * buf_ctx_dst = (ggml_backend_tops_buffer_context *)buf_dst->context;

    if (tops_ctx_src->device != buf_ctx_src->device || tops_ctx_dst->device != buf_ctx_dst->device) {
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);

        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        if (tops_ctx_src->device == tops_ctx_dst->device) {
            TOPS_CHECK(topsMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), topsMemcpyDeviceToDevice, tops_ctx_src->stream()));
        } else {
            return false;
        }

        // record event on src stream after the copy
        if (!tops_ctx_src->copy_event) {
            ggml_tops_set_device(tops_ctx_src->device);
            TOPS_CHECK(topsEventCreateWithFlags(&tops_ctx_src->copy_event, topsEventDisableTiming));
        }

        TOPS_CHECK(topsEventRecord(tops_ctx_src->copy_event, tops_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        TOPS_CHECK(topsStreamWaitEvent(tops_ctx_dst->stream(), tops_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        TOPS_CHECK(topsMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), topsMemcpyDeviceToDevice, tops_ctx_src->stream()));
    }

    return true;
}

static void ggml_backend_tops_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_tops_context * tops_ctx = (ggml_backend_tops_context *)backend->context;

    TOPS_CHECK(topsEventRecord((topsEvent_t)event->context, tops_ctx->stream()));
}

static void ggml_backend_tops_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_tops_context * tops_ctx = (ggml_backend_tops_context *)backend->context;

    if (ggml_backend_is_tops(backend)) {
        TOPS_CHECK(topsStreamWaitEvent(tops_ctx->stream(), (topsEvent_t)event->context, 0));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_tops_synchronize(ggml_backend_t backend) {
    ggml_backend_tops_context * tops_ctx = (ggml_backend_tops_context *)backend->context;

    TOPS_CHECK(topsStreamSynchronize(tops_ctx->stream()));

    GGML_UNUSED(backend);
}

static const ggml_backend_i ggml_backend_tops_interface = {
    /* .get_name                = */ ggml_backend_tops_name,
    /* .free                    = */ ggml_backend_tops_free,
    /* .set_tensor_async        = */ ggml_backend_tops_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_tops_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_tops_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_tops_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_tops_graph_compute,
    /* .event_record            = */ ggml_backend_tops_event_record,
    /* .event_wait              = */ ggml_backend_tops_event_wait,
};

ggml_backend_t ggml_backend_tops_init(int32_t device) {
    if (device < 0 || device >= ggml_backend_tops_get_device_count()) {
        GGML_LOG_ERROR("%s: error: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_tops_context* ctx = new ggml_backend_tops_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return nullptr;
    }
    ggml_tops_set_device(ctx->device);
    ggml_backend_t tops_backend =
        new ggml_backend{/* .guid      = */ ggml_backend_tops_guid(),
                         /* .interface = */ ggml_backend_tops_interface,
                         /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_tops_reg(), device),
                         /* .context   = */ ctx};

    return tops_backend;
}

static ggml_backend_t ggml_backend_tops_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_tops_device_context * ctx = (ggml_backend_tops_device_context *)dev->context;
    return ggml_backend_tops_init(ctx->device);
}

static bool ggml_backend_buft_is_tops(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_tops_buffer_type_name;
}

static bool ggml_backend_tops_supports_buft(
    ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (ggml_backend_buft_is_tops(buft)) {
        ggml_backend_tops_device_context * dev_ctx = (ggml_backend_tops_device_context *)dev->context;
        ggml_backend_tops_buffer_type_context * buft_ctx =
                        (ggml_backend_tops_buffer_type_context *)buft->context;
        return buft_ctx->device == dev_ctx->device;
    }
    return false;
}

/**
 * @brief Determines if a tensor operation should be offloaded to the TOPS
 * backend.
 *
 * This function checks if a given tensor operation should be offloaded to the
 * TOPS backend based on the operation type and the size of the tensor. It
 * returns true if the second dimension (ne[1]) of the tensor is greater than or
 * equal to the minimum batch size and the operation is not GGML_OP_GET_ROWS.
 *
 * @param backend Pointer to the TOPS backend.
 * @param op Pointer to the tensor operation to check.
 * @return bool Returns true if the operation should be offloaded, otherwise
 * false.
 */
static bool ggml_backend_tops_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const int min_batch_size = 32;
    GGML_UNUSED(dev);

    return op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS;
}

/**
 * @brief Retrieves the name associated with a TOPS host buffer type.
 *
 * This function returns the descriptive name associated with the specified
 * TOPS host buffer type context.
 *
 * @param buft Pointer to the host buffer type context.
 * @return Const pointer to the C-style string containing the name.
 */
static const char * ggml_backend_tops_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return "TOPS_Host";

    GGML_UNUSED(buft);
}


/**
 * @brief Interface for managing CANN host buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
ggml_backend_buffer_type_t ggml_backend_tops_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cann_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ NULL,//ggml_backend_tops_host_buffer_type_name,//TODO need check
            /* .alloc_buffer     = */ NULL,//TODO need check
            /* .get_alignment    = */ NULL,//ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL,//ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ NULL,//ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ NULL,//ggml_backend_reg_dev_get(ggml_backend_tops_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cann_buffer_type_host;
}

static ggml_backend_buffer_type_t ggml_backend_tops_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_tops_host_buffer_type();
}

/**
 * @brief Creates a new event for the TOPS backend device.
 *
 * This function initializes a new event for the TOPS backend by setting the
 * device and creating an ACL runtime event. The created event is then wrapped
 * in a ggml_backend_event structure and returned.
 *
 * @param backend Pointer to the TOPS backend.
 * @return ggml_backend_event_t Returns a pointer to the new event structure.
 */
static ggml_backend_event_t ggml_backend_tops_device_event_new(
    ggml_backend_dev_t dev) {
    ggml_backend_tops_device_context * dev_ctx = (ggml_backend_tops_device_context *)dev->context;

    ggml_tops_set_device(dev_ctx->device);

    topsEvent_t event;
    TOPS_CHECK(topsEventCreate(&event));

    return new ggml_backend_event{
        /* .device = */ ggml_backend_reg_dev_get(ggml_backend_tops_reg(), dev_ctx->device),
        /* .context = */ event,
    };
}

/**
 * @brief Frees a TOPS backend event.
 *
 * This function destroys the ACL runtime event associated with the given TOPS
 * backend event and then deletes the event structure itself.
 *
 * @param event Pointer to the event structure to be freed.
 */
static void ggml_backend_tops_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    TOPS_CHECK(topsEventDestroy((topsEvent_t)(event->context)));

    delete event;
    GGML_UNUSED(dev);
}

/**
 * @brief Synchronizes the given event on the TOPS backend.
 *
 * This function waits for the specified event to complete on the ACL runtime.
 *
 * @param event Pointer to the event structure to be synchronized.
 */
static void ggml_backend_tops_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    //TODO
    GGML_UNUSED(dev);
    TOPS_CHECK(topsEventSynchronize((topsEvent_t)event->context));
}

static const ggml_backend_device_i ggml_backend_tops_device_interface = {
    /* .get_name                = */ ggml_backend_tops_device_get_name,
    /* .get_description         = */ ggml_backend_tops_device_get_description,
    /* .get_memory              = */ ggml_backend_tops_device_get_memory,
    /* .get_type                = */ ggml_backend_tops_device_get_type,
    /* .get_props               = */ ggml_backend_tops_device_get_props,
    /* .init_backend            = */ ggml_backend_tops_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_tops_device_get_buffer_type,
    /* .get_host_buffer_type    = */ NULL,//TODO
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_tops_supports_op,
    /* .supports_buft           = */ ggml_backend_tops_supports_buft,
    /* .offload_op              = */ ggml_backend_tops_device_offload_op,
    /* .event_new               = */ ggml_backend_tops_device_event_new,
    /* .event_free              = */ ggml_backend_tops_device_event_free,
    /* .event_synchronize       = */ ggml_backend_tops_device_event_synchronize,
};

// backend reg

struct ggml_backend_tops_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_tops_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_TOPS_NAME;
}

static size_t ggml_backend_tops_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_tops_reg_context * ctx = (ggml_backend_tops_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_tops_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_tops_reg_context * ctx = (ggml_backend_tops_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static void * ggml_backend_tops_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    //TODO get function addres according to the name
    // if (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
    //     return (void *)ggml_backend_cuda_split_buffer_type;
    // }
    // if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
    //     return (void *)ggml_backend_cuda_register_host_buffer;
    // }
    // if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
    //     return (void *)ggml_backend_cuda_unregister_host_buffer;
    // }
    // if (strcmp(name, "ggml_backend_get_features") == 0) {
    //     return (void *)ggml_backend_cuda_get_features;
    // }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_tops_reg_interface = {
    /* .get_name          = */ ggml_backend_tops_reg_get_name,
    /* .get_device_count  = */ ggml_backend_tops_reg_get_device_count,
    /* .get_device        = */ ggml_backend_tops_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_tops_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_tops_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_tops_reg_context * ctx = new ggml_backend_tops_reg_context;

            for (int i = 0; i < ggml_tops_info().device_count; i++) {
                ggml_backend_tops_device_context * dev_ctx = new ggml_backend_tops_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_TOPS_NAME + std::to_string(i);

                ggml_tops_set_device(i);
                topsDeviceProp_t prop;
                TOPS_CHECK(topsGetDeviceProperties(&prop, i));
                dev_ctx->description = prop.name;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_tops_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_tops_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_tops_reg)