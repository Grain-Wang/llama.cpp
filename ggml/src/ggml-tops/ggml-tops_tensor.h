#pragma once
#include <topsaten/topsaten.h>
#include <topsaten/topsaten_define.h>
#include "common.h"
#include "ggml.h"

#include <algorithm>


topsatenDataType_t ggml_tops_type_mapping(ggml_type type);

topsatenTensor* ggml_tops_create_tensor(const ggml_tensor* tensor, int64_t* ne = nullptr,
    size_t* nb = nullptr, int64_t ndims = 0,
    topsatenMemoryFormat_t format = TOPSATEN_MEMORY_NONE, topsatenDataType_t atens_type = TOPSATEN_DATA_NONE,
    size_t offset = 0);

topsatenTensor* ggml_tops_create_tensor(void* buffer,int64_t* ne, size_t* nb, int64_t ndims, ggml_type org_type,ggml_type target_type);

int64_t ggml_tops_get_bcast_shape(const ggml_tensor* src0, const ggml_tensor* src1,
    int64_t* bcast_ne_src0, int64_t* bcast_ne_src1,
    size_t* bcast_nb_src0, size_t* bcast_nb_src1);

int64_t ggml_tops_get_mulmat_bcast_shape(
    const int64_t* input_ne, const int64_t* weight_ne, const int64_t* dst_ne,
    const size_t* input_nb, const size_t* weight_nb, const size_t* dst_nb,
    int64_t* bcast_input_ne, int64_t* bcast_weight_ne, int64_t* bcast_dst_ne,
    size_t* bcast_input_nb, size_t* bcast_weight_nb, size_t* bcast_dst_nb);

void debug_tensor(ggml_backend_tops_context &ctx, topsatenTensor *tensor, topsStream_t stream = nullptr);

// Bcast macro to avoid duplicate code.
#define BCAST_SHAPE(src0, src1)                                              \
    int64_t bcast_##src0##_ne[GGML_MAX_DIMS * 2];                            \
    int64_t bcast_##src1##_ne[GGML_MAX_DIMS * 2];                            \
    size_t bcast_##src0##_nb[GGML_MAX_DIMS * 2];                             \
    size_t bcast_##src1##_nb[GGML_MAX_DIMS * 2];                             \
    int64_t bcast_dims = ggml_tops_get_bcast_shape(                          \
        src0, src1, bcast_##src0##_ne, bcast_##src1##_ne, bcast_##src0##_nb, \
        bcast_##src1##_nb);


#define BCAST_MUL_MAT_SHAPE(input, weight, dst)                         \
    int64_t bcast_##input##_ne[GGML_MAX_DIMS * 2];                      \
    int64_t bcast_##weight##_ne[GGML_MAX_DIMS * 2];                     \
    int64_t bcast_##dst##_ne[GGML_MAX_DIMS * 2];                        \
    size_t bcast_##input##_nb[GGML_MAX_DIMS * 2];                       \
    size_t bcast_##weight##_nb[GGML_MAX_DIMS * 2];                      \
    size_t bcast_##dst##_nb[GGML_MAX_DIMS * 2];                         \
    int64_t bcast_dims = ggml_tops_get_mulmat_bcast_shape(              \
        input->ne, weight->ne, dst->ne, input->nb, weight->nb, dst->nb, \
        bcast_##input##_ne, bcast_##weight##_ne, bcast_##dst##_ne,      \
        bcast_##input##_nb, bcast_##weight##_nb, bcast_##dst##_nb);