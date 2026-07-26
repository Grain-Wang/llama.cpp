#include "ggml-tops_tensor.h"

topsatenDataType_t ggml_tops_type_mapping(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return TOPSATEN_DATA_FP32;
        case GGML_TYPE_F16:
            return TOPSATEN_DATA_FP16;
        case GGML_TYPE_BF16:
            return TOPSATEN_DATA_BF16;
        case GGML_TYPE_I8:
            return TOPSATEN_DATA_I8;
        case GGML_TYPE_I16:
            return TOPSATEN_DATA_I16;
        case GGML_TYPE_I32:
            return TOPSATEN_DATA_I32;
        case GGML_TYPE_I64:
            return TOPSATEN_DATA_I64;
        default:
            return TOPSATEN_DATA_NONE;
    }
    return TOPSATEN_DATA_NONE;
}

topsatenTensor* ggml_tops_create_tensor(const ggml_tensor* tensor, int64_t* ne ,
    size_t* nb , int64_t ndims, topsatenMemoryFormat_t format, topsatenDataType_t atens_type, size_t offset) {
    // If tensor is bcasted, Up to GGML_MAX_DIMS additional dimensions will be
    // added.
    int64_t dims[GGML_MAX_DIMS * 2], strides[GGML_MAX_DIMS * 2];
    int64_t s = 1;

    if (ne == nullptr) {
        for (int i = 0; i < GGML_MAX_DIMS; i++) { 
            dims[i] = tensor->ne[i];
            // The step size of tops is in elements(just in byte now).
            strides[i] = tensor->nb[i] / ggml_element_size(tensor); 
            // s *= dims[i];
            // strides[i] = s;
        }
    } else {
        // With bcast
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            dims[i] = ne[i];
            strides[i] = nb[i]/ggml_element_size(tensor);
        }
    }

    int64_t final_dims = (ndims == 0 ? GGML_MAX_DIMS : ndims);    
    
    // int64_t acl_storage_len = 1;
    // for (int i = 0; i < final_dims; i++) {
    //     acl_storage_len += (acl_ne[i] - 1) * acl_stride[i];
    // }

    // Reverse ne and stride.
    std::reverse(dims, dims + final_dims);
    std::reverse(strides, strides + final_dims);

    auto aten_dims = topsatenSize_t{dims, final_dims};
    auto aten_strides = topsatenSize_t{strides, final_dims};
    auto aten_dtype = atens_type == TOPSATEN_DATA_NONE ? ggml_tops_type_mapping(tensor->type) : atens_type;

    auto aten_tensor =
        new topsatenTensor(aten_dims, aten_strides, aten_dtype, tensor->data);

    return aten_tensor;
}

topsatenTensor* ggml_tops_create_tensor(void* buffer,int64_t* ne, size_t* nb, int64_t ndims,ggml_type org_type, ggml_type target_type) {
    int64_t dims[GGML_MAX_DIMS * 2], strides[GGML_MAX_DIMS * 2];

    for (int i = 0; i < GGML_MAX_DIMS; i++) { 
        dims[i] = ne[i];
        // The step size of tops is in elements(just in byte now).
        strides[i] = nb[i] / ggml_type_size(org_type); 
    }


    int64_t final_dims = (ndims == 0 ? GGML_MAX_DIMS : ndims);    
    

    std::reverse(dims, dims + final_dims);
    std::reverse(strides, strides + final_dims);

    auto aten_dims = topsatenSize_t{dims, final_dims};
    auto aten_strides = topsatenSize_t{strides, final_dims};
    auto aten_dtype =  ggml_tops_type_mapping(target_type);

    auto aten_tensor =
        new topsatenTensor(aten_dims, aten_strides, aten_dtype, buffer);

    return aten_tensor;
}

int64_t ggml_tops_get_bcast_shape(const ggml_tensor* src0,
    const ggml_tensor* src1,
    int64_t* bcast_src0_ne,
    int64_t* bcast_src1_ne, size_t* bcast_src0_nb,
    size_t* bcast_src1_nb) {
    GGML_ASSERT(ggml_can_repeat(src1, src0));
    int bcast_dim_cnt = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t nr = src0->ne[i] / src1->ne[i];
        bcast_src0_ne[bcast_dim_cnt] = src0->ne[i] / nr;
        bcast_src1_ne[bcast_dim_cnt] = src1->ne[i];
        bcast_src0_nb[bcast_dim_cnt] = src0->nb[i];
        bcast_src1_nb[bcast_dim_cnt] = src1->nb[i];
        bcast_dim_cnt++;
        if (nr != 1) {
            // Need to add an extra dim.
            bcast_src0_ne[bcast_dim_cnt] = nr;
            bcast_src1_ne[bcast_dim_cnt] = 1;
            bcast_src0_nb[bcast_dim_cnt] = bcast_src0_nb[bcast_dim_cnt - 1] *
                        bcast_src0_ne[bcast_dim_cnt - 1];
            bcast_src1_nb[bcast_dim_cnt] = bcast_src1_nb[bcast_dim_cnt - 1] *
                        bcast_src1_ne[bcast_dim_cnt - 1];
            bcast_dim_cnt++;
        }
    }
    return bcast_dim_cnt;
}

int64_t ggml_tops_get_mulmat_bcast_shape(
    const int64_t* input_ne, const int64_t* weight_ne, const int64_t* dst_ne,
    const size_t* input_nb, const size_t* weight_nb, const size_t* dst_nb,
    int64_t* bcast_input_ne, int64_t* bcast_weight_ne, int64_t* bcast_dst_ne,
    size_t* bcast_input_nb, size_t* bcast_weight_nb, size_t* bcast_dst_nb) {
    // input and dst shoule in same shape, except first two dims.
    GGML_ASSERT(input_ne[2] == dst_ne[2]);
    GGML_ASSERT(input_ne[3] == dst_ne[3]);

    int bcast_dim_cnt = 0;

    // For mul_mat, a dimension needs to be added before the dimension that
    // weight needs to be expanded to satisfy the bcast rule of matrix
    // multiplication.
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t nr = input_ne[i] / weight_ne[i];
        // Do not use bcast in the first two dimensions because we only support
        // the bcast batch dimension. Just copy them.
        if (i < 2 || nr == 1) {
            bcast_input_ne[bcast_dim_cnt] = input_ne[i];
            bcast_weight_ne[bcast_dim_cnt] = weight_ne[i];
            bcast_dst_ne[bcast_dim_cnt] = dst_ne[i];

            bcast_input_nb[bcast_dim_cnt] = input_nb[i];
            bcast_weight_nb[bcast_dim_cnt] = weight_nb[i];
            bcast_dst_nb[bcast_dim_cnt] = dst_nb[i];
            bcast_dim_cnt++;
        } else {
            // Need to add an extra dim.
            bcast_input_ne[bcast_dim_cnt] = nr;
            bcast_dst_ne[bcast_dim_cnt] = nr;
            bcast_weight_ne[bcast_dim_cnt] = 1;
            bcast_input_nb[bcast_dim_cnt] = input_nb[i];
            bcast_dst_nb[bcast_dim_cnt] = dst_nb[i];
            bcast_weight_nb[bcast_dim_cnt] = weight_nb[i];
            bcast_dim_cnt++;

            bcast_input_ne[bcast_dim_cnt] = input_ne[i] / nr;
            bcast_dst_ne[bcast_dim_cnt] = dst_ne[i] / nr;
            bcast_weight_ne[bcast_dim_cnt] = weight_ne[i];
            bcast_input_nb[bcast_dim_cnt] = bcast_input_nb[bcast_dim_cnt - 1] *
                                            bcast_input_ne[bcast_dim_cnt - 1];
            bcast_dst_nb[bcast_dim_cnt] = bcast_dst_nb[bcast_dim_cnt - 1] *
                                          bcast_dst_ne[bcast_dim_cnt - 1];
            bcast_weight_nb[bcast_dim_cnt] =
                bcast_weight_nb[bcast_dim_cnt - 1] *
                bcast_weight_ne[bcast_dim_cnt - 1];
            bcast_dim_cnt++;
        }
    }
    return bcast_dim_cnt;
}

static int atens_type_bytes(topsatenDataType_t data_type) {
    int nbytes = 0;
    switch (data_type)
    {
        case TOPSATEN_DATA_I8:
        case TOPSATEN_DATA_U8:
            nbytes = 1;
            break;
        case TOPSATEN_DATA_I16:
        case TOPSATEN_DATA_U16:
        case TOPSATEN_DATA_FP16:
        case TOPSATEN_DATA_BF16:
            nbytes = 2;
            break;
        case TOPSATEN_DATA_I32:
        case TOPSATEN_DATA_U32:
        case TOPSATEN_DATA_FP32:
        case TOPSATEN_DATA_EF32:
        case TOPSATEN_DATA_TF32:
            nbytes = 4;
            break;
        default:
            nbytes = 0;
    }

    return nbytes;
}

float fp16_to_float(uint16_t h) {
    uint16_t sign = (h >> 15) & 0x0001;
    uint16_t exp  = (h >> 10) & 0x001F;
    uint16_t mant = h & 0x03FF;

    if (exp == 0) {
        // Subnormal number
        if (mant == 0) return sign ? -0.0f : 0.0f;
        return (sign ? -1.0f : 1.0f) * mant * std::pow(2, -24);
    } else if (exp == 0x1F) {
        // Inf or NaN
        if (mant == 0) return sign ? -INFINITY : INFINITY;
        return NAN;
    } else {
        // Normalized number
        return (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * std::pow(2, exp - 15);
    }
}

void debug_tensor(ggml_backend_tops_context &ctx, topsatenTensor *tensor, topsStream_t stream) {
    if(stream != nullptr)
        topsStreamSynchronize(stream);
    int element_nbyte = atens_type_bytes(tensor->GetTensorDataType());
    topsatenSize_t tensor_shape = tensor->GetTensorShape();
    topsatenSize_t tensor_stride = tensor->GetTensorStrides();
    int rank = tensor_shape.len;

    topsatenSize_t shape = tensor->GetTensorShape();
        auto shape_len = static_cast<int>(shape.len);
        auto shape_size = 1;
        for (auto i=0; i<rank; ++i) {
            shape_size *= shape.data[i];
            // printf("shapedata %d %d\n", i, )
        }
    
    // int buf_nbyte = tensor_shape.data[rank-1]*tensor_stride.data[rank-1];
    int buf_nbyte = shape_size * element_nbyte;

    void *buf = (void *)malloc(buf_nbyte);

    // if (ctx.stream() != nullptr)
    // {
    //     TOPS_CHECK(topsMemcpyAsync(buf, tensor->GetTensorData(), buf_nbyte, topsMemcpyHostToDevice, ctx.stream()));
    // } else 
    {
        TOPS_CHECK(topsMemcpy(buf, tensor->GetTensorData(), buf_nbyte, topsMemcpyHostToDevice));
    }

    if(tensor->GetTensorDataType() == TOPSATEN_DATA_FP16)
    {
        uint16_t  *p_uint16_buf = (uint16_t  *)buf;
        printf("[addr:%x shape:", tensor->GetTensorData());
        for(auto i=0; i< shape.len; ++i) {
            printf("%lld", shape.data[i]);
            if(i!=shape.len-1) printf("x");
        }
        printf("]");
        printf("(len : %d)", shape_size);
        for (auto i = 0; i < std::min(shape_size, 30); ++i) {
        // for (auto i = 0; i < shape_len, 3; ++i) {
            printf("%f ", fp16_to_float(p_uint16_buf[i]));
        }
        printf("\n");
        // exit(0);
    } 
    else if (tensor->GetTensorDataType() == TOPSATEN_DATA_FP32)
    {
        float *p_i32_buf = (float *)buf;
        printf("[addr:%x shape:", tensor->GetTensorData());
        for(auto i=0; i< shape.len; ++i) {
            printf("%lld", shape.data[i]);
            if(i!=shape.len-1) printf("x");
        }
        printf("]");
        printf("(len : %d)", shape_size);
        for (auto i = 0; i < std::min(shape_size, 30); ++i) {
        // for (auto i = 0; i < shape_len, 3; ++i) {
            printf("%f ", p_i32_buf[i]);
        }
        printf("\n");
        // exit(0);
    }
    else if (tensor->GetTensorDataType() == TOPSATEN_DATA_F64)
    {
        double *p_double_buf = (double *)buf;
        printf("[addr:%x shape:%lldx%lldx%lldx%lld]\t%lf\t%lf\t%lf\n",
            tensor->GetTensorData(), shape.data[0], shape.data[1], shape.data[2], shape.data[3],
            p_double_buf[0], p_double_buf[1], p_double_buf[2]);
    }
    else if (tensor->GetTensorDataType() == TOPSATEN_DATA_I32)
    {
        int32_t *p_i32_buf = (int32_t *)buf;
        printf("[addr:%x shape:", tensor->GetTensorData());
        for(auto i=0; i< shape.len; ++i) {
            printf("%lld", shape.data[i]);
            if(i!=shape.len) printf("x");
        }
        printf("]");
        printf("(len : %d)", shape_size);
        for (auto i = 0; i < std::min(shape_size, 30); ++i) {
            printf("%d ", p_i32_buf[i]);
        }
        printf("\n");
    }
    else if (tensor->GetTensorDataType() == TOPSATEN_DATA_I64)
    {
        int64_t *p_i64_buf = (int64_t *)buf;
        printf("[addr:%x shape:", tensor->GetTensorData());
        for(auto i=0; i< shape.len; ++i) {
            printf("%lld", shape.data[i]);
            if(i!=shape.len) printf("x");
        }
        printf("]");
        printf("(len : %d)", shape_size);
        for (auto i = 0; i < std::min(shape_size, 30); ++i) {
            printf("%lld ", p_i64_buf[i]);
        }
        printf("\n");
    }
    else if (tensor->GetTensorDataType() == TOPSATEN_DATA_PRED)
    {
        uint8_t *p_pred = (uint8_t *)buf;
        printf("[addr:%x shape:%lldx%lldx%lldx%lld]\t%d\t%d\t%d\n",
            tensor->GetTensorData(), shape.data[0], shape.data[1], shape.data[2], shape.data[3], p_pred[0], p_pred[1], p_pred[2]);
    }
    else {
        float *p_float_buf = (float *)buf;
        printf("[addr:%x shape:%dx%dx%dx%d]\t%f\t%f\t%f\n",tensor->GetTensorData(),shape.data[0],shape.data[1],shape.data[2],shape.data[3],
            p_float_buf[0],p_float_buf[1],p_float_buf[2]);
    }

    UNUSED_PTR(buf)
    return ;

}