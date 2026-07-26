#include "ggml-tops_ops.h"

static topsatenTensor* tops_values(ggml_backend_tops_context& ctx, void* buffer,
    size_t n_bytes, int64_t* ne, size_t* nb, int64_t ndims,
    topsatenDataType_t type, size_t type_size,
    float value = 1.0f) {
    topsatenTensor* tops_tensor = nullptr;
    
    int64_t buffer_size = ne[ndims-1]*nb[ndims-1];
    char *p_hostbuffer = (char *)malloc(buffer_size);

    if (TOPSATEN_DATA_FP32 == type)
    {
        float *p_floathostbuf = (float *)p_hostbuffer;
        for (size_t i = 0; i < buffer_size/type_size; i++)
        {
            p_floathostbuf[i] = value;
        }
        
    } else 
    {
        GGML_ABORT("unkonwn type!");
    }

    if (ctx.stream() != nullptr)
    {
        TOPS_CHECK(topsMemcpyAsync(buffer, p_hostbuffer, buffer_size, topsMemcpyHostToDevice, ctx.stream()));
    } else {
        TOPS_CHECK(topsMemcpy(buffer, p_hostbuffer, buffer_size, topsMemcpyHostToDevice));
    }
    
    int64_t dims[GGML_MAX_DIMS * 2], strides[GGML_MAX_DIMS * 2];

    // With bcast
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dims[i] = ne[i];
        strides[i] = nb[i];// ggml_element_size(tensor);
    }

    auto aten_dims = topsatenSize_t{dims, ndims};
    auto aten_strides = topsatenSize_t{strides, ndims};
    auto aten_dtype = type;
    auto aten_tensor =
        new topsatenTensor(aten_dims, aten_strides, aten_dtype, buffer);

    UNUSED_PTR(p_hostbuffer)
    return aten_tensor;
}

static topsatenTensor* tops_bias_values(ggml_backend_tops_context& ctx, void* buffer,
    size_t n_bytes, int64_t* ne, size_t* nb, int64_t ndims,
    topsatenDataType_t type, size_t type_size,
    float value = 0.0f) {
    topsatenTensor* tops_tensor = nullptr;
    
    int64_t buffer_size = ne[ndims-1]*nb[ndims-1];
    char *p_hostbuffer = (char *)malloc(buffer_size);

    if (TOPSATEN_DATA_FP32 == type)
    {
        float *p_floathostbuf = (float *)p_hostbuffer;
        for (size_t i = 0; i < buffer_size/type_size; i++)
        {
            p_floathostbuf[i] = value;
        }
        
    } else 
    {
        GGML_ABORT("unkonwn type!");
    }

    if (ctx.stream() != nullptr)
    {
        TOPS_CHECK(topsMemcpyAsync(buffer, p_hostbuffer, buffer_size, topsMemcpyHostToDevice, ctx.stream()));
    } else {
        TOPS_CHECK(topsMemcpy(buffer, p_hostbuffer, buffer_size, topsMemcpyHostToDevice));
    }
    
    int64_t dims[GGML_MAX_DIMS * 2], strides[GGML_MAX_DIMS * 2];

    // With bcast
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dims[i] = ne[i];
        strides[i] = nb[i];// ggml_element_size(tensor);
    }

    auto aten_dims = topsatenSize_t{dims, ndims};
    auto aten_strides = topsatenSize_t{strides, ndims};
    auto aten_dtype = type;
    auto aten_tensor =
        new topsatenTensor(aten_dims, aten_strides, aten_dtype, buffer);

    UNUSED_PTR(p_hostbuffer)
    return aten_tensor;
}

void ggml_tops_op_rms_norm(ggml_backend_tops_context & ctx, ggml_tensor * dst) {
    ggml_tensor* src = dst->src[0];

    int64_t ggml_shape[] = {src->ne[0], src->ne[1], src->ne[2], src->ne[3]};
    fprintf(stderr, "GGML shape:");
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        fprintf(stderr, "%lld%s", ggml_shape[i], (i == GGML_MAX_DIMS) ? "\n" : ", ");
    }

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    topsatenScalar_t epsilon;
    epsilon.dtype = TOPSATEN_DATA_FP32;
    epsilon.fval = eps;

    size_t one_tensor_n_bytes = src->ne[0] * ggml_element_size(src);
    ggml_tops_pool_alloc one_tensor_allocator(ctx.pool(), one_tensor_n_bytes);

    topsatenTensor* tops_gamma = tops_values(
        ctx, one_tensor_allocator.get(), one_tensor_n_bytes, src->ne, src->nb, 1,
        ggml_tops_type_mapping(src->type), ggml_element_size(src));

    auto tops_gamma_data = tops_gamma->GetTensorShape().data;
    fprintf(stderr, "TOPS tops_gamma shape:");
    for (int i = 0; i < tops_gamma->GetTensorShape().len; ++i) {
        fprintf(stderr, "%lld%s", tops_gamma_data[i], (i == GGML_MAX_DIMS) ? "\n" : ", ");
    }
    printf("\n");
    TOPS_ATEN_CHECK(topsvllm::topsvllmRmsNorm(*tops_dst, *tops_src, *tops_gamma, epsilon, stream));
    topsStreamSynchronize(stream);
    debug_tensor(ctx, tops_src);
    debug_tensor(ctx, tops_dst);
    debug_tensor(ctx, tops_gamma);
    //TODO need manage tensor resource

    delete tops_src;
    delete tops_dst;
    delete tops_gamma;

    return;
}

bool ggml_tops_need_bcast(const ggml_tensor* t0, const ggml_tensor* t1) {
    printf("ggml_tops_need_bcast");
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (t1->ne[i] != t0->ne[i] && t1->ne[i] != 1) {
            return true;
        }
    }
    return false;
}

void bcast_shape(ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst, topsatenTensor ** tops_src0,
    topsatenTensor ** tops_src1, topsatenTensor ** tops_dst) {
    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_can_repeat(src1, src0));
    // Need bcast
    if (!ggml_are_same_shape(src0, src1) && ggml_tops_need_bcast(src0, src1)) {
        BCAST_SHAPE(src0, src1)
        // *tops_src0 = ggml_tops_create_tensor(src0, BCAST_PARAM(src0));
        // *tops_src1 = ggml_tops_create_tensor(src1, BCAST_PARAM(src1));
        // *tops_dst  = ggml_tops_create_tensor(dst, BCAST_PARAM(src0));
    } else {
        *tops_src0 = ggml_tops_create_tensor(src0);
        *tops_src1 = ggml_tops_create_tensor(src1);
        *tops_dst  = ggml_tops_create_tensor(dst);
    }
}

void ggml_tops_op_mul(ggml_backend_tops_context & ctx, ggml_tensor * dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    topsStream_t stream = ctx.stream();

    topsatenTensor* tops_src0 = ggml_tops_create_tensor(src0);
    topsatenTensor* tops_src1 = ggml_tops_create_tensor(src1);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);

    bcast_shape(src0, src1, dst, &tops_src0, &tops_src1, &tops_dst);
    debug_tensor(ctx, tops_src0);
    debug_tensor(ctx, tops_src1);
    TOPS_ATEN_CHECK(topsaten::topsatenMul(*tops_dst,*tops_src0,*tops_src1,stream));
    debug_tensor(ctx, tops_dst);

    //TODO need manage tensor resource
    delete tops_src0;
    delete tops_src1;
    delete tops_dst;

    return;
}

/**
 * @brief Matmul operator.
 *
 * @param out: The output tensor.
 * @param lhs: The input tensor of lhs.
 * @param rhs: The input tensor of rhs.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenMatmul(topsatenTensor &out,
//     const topsatenTensor &lhs,
//     const topsatenTensor &rhs,
//     topsStream_t stream = nullptr);
void ggml_tops_op_mul_mat(ggml_backend_tops_context & ctx, ggml_tensor * dst) {
    ggml_tensor* weight = dst->src[0];
    ggml_tensor* input = dst->src[1];
    topsStream_t stream = ctx.stream();

    if(dst->src[0]->ne[2]!= 1) {
        dst->src[1]->ne[1] = dst->src[1]->ne[2]/dst->src[0]->ne[2];
        dst->src[1]->ne[2] = dst->src[0]->ne[2];
    }

    topsatenTensor* tops_input = ggml_tops_create_tensor(input);
    topsatenTensor* tops_weight = ggml_tops_create_tensor(weight);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    //restore

    auto tops_input_data = tops_input->GetTensorShape().data;
    size_t one_tensor_n_bytes = weight->ne[0]*weight->ne[1]*weight->ne[2]*weight->ne[3]*ggml_type_size(GGML_TYPE_F32);
    ggml_tops_pool_alloc one_tensor_allocator(ctx.pool(), one_tensor_n_bytes);
    auto dims = weight->ne;
    int64_t transpose_ne[] = {weight->ne[1], weight->ne[0], weight->ne[2], weight->ne[3], weight->ne[4]};
    size_t transpose_nb[] = {weight->nb[1], weight->nb[0], weight->nb[2], weight->nb[3], weight->nb[4]};
    topsatenTensor* tops_weight_converted = ggml_tops_create_tensor(one_tensor_allocator.get(),transpose_ne, transpose_nb, 4,weight->type, GGML_TYPE_F32);

    topsatenDataType_t converted_dtype = TOPSATEN_DATA_FP32;
    auto tops_weight_convert_data = tops_weight_converted->GetTensorShape().data;
    auto tops_weight_data = tops_weight->GetTensorShape().data;
    int64_t tops_ndims = tops_weight->GetTensorShape().len;

    auto len = tops_weight_converted->GetTensorShape().len;
    bool issame = true;
    for(int i = 0;i<len;i++)
    {
        if(tops_weight_convert_data[i]!=tops_weight_data[i])
        {
            issame = false;
        }
    }
    if(issame){
        TOPS_ATEN_CHECK(topsaten::topsatenTo(*tops_weight_converted, *tops_weight, converted_dtype,false,true,TOPSATEN_MEMORY_PRESERVE, stream));
    }
    else
    {
        TOPS_ATEN_CHECK(topsaten::topsatenTranspose(*tops_weight,*tops_weight,len-1,len-2,stream));
        TOPS_ATEN_CHECK(topsaten::topsatenTo(*tops_weight_converted, *tops_weight, converted_dtype,false,true,TOPSATEN_MEMORY_PRESERVE, stream));
    }

    debug_tensor(ctx, tops_input);
    debug_tensor(ctx, tops_weight);
    debug_tensor(ctx, tops_weight_converted);
    if(dst->src[0]->ne[2]!= 1) {
        dst->ne[1] = dst->src[0]->ne[1];
        dst->ne[2] = dst->src[0]->ne[2];
        dst->ne[0] = dst->src[1]->ne[1];
        topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
        TOPS_ATEN_CHECK(topsaten::topsatenTranspose(*tops_dst,*tops_dst,len-1,len-2,stream));
        TOPS_ATEN_CHECK(topsaten::topsatenMatmul(*tops_dst,*tops_input,*tops_weight_converted,stream));
        debug_tensor(ctx, tops_dst);
        dst->ne[2] = dst->ne[0]*dst->ne[2];
        dst->ne[0] = dst->ne[3];
        topsatenTensor* tops_dst_re = ggml_tops_create_tensor(dst);
        TOPS_ATEN_CHECK(topsaten::topsatenTranspose(*tops_dst_re,*tops_dst_re,len-1,len-2,stream));
        debug_tensor(ctx, tops_dst_re);
        delete tops_dst_re;
    }
    else{
        TOPS_ATEN_CHECK(topsaten::topsatenMatmul(*tops_dst,*tops_input,*tops_weight_converted,stream));
        debug_tensor(ctx, tops_dst);
    }

    delete tops_input;
    delete tops_weight;
    delete tops_dst;
    delete tops_weight_converted;

    return;

}

/**
 * @brief Rotary_embedding.
 *
 * @param query: The output/input tensor [num_tokens, num_heads * head_size]
 * float32 float16.
 * @param key: The output/input tensor [num_tokens, num_kv_heads * head_size]
 * float32 float16.
 * @param positions: Input tensor [num_tokens] int64.
 * @param cos_sin_cache: Input tensor [max_position, rot_dim] float32 float16.
 * @param head_size: Input scalar.
 * @param is_neox: Input scalar.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsvllmRotaryEmbedding(
//     topsatenTensor& query, topsatenTensor& key, const topsatenTensor& positions,
//     const topsatenTensor& cos_sin_cache, int head_size, bool is_neox,
//     topsStream_t stream = nullptr);
void ggml_tops_op_rope(ggml_backend_tops_context & ctx, ggml_tensor * dst) {
    // ggml_tensor* src_key = dst->src[0]; //f32
    // ggml_tensor* src_positions = dst->src[1]; //i32
    // topsStream_t stream = ctx.stream();

    // topsatenTensor* tops_key = ggml_tops_create_tensor(src_key);
    // topsatenTensor* tops_positions = ggml_tops_create_tensor(src_positions);
    // topsatenTensor* tops_query = ggml_tops_create_tensor(dst);

    // int n_dims = static_cast<int>(dst->op_params[1]);
    // int max_position = static_cast<int>(dst->op_params[5]);
    // float base = dst->op_params[7] ? dst->op_params[7] : 10000.0f; // 基数，默认 10000

    // int rot_dim = n_dims / 2;

    // topsatenTensor* tops_cos_sin_cache;
    // // topsatenTensor* tops_cos_sin_cache = get_or_create_rope_cache(max_position, rot_dim, base, stream);

    // int head_size = n_dims; 
    // bool is_neox = (dst->op_params[2] == GGML_ROPE_TYPE_NEOX);

    // debug_tensor(ctx, tops_query);
    // debug_tensor(ctx, tops_key);
    // debug_tensor(ctx, tops_positions);
    // debug_tensor(ctx, tops_cos_sin_cache);
    // TOPS_ATEN_CHECK(topsvllm::topsvllmRotaryEmbedding(*tops_query,*tops_key,*tops_positions,*tops_cos_sin_cache,head_size,is_neox,stream));
    // debug_tensor(ctx, tops_query);

    // delete tops_query;
    // delete tops_key;
    // delete tops_positions;
    // delete tops_cos_sin_cache;
    // delete tops_query;

    // return;
}

void ggml_tops_op_cpy(ggml_backend_tops_context & ctx, ggml_tensor * dst) {
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    bool non_blocking = false;
    if (dst->op_params != nullptr && dst->op_params[0] != 0) {
        non_blocking = static_cast<bool>(dst->op_params[0]);
    }

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenCopy(*tops_dst,*tops_src,non_blocking,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;
}

void ggml_tops_op_softmax(ggml_backend_tops_context & ctx, ggml_tensor * dst) {
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    int64_t softmax_dim = -1;
    if (dst->op_params[1] != 0) {
        softmax_dim = static_cast<int64_t>(dst->op_params[1]);
    }

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenSoftmaxForward(*tops_dst,*tops_src,softmax_dim,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;
}

void ggml_tops_op_dup(ggml_backend_tops_context & ctx, ggml_tensor * dst) {
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    int32_t format_idx = dst->op_params[0];
    topsatenMemoryFormat_t format = static_cast<topsatenMemoryFormat_t>(format_idx);

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenClone(*tops_dst,*tops_src,format,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;
}

void ggml_tops_op_transpose(ggml_backend_tops_context & ctx, ggml_tensor * dst) {
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    int64_t dim0 = -1;
    int64_t dim1 = -1;

    if (dst->op_params[0] != 0) {
        dim0 = static_cast<int64_t>(dst->op_params[0]);
    }
    if (dst->op_params[1] != 0) {
        dim1 = static_cast<int64_t>(dst->op_params[1]);
    }

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenTranspose(*tops_dst,*tops_src,dim0,dim1,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;
}

void ggml_tops_op_reshape(ggml_backend_tops_context & ctx, ggml_tensor * dst) {

    // NOP
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);

}

void ggml_tops_op_view(ggml_backend_tops_context & ctx, ggml_tensor * dst) {

    // NOP
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);

}

void ggml_compute_forward_permute(ggml_backend_tops_context & ctx, ggml_tensor * dst) {

    // NOP
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);

}

void ggml_tops_op_glu(ggml_backend_tops_context & ctx, ggml_tensor * dst) {
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();
    int32_t dim = -1;

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenGlu(*tops_dst,*tops_src, dim,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;
}

void ggml_tops_op_repeat(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();
    topsatenSize_t repeats;

    auto tops_src_data = tops_src->GetTensorShape().data;
    auto tops_output_data = tops_dst->GetTensorShape().data;

    repeats.len = tops_src->GetTensorShape().len;
    int64_t* repeats_temp = new int64_t[repeats.len];
    for (int i = 0; i < repeats.len; ++i) {
        if (1 + i < GGML_MAX_OP_PARAMS) {
            repeats_temp[i] = tops_output_data[i]/tops_src_data[i];
        }
    }
    repeats.data = repeats_temp;

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenRepeat(*tops_dst,*tops_src, repeats,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;
}


void ggml_tops_op_add(ggml_backend_tops_context & ctx, ggml_tensor * dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    topsStream_t stream = ctx.stream();

    topsatenScalar_t alpha;
    alpha.dtype = TOPSATEN_DATA_FP32;
    alpha.fval = 1.0;
    // topsatenScalar_t alpha;
    // alpha.dtype = TOPSATEN_DATA_FP32;
    // if (dst->op_params[0] != 0) {
    //     float input_val;
    //     memcpy(&input_val, &dst->op_params[2], sizeof(float));
    //     alpha.fval = input_val;
    // } else {
    //     input_scale.fval = 1.0;
    // }

    topsatenTensor* tops_src0 = ggml_tops_create_tensor(src0);
    topsatenTensor* tops_src1= ggml_tops_create_tensor(src1);
    topsatenTensor* tops_dst= ggml_tops_create_tensor(dst);

    auto tops_lhs_data = tops_src0->GetTensorShape().data;
    auto tops_rhs_data = tops_src1->GetTensorShape().data;
    auto tops_output_data = tops_dst->GetTensorShape().data;
    auto tops_lhs_nb = tops_src0->GetTensorStrides().data;
    auto tops_rhs_nb = tops_src1->GetTensorStrides().data;
    auto tops_output_nb = tops_dst->GetTensorStrides().data;

    fprintf(stderr, "TOPS lhs shape:");
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        fprintf(stderr, "%lld%s", tops_lhs_data[i], (i == GGML_MAX_DIMS) ? "\n" : ", ");
    }

    fprintf(stderr, "TOPS rhs shape:");
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        fprintf(stderr, "%lld%s", tops_rhs_data[i], (i == GGML_MAX_DIMS) ? "\n" : ", ");
    }

    fprintf(stderr, "TOPS output shape:");
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        fprintf(stderr, "%lld%s", tops_output_data[i], (i == GGML_MAX_DIMS) ? "\n" : ", ");
    }
    printf("\n");

    fprintf(stderr, "TOPS lhs nb:");
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        fprintf(stderr, "%lld%s", tops_lhs_data[i], (i == GGML_MAX_DIMS) ? "\n" : ", ");
    }

    fprintf(stderr, "TOPS rhs shape:");
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        fprintf(stderr, "%lld%s", tops_rhs_data[i], (i == GGML_MAX_DIMS) ? "\n" : ", ");
    }

    fprintf(stderr, "TOPS output nb:");
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        fprintf(stderr, "%lld%s", tops_output_nb[i], (i == GGML_MAX_DIMS) ? "\n" : ", ");
    }

    bcast_shape(src0, src1, dst, &tops_src0, &tops_src1, &tops_dst);
    debug_tensor(ctx, tops_src0);
    debug_tensor(ctx, tops_src1);
    TOPS_ATEN_CHECK(topsaten::topsatenAdd(*tops_dst,*tops_src0,*tops_src1,alpha,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src0;
    delete tops_src1;
    delete tops_dst;

    return;
}

void ggml_tops_op_sub(ggml_backend_tops_context & ctx, ggml_tensor * dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    topsStream_t stream = ctx.stream();

    topsatenScalar_t alpha;
    alpha.dtype = TOPSATEN_DATA_FP32;
    alpha.fval = 1.0;

    topsatenTensor* tops_src0 = ggml_tops_create_tensor(src0);
    topsatenTensor* tops_src1= ggml_tops_create_tensor(src1);
    topsatenTensor* tops_dst= ggml_tops_create_tensor(dst);

    bcast_shape(src0, src1, dst, &tops_src0, &tops_src1, &tops_dst);
    debug_tensor(ctx, tops_src0);
    debug_tensor(ctx, tops_src1);
    TOPS_ATEN_CHECK(topsaten::topsatenSub(*tops_dst,*tops_src0,*tops_src1,alpha,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src0;
    delete tops_src1;
    delete tops_dst;

    return;
}

void ggml_tops_op_acc(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src0 = dst->src[0];
    topsStream_t stream = ctx.stream();

    topsatenTensor* tops_src0 = ggml_tops_create_tensor(src0);
    topsatenTensor* tops_dst= ggml_tops_create_tensor(dst);

    topsatenScalar_t rhs;
    rhs.dtype = TOPSATEN_DATA_FP32;
    float rhs_val;
    memcpy(&rhs_val, &dst->op_params[1], sizeof(float));
    rhs.fval = rhs_val;

    topsatenScalar_t alpha;
    alpha.dtype = TOPSATEN_DATA_FP32;
    alpha.fval = 1.0;

    debug_tensor(ctx, tops_src0);
    TOPS_ATEN_CHECK(topsaten::topsatenAdd(*tops_dst,*tops_src0,rhs,alpha,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src0;
    delete tops_dst;
    
    return;
}

void ggml_tops_op_div(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    topsStream_t stream = ctx.stream();

    topsatenTensor* tops_src0 = ggml_tops_create_tensor(src0);
    topsatenTensor* tops_src1= ggml_tops_create_tensor(src1);
    topsatenTensor* tops_dst= ggml_tops_create_tensor(dst);

    bcast_shape(src0, src1, dst, &tops_src0, &tops_src1, &tops_dst);
    debug_tensor(ctx, tops_src0);
    debug_tensor(ctx, tops_src1);
    TOPS_ATEN_CHECK(topsaten::topsatenDiv(*tops_dst,*tops_src0,*tops_src1,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src0;
    delete tops_src1;
    delete tops_dst;

}

void ggml_tops_op_abs(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenAbs(*tops_dst,*tops_src, stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;

}

void ggml_tops_op_neg(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenNeg(*tops_dst,*tops_src, stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;

}

void ggml_tops_op_gelu(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();
    const char* const approximate = "none";

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenGelu(*tops_dst,*tops_src, approximate, stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;

}

void ggml_tops_op_silu(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenSilu(*tops_dst,*tops_src,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;

}

void ggml_tops_op_gelu_quick(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsvllm::topsvllmGeluQuick(*tops_dst,*tops_src,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;

}

void ggml_tops_op_tanh(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenTanh(*tops_dst,*tops_src,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;

}

void ggml_tops_op_relu(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenRelu(*tops_dst,*tops_src,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;

}

void ggml_tops_op_sigmoid(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenSigmoid(*tops_dst,*tops_src,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;

}

void ggml_tops_op_hardsigmoid(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenHardsigmoid(*tops_dst,*tops_src,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;

}

void ggml_tops_op_hardswish(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenHardswish(*tops_dst,*tops_src,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;

}

void ggml_tops_op_exp(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenExp(*tops_dst,*tops_src,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;

}

void ggml_tops_op_elu(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    topsatenScalar_t alpha;
    alpha.dtype = TOPSATEN_DATA_FP32;
    alpha.fval = 1.0;

    topsatenScalar_t scale;
    scale.dtype = TOPSATEN_DATA_FP32;
    scale.fval = 1.0;

    topsatenScalar_t input_scale;
    input_scale.dtype = TOPSATEN_DATA_FP32;
    input_scale.fval = 1.0;

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenElu(*tops_dst,*tops_src,alpha,scale,input_scale,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;

}

void ggml_tops_op_sgn(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src0 = dst->src[0];
    
    topsatenTensor* tops_src = ggml_tops_create_tensor(src0);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenSign(*tops_dst,*tops_src,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;
}

void ggml_tops_op_norm(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_input = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    topsatenSize_t normalized_shape;
    auto input_shape = tops_input->GetTensorShape();
    int64_t norm_dims[1];
    norm_dims[0] = input_shape.data[3];
    normalized_shape.len = 1;
    normalized_shape.data = norm_dims;

    topsatenScalar_t epsilon;
    epsilon.dtype = TOPSATEN_DATA_FP32;
    float eps = 1e-6f;
    epsilon.fval = eps;

    size_t weight_n_bytes = src->ne[0] * ggml_element_size(src);
    ggml_tops_pool_alloc weight_allocator(ctx.pool(), weight_n_bytes);

    topsatenTensor* tops_weight = tops_values(
        ctx, weight_allocator.get(), weight_n_bytes, src->ne, src->nb, 1,
        ggml_tops_type_mapping(src->type), ggml_element_size(src), 1.0f);

    size_t bais_n_bytes = src->ne[0] * ggml_element_size(src);
    ggml_tops_pool_alloc bais_allocator(ctx.pool(), bais_n_bytes);

    topsatenTensor* tops_bias = tops_bias_values(
        ctx, bais_allocator.get(), bais_n_bytes, src->ne, src->nb, 1,
        ggml_tops_type_mapping(src->type), ggml_element_size(src), 0.0f);

    debug_tensor(ctx, tops_input);
    debug_tensor(ctx, tops_weight);
    debug_tensor(ctx, tops_bias);
    TOPS_ATEN_CHECK(topsaten::topsatenLayerNorm(*tops_dst,*tops_input,normalized_shape,*tops_weight,*tops_bias, epsilon,stream));
    topsStreamSynchronize(stream);
    debug_tensor(ctx, tops_dst);

    delete tops_input;
    delete tops_weight;
    delete tops_bias;
    delete tops_dst;

    return;
}

void ggml_tops_op_groupnorm(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_input = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    int64_t group;
    group = static_cast<int64_t>(dst->op_params[0]);

    double eps;
    topsatenScalar_t negative_slope;
    float eps_val;
    memcpy(&eps_val, &dst->op_params[1], sizeof(float));
    eps = 1e-6f;

    int64_t weight_ne[] = {src->ne[2], src->ne[1],
        src->ne[0], src->ne[3]};
    size_t weight_nb[] = {src->nb[2], src->nb[1],
        src->nb[0], src->nb[3]};

    size_t weight_n_bytes = src->ne[0]*src->ne[1]*src->ne[2]*src->ne[3] * ggml_element_size(src);
    ggml_tops_pool_alloc weight_allocator(ctx.pool(), weight_n_bytes);

    topsatenTensor* tops_weight = tops_values(
        ctx, weight_allocator.get(), weight_n_bytes, weight_ne, weight_nb, 1,
        ggml_tops_type_mapping(src->type), ggml_element_size(src));

    size_t bais_n_bytes = src->ne[0]*src->ne[1]*src->ne[2]*src->ne[3] * ggml_element_size(src);
    ggml_tops_pool_alloc bais_allocator(ctx.pool(), bais_n_bytes);

    topsatenTensor* tops_bias = tops_values(
        ctx, bais_allocator.get(), bais_n_bytes, weight_ne, weight_nb, 1,
        ggml_tops_type_mapping(src->type), ggml_element_size(src), 0.0f);

    auto weight_rank = tops_weight->GetTensorShape().len;
    auto weight_dim0 = tops_weight->GetTensorShape().data[0];
    auto bias_rank = tops_bias->GetTensorShape().len;
    auto biad_dim0 = tops_bias->GetTensorShape().data[0];

    debug_tensor(ctx, tops_input);
    debug_tensor(ctx, tops_weight);
    debug_tensor(ctx, tops_bias);
    TOPS_ATEN_CHECK(topsaten::topsatenGroupNorm(*tops_dst,*tops_input,group,*tops_weight,*tops_bias, eps,stream))
    debug_tensor(ctx, tops_dst);

    delete tops_input;
    delete tops_weight;
    delete tops_bias;
    delete tops_dst;

    return;
}

void ggml_tops_op_concat(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src_0 = dst->src[0];
    ggml_tensor* src_1 = dst->src[1];

    topsatenTensor* tops_src0 = ggml_tops_create_tensor(src_0);
    topsatenTensor* tops_src1 = ggml_tops_create_tensor(src_1);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    std::vector<topsatenTensor> aten_inputs;
    aten_inputs.push_back(*tops_src0);
    aten_inputs.push_back(*tops_src1);

    int64_t dim = static_cast<int64_t>(dst->op_params[0]);

    int64_t topsaten_dim = -1;
    topsaten_dim = 3 - dim;

    for (auto& input : aten_inputs) {
        debug_tensor(ctx, &input); 
    }

    TOPS_ATEN_CHECK(topsaten::topsatenCat(*tops_dst, aten_inputs, topsaten_dim, stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src0;
    delete tops_src1;
    delete tops_dst;

    return;
}

void ggml_tops_op_upscale(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src_input = dst->src[0];

    topsatenTensor* tops_input = ggml_tops_create_tensor(src_input);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    int32_t scale_factor = 2;

    int64_t in_h = src_input->ne[0];
    int64_t in_w = src_input->ne[1];
    int64_t out_h = static_cast<int64_t>(in_h * scale_factor);
    int64_t out_w = static_cast<int64_t>(in_w * scale_factor);
    int64_t output_size[2] = {out_h,out_w};
    topsatenSize_t tops_output_size(output_size, 2);

    topsatenScalar_t tops_scales_h;
    tops_scales_h.dtype = TOPSATEN_DATA_FP32;
    tops_scales_h.fval = static_cast<float>(scale_factor);

    topsatenScalar_t tops_scales_w;
    tops_scales_w.dtype = TOPSATEN_DATA_FP32;
    tops_scales_w.fval = static_cast<float>(scale_factor);

    bool align_corners = 0;

    debug_tensor(ctx, tops_input);
    if (dst->op_params[0] != 1){
        TOPS_ATEN_CHECK(topsaten::topsatenUpsampleNearest2d(*tops_dst,*tops_input,tops_output_size,tops_scales_h,tops_scales_w,stream))
    } else {
        TOPS_ATEN_CHECK(topsaten::topsatenUpsampleBilinear2d(*tops_dst,*tops_input,tops_output_size,align_corners,tops_scales_h,tops_scales_w,stream))
    }

    debug_tensor(ctx, tops_dst);

    delete tops_input;
    delete tops_dst;

    return;
}

void ggml_tops_op_pad(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src_input = dst->src[0];

    topsatenTensor* tops_input = ggml_tops_create_tensor(src_input);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    int pad0 = 1, pad1 = 1;
    // pad0 = static_cast<int>(dst->op_params[0]);
    // pad1 = static_cast<int>(dst->op_params[1]);

    int64_t pad_count = 8;
    int64_t* pad_data = new int64_t[pad_count]{
        0, pad0, 0, pad1,
        0, 0, 0, 0
    };

    topsatenSize_t pads;
    pads.len = pad_count;
    pads.data = pad_data;

    topsatenPadMode_t mode = TOPSATEN_PAD_CONSTANT;
    topsatenScalar_t value;
    value.dtype = TOPSATEN_DATA_FP32;
    value.fval = 0.0f;
    
    debug_tensor(ctx, tops_input);
    TOPS_ATEN_CHECK(topsaten::topsatenPad(*tops_dst,*tops_input,pads,mode,value,stream))
    debug_tensor(ctx, tops_dst);

    delete tops_input;
    delete tops_dst;

    return;
}

void ggml_tops_op_arange(ggml_backend_tops_context & ctx, ggml_tensor * dst){

    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    topsatenScalar_t end;
    end.dtype = TOPSATEN_DATA_FP32;
    float end_float = *reinterpret_cast<const float*>(&dst->op_params[1]);;
    end.fval = end_float;

    auto type = tops_dst->GetTensorDataType();
    topsatenDataType_t dtype = type;
    const topsatenLayoutType_t layout = TOPSATEN_LAYOUT_STRIDED;
    bool pinMemory = false;

    TOPS_ATEN_CHECK(topsaten::topsatenArange(*tops_dst,end,dtype,layout,pinMemory,stream))
    debug_tensor(ctx, tops_dst);

    delete tops_dst;

    return;
}

void ggml_tops_op_leaky_relu(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src_input = dst->src[0];

    topsatenTensor* tops_input = ggml_tops_create_tensor(src_input);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    topsatenScalar_t negative_slope;
    negative_slope.dtype = TOPSATEN_DATA_FP32;
    float slope_val;
    memcpy(&slope_val, &dst->op_params[0], sizeof(float));
    negative_slope.fval = slope_val;

    debug_tensor(ctx, tops_input);
    TOPS_ATEN_CHECK(topsaten::topsatenLeakyRelu(*tops_dst,*tops_input,negative_slope,stream))
    debug_tensor(ctx, tops_dst);

    delete tops_input;
    delete tops_dst;

    return;
}

/**
 * @brief Invoke fused moe kernel.
 *
 * @param C: The output cache tensor with shape (M, topk, N), where M is the
 *          total number of tokens post padding, topk is the number of times
 *          each token is repeated, and N is the output feature dimension.
 *          Type Constraints: should be same as A.
 * @param A: The input tensor representing tokens with shape (*, K), where '*'
 *          can be any shape representing batches and K is the feature
 *          dimension of each token.
 *          Type Constraints: TOPSATEN_DATA_FP16/TOPSATEN_DATA_BF16.
 * @param B: The stacked MOE weight tensor with shape (E, N, K), where E is the
 *          number of experts, K is the input feature dimension, and N is the
 *          output feature dimension.
 *          Type Constraints: should be same as A.
 * @param topk_weights: The top-k expert weights for each token.
 *                      Type Constraints: TOPSATEN_DATA_FP32
 * @param topk_ids: The top-k expert indices for each token.
 *                  Type Constraints: TOPSATEN_DATA_I32
 * @param sorted_token_ids: Sorted token indices according to their allocated
 *                         expert.
 *                          Type Constraints: TOPSATEN_DATA_I32
 * @param experts_ids: The assigned expert index for each block.
 *                     Type Constraints: TOPSATEN_DATA_I32
 * @param num_tokens_post_pad: Shape: [1]. Number of tokens after padding.
 *                             Type Constraints: TOPSATEN_DATA_I32
 * @param mul_routed_weight: The flag used to indicate whether topk_weights
 * participates in the calculation.
 * @param topk: Number of experts for each token.
 * @param block_size: Block size.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsvllmInvokeFusedMoeKernel(
//     topsatenTensor& C, const topsatenTensor& A, const topsatenTensor& B,
//     const topsatenTensor& topk_weights, const topsatenTensor& topk_ids,
//     const topsatenTensor& sorted_token_ids, const topsatenTensor& experts_ids,
//     const topsatenTensor& num_tokens_post_pad, bool mul_routed_weight, int topk,
//     int block_size, topsStream_t stream = nullptr);
void ggml_tops_op_mul_mat_id(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src0 = dst -> src[0];// AS F16
    ggml_tensor* src1 = dst -> src[1];// B F32
    ggml_tensor* src2 = dst -> src[2];// IDS I32

    topsatenTensor* tops_A = ggml_tops_create_tensor(src0);
    topsatenTensor* tops_B = ggml_tops_create_tensor(src1);
    topsatenTensor* tops_ids = ggml_tops_create_tensor(src2);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    const int topk = dst->ne[1];
    const int block_size = 128;
    const bool mul_routed_weight = false;
    int64_t M = src0->ne[0] * src0->ne[1] * src0->ne[2];

    size_t sorted_mem_size = M * sizeof(int32_t);
    // topsatenTensor* ggml_tops_create_tensor(const ggml_tensor* tensor, int64_t* ne = nullptr,
    //     size_t* nb = nullptr, int64_t ndims = 0,
    //     topsatenMemoryFormat_t format = TOPSATEN_MEMORY_NONE, topsatenDataType_t atens_type = TOPSATEN_DATA_NONE,
    //     size_t offset = 0);
    
    // topsatenTensor* ggml_tops_create_tensor(void* buffer,int64_t* ne, size_t* nb, int64_t ndims, ggml_type org_type,ggml_type target_type);





    size_t weight_n_bytes = src2->ne[0] * src2->ne[1]*src2->ne[2] * src2->ne[3]* ggml_element_size(src1);
    int64_t tw_dims[] = {src2->ne[0] * src2->ne[1]};
    int64_t num_tokens = src2->ne[0] * src2->ne[1];
    ggml_tops_pool_alloc weight_allocator(ctx.pool(), weight_n_bytes);
    topsatenTensor* tops_weight = tops_values(ctx, weight_allocator.get(), weight_n_bytes, src1->ne, src1->nb, 1,
        ggml_tops_type_mapping(GGML_TYPE_F32), ggml_element_size(src1), 1.0f);
        topsatenTensor* tops_values(ggml_backend_tops_context& ctx, void* buffer,
            size_t n_bytes, int64_t* ne, size_t* nb, int64_t ndims,
            topsatenDataType_t type, size_t type_size,
            float value = 1.0f);

    // int64_t sorted_ne[] = {src0->M, src0->ne[0], src0->ne[2], src0->ne[3]};
    // size_t sorted_nb[] = {src0->nb[1], src0->nb[0], src0->nb[2], src0->nb[3], src0->nb[4]};
    topsatenTensor* tops_sorted_token_ids;
    ggml_tops_pool_alloc sti_allocator(ctx.pool(), weight_n_bytes);
    // topsatenTensor* sorted_token_ids = ggml_tops_create_tensor(sti_allocator.get(), src1->ne, src1->nb, 1,
    //     ggml_tops_type_mapping(GGML_TYPE_F32), ggml_element_size(src2), 0.0f);

    // int* sti_data = (int*)tops_sorted_token_ids->GetTensorData();
    // for (int i = 0; i < num_tokens; i++) {
    //     sti_data[i] = i;
    // }

    int num_blocks = (num_tokens + block_size - 1) / block_size;
    size_t ids_n_bytes  = num_blocks* ggml_element_size(src2);
    int64_t ids_dims[] = {src2->ne[0] * src2->ne[1]};
    ggml_tops_pool_alloc ids_allocator(ctx.pool(), ids_n_bytes);
    topsatenTensor* tops_experts_ids = tops_bias_values(ctx, ids_allocator.get(), ids_n_bytes, src2->ne, src2->nb, 1,
        ggml_tops_type_mapping(GGML_TYPE_F32), ggml_element_size(src2), 0.0f);

    int64_t ntpp_dims[] = {1};
    size_t ntpp_n_bytes = 1 * ggml_element_size(src2);
    ggml_tops_pool_alloc ntpp_allocator(ctx.pool(), ntpp_n_bytes);
    topsatenTensor* tops_num_tokens_post_pad;

    // int64_t tensor_dims[] = {1};
    // tops_num_tokens_post_pad.shape = tensor_dims;
    // tops_num_tokens_post_pad.ndim = 1;

    int32_t num_tokens_post_pad_val = static_cast<int32_t>(dst->ne[0]);
    // tops_num_tokens_post_pad.data = &num_tokens_post_pad_val;
    // debug_tensor(ctx, tops_input);
    // debug_tensor(ctx, tops_add_lhs);

    TOPS_ATEN_CHECK(topsvllm::topsvllmInvokeFusedMoeKernel(*tops_dst,*tops_A,*tops_B,*tops_weight,*tops_ids,*tops_sorted_token_ids,*tops_experts_ids, 
        *tops_num_tokens_post_pad,mul_routed_weight,topk,block_size,stream));
    // debug_tensor(ctx, tops_dst);

    // delete tops_input;
    // delete tops_add_lhs;
    // delete tops_dst;

    return;
}

void ggml_tops_op_scale(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst -> src[0];

    topsatenTensor* tops_input = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    size_t one_tensor_n_bytes = src->ne[0]*src->ne[1]*src->ne[2]*src->ne[3]*ggml_type_size(GGML_TYPE_F32);
    ggml_tops_pool_alloc one_tensor_allocator(ctx.pool(), one_tensor_n_bytes);
    topsatenTensor* tops_add_lhs = ggml_tops_create_tensor(one_tensor_allocator.get(), src->ne, src->nb, 4,src->type, GGML_TYPE_F32);

    topsatenScalar_t mul_scalar;
    mul_scalar.dtype = TOPSATEN_DATA_FP32;
    float mul_val;
    memcpy(&mul_val, &dst->op_params[0], sizeof(float));
    mul_scalar.fval = mul_val;

    topsatenScalar_t alpha;
    alpha.dtype = TOPSATEN_DATA_FP32;
    alpha.fval = 1.0;

    TOPS_ATEN_CHECK(topsaten::topsatenMul(*tops_add_lhs,mul_scalar,*tops_input,stream));

    topsatenScalar_t add_scalar;
    add_scalar.dtype = TOPSATEN_DATA_FP32;
    float add_val;
    memcpy(&add_val, &dst->op_params[1], sizeof(float));
    add_scalar.fval = add_val;

    debug_tensor(ctx, tops_input);
    debug_tensor(ctx, tops_add_lhs);
    TOPS_ATEN_CHECK(topsaten::topsatenAdd(*tops_dst,*tops_add_lhs,add_scalar,alpha,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_input;
    delete tops_add_lhs;
    delete tops_dst;

    return;
}

void ggml_tops_op_sqr(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* input = dst->src[0];

    topsatenTensor* tops_input = ggml_tops_create_tensor(input);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    topsatenScalar_t exponent;
    exponent.dtype = TOPSATEN_DATA_FP32;
    exponent.fval = 2.0f; 

    debug_tensor(ctx, tops_input);
    topsatenStatus_t status = topsaten::topsatenPow(*tops_dst, *tops_input,exponent,stream);
    debug_tensor(ctx, tops_dst);

    delete tops_input;
    delete tops_dst;
    
    return;
}

void ggml_tops_op_sqrt(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src0 = dst->src[0];
    topsStream_t stream = ctx.stream();

    topsatenTensor* tops_src0 = ggml_tops_create_tensor(src0);
    topsatenTensor* tops_dst= ggml_tops_create_tensor(dst);

    debug_tensor(ctx, tops_src0);
    TOPS_ATEN_CHECK(topsaten::topsatenSqrt(*tops_dst,*tops_src0,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src0;
    delete tops_dst;

    return;
}

void ggml_tops_op_clamp(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src0 = dst->src[0];
    topsStream_t stream = ctx.stream();

    topsatenTensor* tops_src0 = ggml_tops_create_tensor(src0);
    topsatenTensor* tops_dst= ggml_tops_create_tensor(dst);

    topsatenScalar_t min_scalar;
    min_scalar.dtype = TOPSATEN_DATA_FP32;
    float min_val;
    memcpy(&min_val, &dst->op_params[0], sizeof(float));
    min_scalar.fval = min_val;

    topsatenScalar_t max_scalar;
    max_scalar.dtype = TOPSATEN_DATA_FP32;
    float max_val;
    memcpy(&max_val, &dst->op_params[1], sizeof(float));
    max_scalar.fval = max_val;

    debug_tensor(ctx, tops_src0);
    TOPS_ATEN_CHECK(topsaten::topsatenClamp(*tops_dst,*tops_src0,min_scalar,max_scalar,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src0;
    delete tops_dst;

    return;
}

void ggml_tops_op_pool_2d(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    topsatenSize_t kernel_size;
    int kH = static_cast<int>(dst->op_params[1]);
    int kW = static_cast<int>(dst->op_params[2]);
    int64_t kernel[] = {static_cast<int64_t>(kW), static_cast<int64_t>(kH)};
    kernel_size.data = kernel;
    kernel_size.len = 2;

    topsatenSize_t stride;
    int sH = static_cast<int>(dst->op_params[3]);
    int sW = static_cast<int>(dst->op_params[4]);
    int64_t stride_vals[] = {static_cast<int64_t>(sW), static_cast<int64_t>(sH)};
    stride.data = stride_vals; 
    stride.len = 2;

    topsatenSize_t padding;
    int pH = static_cast<int>(dst->op_params[5]);
    int pW = static_cast<int>(dst->op_params[6]);
    int64_t padding_vals[] = {static_cast<int64_t>(pW), static_cast<int64_t>(pH)};
    padding.data = padding_vals; 
    padding.len = 2;

    int64_t dilation_h = 1;
    topsatenSize_t dilation;
    int64_t dilation_vals[2] = {dilation_h, 1};
    dilation.data = dilation_vals; 
    dilation.len = 2;

    int64_t output_padding_vals[2] = {0, 0};
    topsatenSize_t output_padding;
    output_padding.data = output_padding_vals; 
    output_padding.len = 2;

    bool ceil_mode = 0;
    bool count_include_pad = 1;

    topsatenScalar_t divisor_override{};
    divisor_override.ival = -1;
    divisor_override.dtype = TOPSATEN_DATA_NONE;

    debug_tensor(ctx, tops_src);
    if (dst->op_params[0] != 0){
        TOPS_ATEN_CHECK(topsaten::topsatenAvgPool2d(*tops_dst,*tops_src,kernel_size,stride,padding,ceil_mode,count_include_pad,divisor_override,stream));
    } else {
        TOPS_ATEN_CHECK(topsaten::topsatenMaxPool2d(*tops_dst,*tops_src,kernel_size,stride,padding,dilation,ceil_mode,stream));
    }

    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;
}

void ggml_tops_op_sum(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src0 = dst->src[0];

    topsatenTensor* tops_src0 = ggml_tops_create_tensor(src0);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    int32_t dtype_idx = dst->op_params[0];
    topsatenDataType_t data_type = static_cast<topsatenDataType_t>(dtype_idx);

    debug_tensor(ctx, tops_src0);
    TOPS_ATEN_CHECK(topsaten::topsatenSum(*tops_dst,*tops_src0,data_type,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src0;
    delete tops_dst;

    return;
}

void ggml_tops_op_sum_rows(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src0 = dst->src[0];
    topsStream_t stream = ctx.stream();

    topsatenTensor* tops_src0 = ggml_tops_create_tensor(src0);
    topsatenTensor* tops_dst= ggml_tops_create_tensor(dst);

    int64_t input_dim = tops_src0 -> GetTensorShape().len;
    int64_t target_sum_dim = input_dim - 1; 
    int64_t* dim_data = new int64_t[1];
    dim_data[0] = target_sum_dim;

    topsatenSize_t dimensions;
    dimensions.len = 1;
    dimensions.data = dim_data;

    bool keepdims = true;
    topsatenDataType_t data_type = TOPSATEN_DATA_FP32;

    debug_tensor(ctx, tops_src0);
    TOPS_ATEN_CHECK(topsaten::topsatenSum(*tops_dst,*tops_src0, dimensions,keepdims,data_type,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src0;
    delete tops_dst;

    return;
}

void ggml_tops_op_argsort(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src0 = dst->src[0];
    topsStream_t stream = ctx.stream();

    topsatenTensor* tops_src0 = ggml_tops_create_tensor(src0);
    topsatenTensor* tops_dst= ggml_tops_create_tensor(dst);

    bool stable = false;
    int64_t dim = -1;
    bool descending = false;

    descending = (dst->op_params[0] != 0);


    debug_tensor(ctx, tops_src0);
    TOPS_ATEN_CHECK(topsaten::topsatenArgSort(*tops_dst,*tops_src0, stable,dim,descending,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src0;
    delete tops_dst;

    return;
}

void ggml_tops_op_argmax(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src0 = dst->src[0];
    topsStream_t stream = ctx.stream();

    topsatenTensor* tops_src0 = ggml_tops_create_tensor(src0);
    topsatenTensor* tops_dst= ggml_tops_create_tensor(dst);

    topsatenScalar_t dimensions;
    int64_t input_dim = tops_src0->GetTensorShape().len;
    int64_t target_dim = input_dim - 1;
    dimensions.dtype = TOPSATEN_DATA_FP32;
    dimensions.ival = target_dim;

    bool keepdims = false;

    debug_tensor(ctx, tops_src0);
    TOPS_ATEN_CHECK(topsaten::topsatenArgmax(*tops_dst,*tops_src0, dimensions,keepdims,stream));
    debug_tensor(ctx, tops_dst,stream);

    delete tops_src0;
    delete tops_dst;

    return;
}

void ggml_tops_op_cos(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenCos(*tops_dst,*tops_src,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;
}

void ggml_tops_op_sin(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenSin(*tops_dst,*tops_src,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;
}

/**
 * @brief ConvTranspose2d operator.
 *
 * @param output: The output tensor.
 * @param input: The input tensor of feature.
 * @param filter: The input tensor of reduce window.
 * @param bias: The input tensor of bias, disabled if tensor datatype is
 *        TOPSATEN_DATA_NONE.
 * @param stride: The stride of the convolving kernel.
 * @param padding: The amount of padding applied to the input shape, can be
 *        either a conv-rank list of ints giving the amount of padding applied
 *        on both sides (eg. [pad_h, pad_w] for conv2d), or a 2x conv-rank list
 *        of ints giving the amount of padding applied on each side
 *        (eg. [pad_h_low, pad_h_high, pad_w_low, pad_w_high] for conv2d).
 * @param output_padding: Padding added to output shape, not currently enabled.
 * @param groups: Controls the connections between inputs and outputs,
 *        in_channels and out_channels must both be divisible by groups.
 * @param dilation: The spacing between kernel elements.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenConvTranspose2d(
//     topsatenTensor& output, const topsatenTensor& input,
//     const topsatenTensor& filter, const topsatenTensor& bias,
//     const topsatenSize_t& stride, const topsatenSize_t& padding,
//     const topsatenSize_t& output_padding, int64_t groups,
//     const topsatenSize_t& dilation, topsStream_t stream = nullptr);
void ggml_tops_op_transpose_1d(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[1]; // WC_i1N
    ggml_tensor* src_filter = dst->src[0]; //SC_oC_i1
    ggml_tensor* src_bias = dst->src[2]; 

    topsatenTensor* tops_src = ggml_tops_create_tensor(src); //N1WC_i
    topsatenTensor* tops_filter = ggml_tops_create_tensor(src_filter); //RSC_oC_i
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    int len = tops_src->GetTensorShape().len;
    TOPS_ATEN_CHECK(topsaten::topsatenTranspose(*tops_src,*tops_src,len-1,len-2,stream));
    TOPS_ATEN_CHECK(topsaten::topsatenTranspose(*tops_filter,*tops_filter,len-1,len-2,stream));
    TOPS_ATEN_CHECK(topsaten::topsatenTranspose(*tops_dst,*tops_dst,len-1,len-2,stream));

    size_t bais_n_bytes = src->ne[1] * src->ne[0] * ggml_element_size(src);
    ggml_tops_pool_alloc bais_allocator(ctx.pool(), bais_n_bytes);

    int64_t weight_ne[] = {src->ne[1], src->ne[0],
        src->ne[2], src->ne[3]};
    size_t weight_nb[] = {src->nb[1], src->nb[0],
        src->nb[2], src->nb[3]};

    topsatenTensor* tops_bias = tops_bias_values(ctx, bais_allocator.get(), bais_n_bytes, weight_ne, weight_nb, 1,
        ggml_tops_type_mapping(src->type), ggml_element_size(src), 0.0f);

    topsatenSize_t stride;
    int sH = 1;
    int s0 = static_cast<int>(dst->op_params[0]);
    int64_t stride_dims[] = {static_cast<int64_t>(s0), static_cast<int64_t>(sH)};
    stride.data = stride_dims;
    stride.len = 2;

    topsatenSize_t padding;
    int pH = 0;
    int p0 = static_cast<int>(dst->op_params[1]);
    int64_t padding_dims[] = {static_cast<int64_t>(p0), static_cast<int64_t>(pH)};
    padding.data = padding_dims;
    padding.len = 2;

    topsatenSize_t output_padding;
    int64_t output_padding_dims[] = {0, 0}; 
    output_padding.data = output_padding_dims;
    output_padding.len = 2;

    topsatenSize_t dilation;
    int dH = 1;
    int d0 = static_cast<int>(dst->op_params[2]);
    int64_t dilation_dims[] = {static_cast<int64_t>(d0), static_cast<int64_t>(dH)};
    dilation.data = dilation_dims;
    dilation.len = 2;

    int groups = 1;

    debug_tensor(ctx, tops_src);
    debug_tensor(ctx, tops_filter);
    debug_tensor(ctx, tops_bias);
    debug_tensor(ctx, tops_dst);
    TOPS_ATEN_CHECK(topsaten::topsatenConvTranspose2d(*tops_dst,*tops_src,*tops_filter,*tops_bias,stride,padding,output_padding, groups,dilation,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;
}

void ggml_tops_op_log(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenLog(*tops_dst,*tops_src,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;
}

void ggml_tops_op_mean(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src = dst->src[0];

    topsatenTensor* tops_src = ggml_tops_create_tensor(src);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    int64_t dims_in = -1;
    bool keepdims = true;
    topsatenDataType_t data_type = TOPSATEN_DATA_FP32;

    debug_tensor(ctx, tops_src);
    TOPS_ATEN_CHECK(topsaten::topsatenMean(*tops_dst,*tops_src,dims_in,keepdims,data_type,stream));
    debug_tensor(ctx, tops_dst);

    delete tops_src;
    delete tops_dst;

    return;
}

void ggml_tops_op_reflect_1d(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src_input = dst->src[0];

    topsatenTensor* tops_input = ggml_tops_create_tensor(src_input);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    
    auto tops_lhs_data = tops_input->GetTensorShape().data;
    auto tops_output_data = tops_dst->GetTensorShape().data;
    auto tops_lhs_nb = tops_input->GetTensorStrides().data;
    auto tops_output_nb = tops_dst->GetTensorStrides().data;

    int pad0 = 10, pad1 = 9;
    // pad0 = static_cast<int>(dst->op_params[0]);
    // pad1 = static_cast<int>(dst->op_params[1]);

    int64_t pad_count = 8;
    int64_t* pad_data = new int64_t[pad_count]{
        pad0, pad1, 0, 0,
        0, 0, 0, 0
    };

    topsatenSize_t pads;
    pads.len = pad_count;
    pads.data = pad_data;

    topsatenPadMode_t mode = TOPSATEN_PAD_REFLECT;
    topsatenScalar_t value;
    value.dtype = TOPSATEN_DATA_FP32;
    value.fval = 0.0f;
    
    debug_tensor(ctx, tops_input);
    TOPS_ATEN_CHECK(topsaten::topsatenPad(*tops_dst,*tops_input,pads,mode,value,stream))
    debug_tensor(ctx, tops_dst);

    delete tops_input;
    delete tops_dst;

    return;
}

void ggml_tops_op_count_equal(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];

    topsatenTensor* tops_src0 = ggml_tops_create_tensor(src0);
    topsatenTensor* tops_src1 = ggml_tops_create_tensor(src1);
    topsStream_t stream = ctx.stream();

    size_t one_tensor_n_bytes = src0->ne[0]*src0->ne[1]*src0->ne[2]*src0->ne[3]*ggml_type_size(GGML_TYPE_F32);
    ggml_tops_pool_alloc one_tensor_allocator(ctx.pool(), one_tensor_n_bytes);
    topsatenTensor* tops_eq = ggml_tops_create_tensor(one_tensor_allocator.get(),src0->ne, src0->nb, 4,src0->type, GGML_TYPE_F32);
    topsatenTensor* tops_eq_pr = new topsatenTensor(tops_eq->GetTensorShape(), tops_eq->GetTensorStrides(), TOPSATEN_DATA_PRED, tops_eq->GetTensorData());

    debug_tensor(ctx, tops_src0);
    debug_tensor(ctx, tops_src1);
    TOPS_ATEN_CHECK(topsaten::topsatenEq(*tops_eq_pr,*tops_src0,*tops_src1,stream));
    debug_tensor(ctx, tops_eq_pr, stream);

    topsatenTensor* tops_eq_i32 = new topsatenTensor(tops_eq_pr->GetTensorShape(), tops_eq_pr->GetTensorStrides(), TOPSATEN_DATA_I32, tops_eq_pr->GetTensorData());
    topsatenDataType_t target_dtype = TOPSATEN_DATA_I32;
    TOPS_ATEN_CHECK(topsaten::topsatenTo(*tops_eq_i32,*tops_eq_pr,target_dtype,false,false, TOPSATEN_MEMORY_CONTIGUOUS,stream));
    debug_tensor(ctx, tops_eq_i32, stream);

    size_t one_tensor_n_bytes_dst = dst->ne[0]*dst->ne[1]*dst->ne[2]*dst->ne[3]*ggml_type_size(GGML_TYPE_F32);
    ggml_tops_pool_alloc one_tensor_allocator_dst(ctx.pool(), one_tensor_n_bytes_dst);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(one_tensor_allocator_dst.get(),dst->ne, dst->nb, 4,dst->type, GGML_TYPE_I32);

    topsatenDataType_t data_type = TOPSATEN_DATA_I32;
    debug_tensor(ctx, tops_eq_i32);
    TOPS_ATEN_CHECK(topsaten::topsatenSum(*tops_dst,*tops_eq_i32,data_type,stream));
    debug_tensor(ctx, tops_dst, stream);

    topsatenTensor* tops_dst_i64 = ggml_tops_create_tensor(dst);
    topsatenDataType_t target_dtype_dst = TOPSATEN_DATA_I64;
    TOPS_ATEN_CHECK(topsaten::topsatenTo(*tops_dst_i64,*tops_dst,target_dtype_dst,true,true, TOPSATEN_MEMORY_CONTIGUOUS,stream));
    debug_tensor(ctx, tops_dst_i64, stream);

    delete tops_src0;
    delete tops_src1;
    delete tops_dst;

    return;
}

/**
* @brief Multi query kv attention operator.
*
* @param output: The output tensor.
* @param query: Input tensor query.
* @param key: Input tensor key.
* @param value: Input tensor value.
* @param attn_bias: Attention mask.
* @param dropout_p: Dropout probability.
* @param scale: Input scale.
* @param stream: Tops stream.
* @return topsatenStatus_t.
*/
// topsatenStatus_t TOPSATEN_EXPORT topsvllmMemoryEfficientAttention(
//     topsatenTensor& output, const topsatenTensor& query,
//     const topsatenTensor& key, const topsatenTensor& value,
//     const topsatenTensor& attn_bias, const topsatenScalar_t& dropout_p,
//     const topsatenScalar_t& scale, topsStream_t stream = nullptr,
//     const topsatenTensor& scale_query = topsatenTensor(),
//     const topsatenTensor& scale_key = topsatenTensor(),
//     const topsatenTensor& scale_value = topsatenTensor(),
//     const topsatenTensor& scale_score = topsatenTensor(),
//     const topsatenTensor& scale_weight = topsatenTensor(),
//     const topsatenTensor& scale_out = topsatenTensor());
void ggml_tops_op_flash_atten_ext(ggml_backend_tops_context & ctx, ggml_tensor * dst){
    ggml_tensor* src_query = dst->src[0];
    ggml_tensor* src_key = dst->src[1];
    ggml_tensor* src_value = dst->src[2];
    ggml_tensor* src_attn_bias = dst->src[3]; 

    int64_t dim1 = dst->ne[1];
    int64_t dim2 = dst->ne[2];
    dst->ne[1] = dim2;
    dst->ne[2] = dim1;

    int64_t nb1 = dst->nb[1];
    dst->nb[2] = nb1;

    topsatenTensor* tops_query = ggml_tops_create_tensor(src_query); //bcast
    topsatenTensor* tops_key = ggml_tops_create_tensor(src_key);
    topsatenTensor* tops_value = ggml_tops_create_tensor(src_value);
    topsatenTensor* tops_attn_bias = ggml_tops_create_tensor(src_attn_bias);
    topsatenTensor* tops_dst = ggml_tops_create_tensor(dst);
    topsStream_t stream = ctx.stream();

    auto len = tops_query->GetTensorShape().len;
    TOPS_ATEN_CHECK(topsaten::topsatenTranspose(*tops_query,*tops_query,len-1,len-2,stream));
    TOPS_ATEN_CHECK(topsaten::topsatenTranspose(*tops_key, *tops_key, len-1, len-2, stream));
    TOPS_ATEN_CHECK(topsaten::topsatenTranspose(*tops_value, *tops_value, len-1, len-2, stream));

    size_t one_tensor_n_bytes_k = src_key->ne[0]*src_key->ne[1]*src_key->ne[2]*src_key->ne[3]*ggml_type_size(GGML_TYPE_F32);
    ggml_tops_pool_alloc one_tensor_allocator_k(ctx.pool(), one_tensor_n_bytes_k);
    topsatenTensor* tops_k = ggml_tops_create_tensor(one_tensor_allocator_k.get(),src_key->ne, src_key->nb, 4,src_key->type, GGML_TYPE_F32);
    topsatenTensor* k_tensor_fp32 = new topsatenTensor(tops_k->GetTensorShape(), tops_k->GetTensorStrides(), TOPSATEN_DATA_FP32, tops_k->GetTensorData());

    size_t one_tensor_n_bytes_v = src_value->ne[0]*src_value->ne[1]*src_value->ne[2]*src_value->ne[3]*ggml_type_size(GGML_TYPE_F32);
    ggml_tops_pool_alloc one_tensor_allocator_v(ctx.pool(), one_tensor_n_bytes_v);
    topsatenTensor* tops_v = ggml_tops_create_tensor(one_tensor_allocator_v.get(),src_value->ne, src_value->nb, 4,src_value->type, GGML_TYPE_F32);
    topsatenTensor* v_tensor_fp32 = new topsatenTensor(tops_v->GetTensorShape(), tops_v->GetTensorStrides(), TOPSATEN_DATA_FP32, tops_v->GetTensorData());

    size_t one_tensor_n_bytes = src_attn_bias->ne[0]*src_attn_bias->ne[1]*src_attn_bias->ne[2]*src_attn_bias->ne[3]*ggml_type_size(GGML_TYPE_F32);
    ggml_tops_pool_alloc one_tensor_allocator(ctx.pool(), one_tensor_n_bytes);
    topsatenTensor* attn_bias = ggml_tops_create_tensor(one_tensor_allocator.get(),src_attn_bias->ne, src_attn_bias->nb, 4,src_attn_bias->type, GGML_TYPE_F32);
    topsatenTensor* tops_attn_bias_f32 = new topsatenTensor(attn_bias->GetTensorShape(), attn_bias->GetTensorStrides(), TOPSATEN_DATA_FP32, attn_bias->GetTensorData());

    float dropout_p_val;
    topsatenScalar_t dropout_p;
    dropout_p.dtype = TOPSATEN_DATA_FP32;
    dropout_p.fval = 0.0;
    if (dst->op_params) {
        memcpy(&dropout_p_val, reinterpret_cast<const void*>(&dst->op_params[0]), sizeof(float));
        dropout_p.fval = dropout_p_val;
    } else {
        dropout_p.fval = 0.0;
    }

    float scale_val;
    topsatenScalar_t scale;
    scale.dtype = TOPSATEN_DATA_FP32;
    if (dst->op_params) {
        memcpy(&scale_val, reinterpret_cast<const void*>(&dst->op_params[1]), sizeof(float));
        scale.fval = scale_val;
    } else{
        scale.fval = 1.0f;
    }
    
    topsatenDataType_t target_dtype_fp32 = TOPSATEN_DATA_FP32;
    TOPS_ATEN_CHECK(topsaten::topsatenTo(
        *k_tensor_fp32, *tops_key, target_dtype_fp32, true, true, TOPSATEN_MEMORY_CONTIGUOUS, stream));
    TOPS_ATEN_CHECK(topsaten::topsatenTo(
        *v_tensor_fp32, *tops_value, target_dtype_fp32, true, true, TOPSATEN_MEMORY_CONTIGUOUS, stream));
    TOPS_ATEN_CHECK(topsvllm::topsvllmMemoryEfficientAttention(
        *tops_dst, *tops_query, *k_tensor_fp32, *v_tensor_fp32, *tops_attn_bias_f32, dropout_p, scale, stream));

    debug_tensor(ctx, tops_query);
    debug_tensor(ctx, k_tensor_fp32);
    debug_tensor(ctx, v_tensor_fp32);
    debug_tensor(ctx, tops_attn_bias_f32);
    debug_tensor(ctx, tops_dst);
    TOPS_ATEN_CHECK(topsvllm::topsvllmMemoryEfficientAttention(*tops_dst, *tops_query, *k_tensor_fp32, *v_tensor_fp32, *tops_attn_bias_f32, dropout_p, scale, stream));

    delete tops_query;
    delete tops_key;
    delete tops_value;
    delete tops_attn_bias;
    delete tops_dst;

    return;
}