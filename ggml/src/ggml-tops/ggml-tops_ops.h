#pragma once
#include <topsaten/topsaten.h>
#include <topsaten/topsaten_vllm.h>
#include "ggml-tops_tensor.h"
#include "common.h"

/**
 * @brief Compute RMSNorm on the input.
 *
 * @param output: Output tensor of shape [..., hidden_size].
 * @param input: Input tensor of shape [..., hidden_size].
 * @param gamma: Gamma tensor of shape [hidden_size].
 * @param epsilon: Value added to denominator for numerical stability.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsvllmRmsNorm(
//     topsatenTensor& output, const topsatenTensor& input,
//     const topsatenTensor& weight, const topsatenScalar_t& epsilon,
//     topsStream_t stream = nullptr);
void ggml_tops_op_rms_norm(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function multiplies lhs by rhs.
 *
 * @param output: The output tensor.
 * @param lhs: The input tensor.
 * @param rhs: The input tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenMul(topsatenTensor &output,
//     const topsatenTensor &lhs,
//     const topsatenTensor &rhs,
//     topsStream_t stream = nullptr);
void ggml_tops_op_mul(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief Add operator.
 *
 * @param out: The output tensor.
 * @param lhs: Input lhs tensor.
 * @param rhs: Input rhs scalar.
 * @param alpha: The multiplier for lhs tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenAdd(topsatenTensor &out,
//                                              const topsatenTensor &lhs,
//                                              const topsatenScalar_t &rhs,
//                                              const topsatenScalar_t &alpha,
//                                              topsStream_t stream = nullptr);
void ggml_tops_op_mul_mat(ggml_backend_tops_context & ctx, ggml_tensor * dst);

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
void ggml_tops_op_rope(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function performs the copy function.
 *
 * @param out: The output tensor.
 * @param input: The source tensor.
 * @param non_blocking: If true and this copy is between CPU and GCU.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenCopy(topsatenTensor &out,
//     const topsatenTensor &input,
//     bool non_blocking,
//     topsStream_t stream = nullptr);
void ggml_tops_op_cpy(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function performs the forward softmax.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param dims: The dims of output tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */

//  topsatenStatus_t TOPSATEN_EXPORT
//  topsatenSoftmaxForward(topsatenTensor &output, const topsatenTensor &input,
//                         const int64_t dims, topsStream_t stream = nullptr);
void ggml_tops_op_softmax(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function performs the clone function.
 *
 * @param out: The output tensor.
 * @param input: The input tensor.
 * @param format: The format of the out tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenClone(
//     topsatenTensor &out, const topsatenTensor &input,
//     const topsatenMemoryFormat_t &format, topsStream_t stream = nullptr);
void ggml_tops_op_dup(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief Returns a tensor that is a transposed version of input.
 * The given dimensions dim0 and dim1 are swapped.
 *
 * @param out: The output tensor.
 * @param input: The input tensor.
 * @param dim0: The first dimension to be transposed.
 * @param dim1: The second dimension to be transposed.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenTranspose(
//     topsatenTensor &output, const topsatenTensor &input, const int64_t &dim0,
//     const int64_t &dim1, topsStream_t stream = nullptr);
void ggml_tops_op_transpose(ggml_backend_tops_context & ctx, ggml_tensor * dst);
void ggml_tops_op_reshape(ggml_backend_tops_context & ctx, ggml_tensor * dst);
void ggml_tops_op_view(ggml_backend_tops_context & ctx, ggml_tensor * dst);
void ggml_tops_op_permute(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function applies the Gated Linear Unit (GLU) activation operation
 *        on the input tensor, splitting it into two halves on a specified
 *        dimension and then multiplying one half by the sigmoid of the other
 *        half for non-linear transformation.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param dim: Dimension on which to split the input. Default: -1.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenGlu(topsatenTensor &output,
//     const topsatenTensor &input,
//     const int32_t dim = -1,
//     topsStream_t stream = nullptr);
void ggml_tops_op_glu(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief Repeats this tensor along the specified dimensions.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param repeats: The number of times to
 *               repeat this tensor along each dimension.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenRepeat(topsatenTensor &output,
//     const topsatenTensor &input,
//     const topsatenSize_t &repeats,
//     topsStream_t stream = nullptr);
void ggml_tops_op_repeat(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief Add operator.
 *
 * @param out: The output tensor.
 * @param lhs: Input lhs tensor.
 * @param rhs: Input rhs tensor.
 * @param alpha: The multiplier for lhs tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenAdd(topsatenTensor &out,
//     const topsatenTensor &lhs,
//     const topsatenTensor &rhs,
//     const topsatenScalar_t &alpha,
//     topsStream_t stream = nullptr);
void ggml_tops_op_add(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function subtracts rhs, scaled by alpha, from lhs.
 *        output = lhs - rhs * alpha.
 *
 * @param output: The output tensor.
 * @param lhs: The input tensor.
 * @param rhs: The tensor to subtract from lhs.
 * @param alpha: The multiplier for rhs.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenSub(topsatenTensor &output,
//     const topsatenTensor &lhs,
//     const topsatenTensor &rhs,
//     const topsatenScalar_t &alpha,
//     topsStream_t stream = nullptr);
void ggml_tops_op_sub(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief Add operator.
 *
 * @param out: The output tensor.
 * @param lhs: Input lhs tensor.
 * @param rhs: Input rhs scalar.
 * @param alpha: The multiplier for lhs tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenAdd(topsatenTensor &out,
//     const topsatenTensor &lhs,
//     const topsatenScalar_t &rhs,
//     const topsatenScalar_t &alpha,
//     topsStream_t stream = nullptr);
void ggml_tops_op_acc(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function divides each element of the input lhs by the
 *        corresponding element of rhs.
 *
 * @param output: The output tensor.
 * @param lhs: The input tensor.
 * @param rhs: The input tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenDiv(topsatenTensor &output,
//     const topsatenTensor &lhs,
//     const topsatenTensor &rhs,
//     topsStream_t stream = nullptr);
void ggml_tops_op_div(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief Computes the absolute value of each element in input.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenAbs(topsatenTensor &output,
//     const topsatenTensor &input,
//     topsStream_t stream = nullptr);
void ggml_tops_op_abs(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function returns a new tensor with the negative of
 *        the elements of input.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenNeg(topsatenTensor &output,
//     const topsatenTensor &input,
//     topsStream_t stream = nullptr);
void ggml_tops_op_neg(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief Applies the Gaussian Error Linear Units function of the elements
 *        of input.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param approximate: The gelu approximation algorithm to use.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenGelu(
//     topsatenTensor &output, const topsatenTensor &input,
//     const char *const approximate = "none", topsStream_t stream = nullptr);
void ggml_tops_op_gelu(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function applies the Sigmoid Linear Unit (SiLU) function,
 *        element-wise.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenSilu(topsatenTensor &output,
//     const topsatenTensor &input,
//     topsStream_t stream = nullptr);
void ggml_tops_op_silu(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief Gelu quick operator.
 *
 * @param out: The output tensor.
 * @param in: Input tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT
// topsvllmGeluQuick(topsatenTensor& out, const topsatenTensor& in,
//                   topsStream_t stream = nullptr);
void ggml_tops_op_gelu_quick(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function returns a new tensor with the hyperbolic tangent
 *        of the elements of input.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenTanh(topsatenTensor &output,
//     const topsatenTensor &input,
//     topsStream_t stream = nullptr);
void ggml_tops_op_tanh(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief Applies the rectified linear unit function element-wise.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenRelu(topsatenTensor &output,
//     const topsatenTensor &input,
//     topsStream_t stream = nullptr);
void ggml_tops_op_relu(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief Computes the expit of the elements of input.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenSigmoid(topsatenTensor &output,
//     const topsatenTensor &input,
//     topsStream_t stream = nullptr);
void ggml_tops_op_sigmoid(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function computes the hard sigmoid activation of the input tensor
 *        and writes the result directly into the output tensor.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT
// topsatenHardsigmoid(topsatenTensor &output, const topsatenTensor &input,
//                     topsStream_t stream = nullptr);
void ggml_tops_op_hardsigmoid(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function applies the Hard Swish activation function to the input
 *        tensor, providing a piecewise linear approximation of the Swish
 *        activation for efficient computation.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT
// topsatenHardswish(topsatenTensor &output, const topsatenTensor &input,
//                   topsStream_t stream = nullptr);
void ggml_tops_op_hardswish(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief Returns a new tensor with the exponential of the elements
 *        of the input tensor.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenExp(topsatenTensor &output,
//     const topsatenTensor &input,
//     topsStream_t stream = nullptr);
void ggml_tops_op_exp(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief Applies the Exponential Linear Unit (ELU) function.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param alpha: Value for the ELU formulation. Default: 1.0.
 * @param scale: Value for the ELU formulation. Default: 1.0.
 * @param input_scale: Value for the ELU formulation. Default: 1.0.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenElu(
//     topsatenTensor &output, const topsatenTensor &input,
//     const topsatenScalar_t &alpha = {TOPSATEN_DATA_FP32, {.fval = 1.0}},
//     const topsatenScalar_t &scale = {TOPSATEN_DATA_FP32, {.fval = 1.0}},
//     const topsatenScalar_t &input_scale = {TOPSATEN_DATA_FP32, {.fval = 1.0}},
//     topsStream_t stream = nullptr);
void ggml_tops_op_elu(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief  It computes a new tensor whose elements have the same angles as the
 * corresponding elements of input and absolute values (i.e. magnitudes) of
 *  one for complex tensors.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */

//  topsatenStatus_t TOPSATEN_EXPORT topsatenSgn(topsatenTensor &output,
//     const topsatenTensor &input,
//     topsStream_t stream = nullptr);
void ggml_tops_op_sgn(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function performs the forward layer normalization.
 *
 * @param output: Normalized tensor.
 * @param input: Tensor to be normalized.
 * @param normalized_shape: The shape of the normalization dimension.
 * @param weight: Weight tensor.
 * @param bias: Bias tensor.
 * @param epsilon: The epsilon value to use to avoid division by zero.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenLayerNorm(
//     topsatenTensor &output, const topsatenTensor &input,
//     const topsatenSize_t &normalized_shape, const topsatenTensor &weight,
//     const topsatenTensor &bias, const topsatenScalar_t &epsilon,
//     topsStream_t stream = nullptr);
void ggml_tops_op_norm(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function performs the forward group normalization.
 *
 * @param output: Normalized tensor.
 * @param input: Tensor to be normalized.
 * @param group: Number of groups to separate the channels into.
 * @param weight: Weight tensor.
 * @param bias: Bias tensor.
 * @param eps: A value added to the denominator for numerical stability.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenGroupNorm(
//     topsatenTensor &output, const topsatenTensor &input, const int64_t group,
//     const topsatenTensor &weight, const topsatenTensor &bias,
//     const double eps = 1e-05, topsStream_t stream = nullptr);
void ggml_tops_op_groupnorm(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief Concatenate operator.
 *
 * @param output: The output tensor.
 * @param inputs: The input tensor.
 * @param dim: Concatenate dim.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT
// topsatenCat(topsatenTensor &output, const std::vector<topsatenTensor> &inputs,
//             int64_t dim, topsStream_t stream = nullptr);
void ggml_tops_op_concat(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function performs the upsample nearest2d function.
 *
 * @param out: The output tensor.
 * @param input: The source tensor.
 * @param scales_h: The topsatenScalar_t of scales_h.
 * @param scales_w: The topsatenScalar_t of scales_w.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenUpsampleNearest2d(
//     topsatenTensor &output, const topsatenTensor &input,
//     const topsatenSize_t &output_size, const topsatenScalar_t &scales_h,
//     const topsatenScalar_t &scales_w, topsStream_t stream = nullptr);
/**
 * @brief This function performs the upsample nearest3d function.
 *
 * @param output: The output tensor.
 * @param input: The source tensor.
 * @param output_size: The size of the output tensor.
 * @param scales_d: The topsatenScalar_t of scales_d.
 * @param scales_h: The topsatenScalar_t of scales_h.
 * @param scales_w: The topsatenScalar_t of scales_w.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenUpsampleNearest3d(
//     topsatenTensor &output, const topsatenTensor &input,
//     const topsatenSize_t &output_size, const topsatenScalar_t &scales_d,
//     const topsatenScalar_t &scales_h, const topsatenScalar_t &scales_w,
//     topsStream_t stream = nullptr);
/**
 * @brief This function performs 2D bilinear upsampling on a multi-channel
 *        input signal. You can define the scale using either the output size
 *        (h, w) or a scaling factor (scale_factor).
 *
 * @param output: The out tensor.
 * @param input: The input tensor.
 * @param size: Output spatial sizes.
 * @param align_corners: If True, the corner pixels of the input and output
 *                       tensors are aligned.
 * @param scale_h: Multiplier for spatial size.
 * @param scale_w: Multiplier for spatial size.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenUpsampleBilinear2d(
//     topsatenTensor &output, const topsatenTensor &input,
//     const topsatenSize_t &size, bool align_corners,
//     const topsatenScalar_t &scale_h, const topsatenScalar_t &scale_w,
//     topsStream_t stream = nullptr);

/**
 * @brief This function performs 3D Trilinear upsampling on a multi-channel
 *        input signal. You can define the scale using either the output size
 *        (d, h, w) or a scaling factor (scale_factor).
 *
 * @param output: The out tensor.
 * @param input: The input tensor.
 * @param output_size: The size of the output tensor.
 * @param align_corners: If True, the corner pixels of the input and output
 *                       tensors are aligned.
 * @param scales_d: The topsopScalar_t of scales_d.
 * @param scales_h: Multiplier for spatial size.
 * @param scales_w: Multiplier for spatial size.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenUpsampleTrilinear3d(
//                                          topsatenTensor &output,
//                                          const topsatenTensor &input,
//                                          const topsatenSize_t &output_size,
//                                          bool align_corners,
//                                          const topsatenScalar_t &scales_d,
//                                          const topsatenScalar_t &scales_h,
//                                          const topsatenScalar_t &scales_w,
//                                          topsStream_t stream);
void ggml_tops_op_upscale(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function performs the pad function.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param pads: M-elements tuple, where m/2 <= input dimensions and m is even.
 * @param mode: `constant`, `reflect`, `replicate` or `circular`. Default:
 * `constant`.
 * @param value: A scalar value to be used if the mode chosen is `constant`
 *              (by default it is 0).
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenPad(topsatenTensor &output,
//     const topsatenTensor &input,
//     const topsatenSize_t &pads,
//     const topsatenPadMode_t &mode,
//     const topsatenScalar_t &value,
//     topsStream_t stream = nullptr);
void ggml_tops_op_pad(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function returns a tensor with values from the interval
 *        [start, end) taken with common difference step beginning from start,
 *        in this function start always equals to 0 and step always equals to 1.
 *
 * @param output: The output tensor.
 * @param end: The ending value for the set of points.
 * @param dtype: The desired data type of returned tensor.
 * @param layout: The desired layout of returned Tensor.
 * @param pinMemory: If set to True, the returned tensor will be allocated in
 *                   pinned memory.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenArange(
//     topsatenTensor &output, const topsatenScalar_t &end,
//     const topsatenDataType_t &dtype, const topsatenLayoutType_t &layout,
//     bool pinMemory, topsStream_t stream = nullptr);
void ggml_tops_op_arange(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief Applies the function: leaky_relu(x) = x >= 0 ? x : negative_slope * x.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param negative_slope: Controls the angle of the negative slope
 *                        (which is used for negative input values). Default:
 * 1e-2.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT
// topsatenLeakyRelu(topsatenTensor &output, const topsatenTensor &input,
//                   const topsatenScalar_t &negative_slope = {TOPSATEN_DATA_FP32,
//                                                             {.fval = 0.01}},
//                   topsStream_t stream = nullptr);
void ggml_tops_op_leaky_relu(ggml_backend_tops_context & ctx, ggml_tensor * dst);

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
void ggml_tops_op_mul_mat_id(ggml_backend_tops_context & ctx, ggml_tensor * dst);

//  mul + add
/**
 * @brief This function multiplies lhs by rhs.
 *
 * @param output: The output tensor.
 * @param lhs: The input tensor.
 * @param rhs: The input scalar.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenMul(topsatenTensor &output,
//     const topsatenTensor &lhs,
//     const topsatenScalar_t &rhs,
//     topsStream_t stream = nullptr);

/**
 * @brief This function multiplies lhs by rhs.
 *
 * @param output: The output tensor.
 * @param lhs: The input tensor.
 * @param rhs: The input scalar.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenAdd(topsatenTensor &output,
//     const topsatenTensor &lhs,
//     const topsatenScalar_t &rhs,
//     topsStream_t stream = nullptr);
void ggml_tops_op_scale(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function takes the power of each element in input with exponent
 *        and returns a tensor with the result.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param exponent: The exponent scalar.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenPow(topsatenTensor &output,
//     const topsatenTensor &input,
//     const topsatenScalar_t &exponent,
//     topsStream_t stream = nullptr);
void ggml_tops_op_sqr(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief Returns a new tensor with the square-root of the elements of input.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenSqrt(topsatenTensor &output,
//     const topsatenTensor &input,
//     topsStream_t stream = nullptr);
void ggml_tops_op_sqrt(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief Returns the tensor after performing a lower and a higher bound
 *        truncation operation on the input tensor.
 *
 * @param output: The output tensor.
 * @param input: Input tensor to perform lower bound truncation operation.
 * @param min: Replace elements less than the specified lower limit tensor
 *            with elements corresponding to the lower limit tensor.
 * @param max: Replace elements less than the specified higher limit tensor
 *            with elements corresponding to the higher limit tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */

//  topsatenStatus_t TOPSATEN_EXPORT topsatenClamp(topsatenTensor &output,
//     const topsatenTensor &input,
//     const topsatenScalar_t &min,
//     const topsatenScalar_t &max,
//     topsStream_t stream = nullptr);
void ggml_tops_op_clamp(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @param output : The output tensor.
 * @param input : The input tensor.
 * @param kernel_size : size of the pooling region.
 *                      Can be a single number or a tuple (kH, kW).
 * @param stride : stride of the pooling operation.
 *                 Can be a single number or a tuple (sH, sW).
 * @param padding : Implicit negative infinity padding to be added on
 *                  both sides, must be >= 0 and <= kernel_size / 2.
 * @param ceil_mode : If True, will use ceil instead of floor to compute
 *                    the output shape.
 * @param count_include_pad : when True, will include the zero-padding
 *                    in the averaging calculation.
 * @param divisor_override : optional(int) if specified, it will be used as
 * divisor,
 *                    otherwise size of the pooling region will be used.
 * @param stream : tops stream.
 * @return topsatenStatus_t
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenAvgPool2d(
//     topsatenTensor &output, const topsatenTensor &input,
//     const topsatenSize_t &kernel_size, const topsatenSize_t &stride,
//     const topsatenSize_t &padding, bool ceil_mode, bool count_include_pad,
//     const topsatenScalar_t &divisor_override, topsStream_t stream = nullptr);
/**
 * @param output : The output tensor.
 * @param input : The input tensor.
 * @param kernel_size : size of the pooling region.
 *                      Can be a single number or a tuple (kH, kW).
 * @param stride : stride of the pooling operation.
 *                 Can be a single number or a tuple (sH, sW).
 * @param padding : Implicit negative infinity padding to be added on
 *                  both sides, must be >= 0 and <= kernel_size / 2.
 * @param dilation : The stride between elements within a sliding window,
 *                   must be > 0.
 * @param ceil_mode : If True, will use ceil instead of floor to compute
 *                    the output shape.
 * @param stream : tops stream.
 * @return topsatenStatus_t
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenMaxPool2d(
//     topsatenTensor &output, const topsatenTensor &input,
//     const topsatenSize_t &kernel_size, const topsatenSize_t &stride,
//     const topsatenSize_t &padding, const topsatenSize_t &dilation,
//     bool ceil_mode, topsStream_t stream = nullptr);
void ggml_tops_op_pool_2d(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
* @brief This function performs a addition computation.
*
* @param output: The output tensor.
* @param input: The input tensor.
* @param data_type: Data type.
* @param stream: Tops stream.
* @return topsatenStatus_t.
*/
// topsatenStatus_t TOPSATEN_EXPORT
// topsatenSum(topsatenTensor &output, const topsatenTensor &input,
//             const topsatenDataType_t &data_type, topsStream_t stream = nullptr);
void ggml_tops_op_sum(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
* @brief This function performs a addition computation.
*
* @param output: The output tensor.
* @param input: The input tensor.
* @param dimensions: The dimensions to reduce sum.
* @param keepdims: Whether the output tensor has dim retained or not.
* @param data_type: Data type.
* @param stream: Tops stream.
* @return topsatenStatus_t.
*/
// topsatenStatus_t TOPSATEN_EXPORT
// topsatenSum(topsatenTensor &output, const topsatenTensor &input,
//             const topsatenSize_t &dimensions, const bool keepdims,
//             const topsatenDataType_t &data_type, topsStream_t stream = nullptr);
void ggml_tops_op_sum_rows(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function returns the indices of the elements that are sorted.
 *
 * @param indices: The output indices tensor.
 * @param input: The input tensor.
 * @param stable: If True, the sorting algorithm will be stable.
 * @param dim: The dimension to sort along.
 * @param descending: Controls the sorting order (ascending or descending).
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenArgSort(topsatenTensor &indices,
//     const topsatenTensor &input,
//     bool stable, int64_t dim,
//     bool descending,
//     topsStream_t stream = nullptr);
void ggml_tops_op_argsort(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function gets the relative indices of maximum value for
 * the dimensions being reduced.
 *
 * @param output_idx: The output tensor of indices.
 * @param input: The input tensor.
 * @param dimensions: The dimensions to reduce.
 * @param keepdims: Whether the output tensor has dim retained or not.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT
// topsatenArgmax(topsatenTensor &output, const topsatenTensor &input,
//                const topsatenScalar_t &dimensions, const bool keepdims,
//                topsStream_t stream = nullptr);
void ggml_tops_op_argmax(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function returns a new tensor with the cosine of
 *        the elements of input.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenCos(topsatenTensor &output,
//     const topsatenTensor &input,
//     topsStream_t stream = nullptr);
void ggml_tops_op_cos(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
 * @brief This function returns a new tensor with the sine of
 *        the elements of input.
 *
 * @param output: The output tensor.
 * @param input: The input tensor.
 * @param stream: Tops stream.
 * @return topsatenStatus_t.
 */
// topsatenStatus_t TOPSATEN_EXPORT topsatenSin(topsatenTensor &output,
//     const topsatenTensor &input,
//     topsStream_t stream = nullptr);
void ggml_tops_op_sin(ggml_backend_tops_context & ctx, ggml_tensor * dst);

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
void ggml_tops_op_transpose_1d(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
* @brief This function returns a new tensor with the ln() of
*        the elements of input.
*
* @param output: The output tensor.
* @param input: The input tensor.
* @param stream: Tops stream.
* @return topsatenStatus_t.
*/
// topsatenStatus_t TOPSATEN_EXPORT topsatenLog(topsatenTensor &output,
//     const topsatenTensor &input,
//     topsStream_t stream = nullptr);
void ggml_tops_op_log(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
* @brief This function returns the mean value of input
*        in the dimension dims.
*
* @param output: The output tensor.
* @param input: The input tensor.
* @param dims_in: The dims to reduce.
* @param keepdims: Whether the output tensor has dim retained or no.
* @param data_type: The data type of the input and output tensor.
* @param stream: Tops stream.
* @return topsatenStatus_t.
*/
// topsatenStatus_t TOPSATEN_EXPORT topsatenMean(
//     topsatenTensor &output, const topsatenTensor &input,
//     const topsatenSize_t &dims_in, bool keepdims,
//     const topsatenDataType_t &data_type, topsStream_t stream = nullptr);
void ggml_tops_op_mean(ggml_backend_tops_context & ctx, ggml_tensor * dst);

/**
* @brief This function performs the pad function.
*
* @param output: The output tensor.
* @param input: The input tensor.
* @param pads: M-elements tuple, where m/2 <= input dimensions and m is even.
* @param mode: `constant`, `reflect`, `replicate` or `circular`. Default:
* `constant`.
* @param value: A scalar value to be used if the mode chosen is `constant`
*              (by default it is 0).
* @param stream: Tops stream.
* @return topsatenStatus_t.
*/
// topsatenStatus_t TOPSATEN_EXPORT topsatenPad(topsatenTensor &output,
//     const topsatenTensor &input,
//     const topsatenSize_t &pads,
//     const topsatenPadMode_t &mode,
//     const topsatenScalar_t &value,
//     topsStream_t stream = nullptr);
void ggml_tops_op_reflect_1d(ggml_backend_tops_context & ctx, ggml_tensor * dst);

// topsatenEq+topsatenSum
void ggml_tops_op_count_equal(ggml_backend_tops_context & ctx, ggml_tensor * dst);

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
void ggml_tops_op_flash_atten_ext(ggml_backend_tops_context & ctx, ggml_tensor * dst);
