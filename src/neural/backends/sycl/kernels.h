#pragma once

#include <sycl/sycl.hpp>
#include "sycl_common.h"
#include "neural/backends/shared/activation.h"

namespace lczero {
namespace sycldnn_backend {

/**
 * @brief Adds two vectors element-wise with optional activation function.
 */
template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize,
                ActivationFunction activation, sycl::queue &sycl_queue);

/**
 * @brief Adds two vectors of equal size, transposing HNC to NHC layout.
 */
template <typename T>
void addVectorsHNC_NHC(T* a, T* b, int N, int H, int C, sycl::queue &sycl_queue);

/**
 * @brief Adds bias to innermost dimension for batched GEMMs with optional activation.
 */
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, ActivationFunction activation, sycl::queue &sycl_queue);

/**
 * @brief Adds bias to innermost dimension for batched GEMMs with custom N stride.
 */
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, int Nstride, ActivationFunction activation, sycl::queue &sycl_queue);

/**
 * @brief Adds bias to NCHW convolution output.
 */
template <typename T>
void addBias_NCHW(T* c, T* a, T* b, int N, int C, int H, int W,
                  ActivationFunction activation, sycl::queue &sycl_queue);

/**
 * @brief Converts tensor layout from NCHW to NHWC with datatype conversion.
 */
template <typename DstType, typename SrcType>
void convertNCHWtoNHWC(DstType* output_tensor, const SrcType* input_tensor,
                       int Nin, int Cin, int Nout, int Cout, int H, int W, sycl::queue &sycl_queue);

/**
 * @brief Data-type conversion without layout change (e.g. float to sycl::half).
 */
template <typename DstType, typename SrcType>
void copyTypeConverted(DstType* op, SrcType* ip, int N, sycl::queue &sycl_queue);

/**
 * @brief Batch normalization kernel.
 */
template <typename T>
void batchNorm(T* output, const T* input, const T* skipInput, int N, int C,
               int H, int W, float* means, float* var_multipliers,
               ActivationFunction activation, sycl::queue &sycl_queue);

/**
 * @brief Unpacks input planes into FP32 NCHW format.
 */
void expandPlanes_Fp32_NCHW(float* output, const uint64_t* masks,
                            const float* values, int n, sycl::queue &sycl_queue);

/**
 * @brief Unpacks input planes into FP16 NHWC format.
 */
void expandPlanes_Fp16_NHWC(sycl::half* output, const uint64_t* masks,
                            const float* values, int n, sycl::queue &sycl_queue);

/**
 * @brief Unpacks input planes into FP16 NCHW format.
 */
void expandPlanes_Fp16_NCHW(sycl::half* output, const uint64_t* masks,
                            const float* values, int n, sycl::queue &sycl_queue);

/**
 * @brief Computes global average pooling across HxW planes.
 */
template <typename T>
void globalAvgPool(int N, int C, T* output, const T* input,
                   const T* prevLayerBias, bool nhwc, sycl::queue &sycl_queue);

/**
 * @brief Applies global scaling vector to feature map.
 */
template <typename T>
void globalScale(int N, int C, T* output, const T* input, const T* scaleBias,
                 const T* prevLayerBias, bool nhwc,
                 ActivationFunction activation, sycl::queue &sycl_queue);

/**
 * @brief Fused Squeeze-and-Excitation kernel for FP16 NHWC layout.
 */
bool Se_Fp16_NHWC(int N, int C, int numFc1Out, sycl::half* output,
                  const sycl::half* skip, const sycl::half* input,
                  const sycl::half* w1, const sycl::half* b1,
                  const sycl::half* w2, const sycl::half* b2,
                  const sycl::half* bPrev, ActivationFunction activation, sycl::queue &sycl_queue);

/**
 * @brief Maps move indices from policy feature representation to policy vector output.
 */
template <typename T>
void PolicyMap(int N, T* output, const T* input, const short* indices,
               int inputSize, int usedSize, int outputSize, sycl::queue &sycl_queue);

/**
 * @brief Winograd filter transform for 3x3 convolutions.
 */
template <typename T>
void FilterTransform(int N, int C, T* transformedFilter, const T* filter, sycl::queue &sycl_queue);

/**
 * @brief Winograd input transform for 3x3 convolutions.
 */
template <typename T, bool nhcw>
void InputTransform(int N, int C, T* transformedInput, const T* input, sycl::queue &sycl_queue);

/**
 * @brief Winograd output transform.
 */
template <typename T, bool use_se, ActivationFunction activation, bool use_bias,
          bool use_skip, bool skipInput_nhcw, bool output_nhcw>
void OutputTransform(int N, int C, int se_K, T* output, const T* input,
                     const T* skip, const T* bias, const T* w1, const T* b1,
                     const T* w2, const T* b2, sycl::queue &sycl_queue);

/**
 * @brief Fused Winograd output and next-layer input transform.
 */
template <typename T, bool use_se, ActivationFunction activation, bool use_bias,
          bool use_skip>
void OutputInputTransform(int N, int C, int se_K, T* output, const T* input,
                          const T* skip, const T* bias, const T* w1,
                          const T* b1, const T* w2, const T* b2, sycl::queue &sycl_queue);

/**
 * @brief Softmax activation kernel over feature channels.
 */
template <typename T>
void Softmax(int N, int C, T* output, const T* input, const T* input2, sycl::queue &sycl_queue);

/**
 * @brief Layer normalization kernel.
 */
template <typename T>
void LayerNorm(int N, int C, T* output, const T* input, const T* bias,
               const T* skip, const T* gammas, const T* betas, float ep,
               float alpha, ActivationFunction act, sycl::queue &sycl_queue);

/**
 * @brief Computes promotion logits for attention policy head.
 */
template <typename T>
void ComputePromotionLogits(int N, int C, T* output, const T* keys,
                            const T* ppo, const T* policy_attn_logits, sycl::queue &sycl_queue);

/**
 * @brief Preprocesses attention body input embedding.
 */
template <typename T>
void inputPreprocessForAttentionBody(T* output, const T* input,
                                     const T* encoding, int N, int input_size,
                                     int encoding_size,
                                     bool is_pe_dense_embedding,
                                     sycl::queue &sycl_queue);

/**
 * @brief Applies multiplicative and additive input gating.
 */
template <typename T>
void applyInputGating(T* output, const T* input, const T* mult, const T* add,
                      int N, int HW, int C, sycl::queue &sycl_queue);

/**
 * @brief Generates strided offset pointers for Multi-Head Attention operations.
 */
template <typename T>
void genOffsetPointers(T** offsets, int heads, int max_batch, int depth,
                       int d_model, T* k, T* q, T* b1, T* v, T* b2, sycl::queue &sycl_queue);

}  // namespace sycldnn_backend
}  // namespace lczero
