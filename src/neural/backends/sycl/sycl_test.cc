#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>
#include <type_traits>
#include "neural/backends/sycl/sycl_common.h"
#include "neural/backends/sycl/kernels.h"

namespace lczero {
namespace sycldnn_backend {
namespace {

// Tests call the production decision function from sycl_common.h directly --
// the old copy here hard-coded its own constants (256 / 49152) that had
// already drifted from the real limits (kMaxResBlockFusingChannels = 384,
// kMaxResBlockFusingSeFp16AmpereSmem = 72 * 512 * sizeof(half)), so the
// tests could never catch a change in the real allowFusing computation.

// 1. Positional Encoding conversion count tests
TEST(SyclBackendTest, PosEncodingConversionCountCalculation) {
  constexpr int kNumPosEncodingChannels = 32;
  const size_t element_count = 64 * kNumPosEncodingChannels; // 2048

  // FP32 sizing
  const size_t float_dest_byte_count = element_count * sizeof(float);
  EXPECT_EQ(element_count, 2048u);
  EXPECT_EQ(float_dest_byte_count, 8192u);

  // FP16 sizing
  const size_t half_dest_byte_count = element_count * sizeof(sycl::half);
  EXPECT_EQ(half_dest_byte_count, 4096u);

  // Source byte count for memcpy from kPosEncoding float array
  const size_t float_src_byte_count = element_count * sizeof(float);
  EXPECT_EQ(float_src_byte_count, 8192u);
}

// 2. FP16 work-group limit fallback test
TEST(SyclBackendTest, Fp16ResidualBlockWgLimitFallback) {
  const size_t shared_mem_size = 65536; // 64 KB
  const size_t max_work_group_size = 256;

  // Case A: C = 512 > max_work_group_size (256) for FP16 network.
  // Must return false so that both intermediate and final residual blocks fall back to OutputTransform.
  bool allow_fusing_over_limit = AllowSubGroupFusing(
      /*C=*/512, /*se_k=*/32, shared_mem_size, max_work_group_size, /*is_fp16=*/true);
  EXPECT_FALSE(allow_fusing_over_limit);
}

// 3. Subgroup path selection test
TEST(SyclBackendTest, SubGroupPathSelectionWhenSupported) {
  const size_t shared_mem_size = 65536; // 64 KB
  const size_t max_work_group_size = 256;

  // Case B: C = 256 <= max_work_group_size (256) and se_k (32) <= C for FP16 network.
  // Must return true to select the optimized subgroup path.
  bool allow_fusing_supported = AllowSubGroupFusing(
      /*C=*/256, /*se_k=*/32, shared_mem_size, max_work_group_size, /*is_fp16=*/true);
  EXPECT_TRUE(allow_fusing_supported);
}

// 4. Shape contract violation test (se_k > C)
TEST(SyclBackendTest, SeShapeContractEnforcement) {
  const size_t shared_mem_size = 65536;
  const size_t max_work_group_size = 256;

  // Case C: se_k (128) > C (64).
  // Must return false to avoid out-of-bounds local memory access.
  bool allow_fusing_invalid_se = AllowSubGroupFusing(
      /*C=*/64, /*se_k=*/128, shared_mem_size, max_work_group_size, /*is_fp16=*/true);
  EXPECT_FALSE(allow_fusing_invalid_se);
}

} // namespace
} // namespace sycldnn_backend
} // namespace lczero
