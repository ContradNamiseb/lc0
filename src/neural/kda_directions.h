/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
#pragma once

namespace lczero {

// Traversal order of the 64 squares for each KdaDirection value. Must match
// KDA_TRAVERSALS in the trainer.
//
// Single definition on purpose. This started as two copies (BLAS backend and
// ONNX converter) on the reasoning that a small, stable, pure function is
// cheaper to duplicate than to share. That held right up until the OpenVINO
// backend needed it in two more places -- a third C++ copy and, worse, a
// hard-coded table inside an OpenCL kernel string, where a divergence would
// not even be a compile error, just silently wrong chess. The GPU table is
// now generated from this function at runtime rather than transcribed.
//
// All 8 directions as one flat table rather than 4 arithmetic cases plus 4
// per-case tables -- same shape as sycl/common_kernels.dp.cpp's
// kKdaDirectionOrder[16][64] (ba43374, "hoist KDA direction-order branch out
// of the per-token loop"), which found the branchless table read faster
// than re-deriving directions 1-4 by arithmetic on every one of a hot
// loop's 64 iterations. Verified there against the original branch chain,
// exact match across all 8 directions x 64 tokens.
inline constexpr int kKdaDirectionOrder[16][64] = {
    // direction 1: forward, rank-major (identity)
    {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
     32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
     48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63},
    // direction 2: reverse, rank-major
    {63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
     47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
     31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
     15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0},
    // direction 3: forward, file-major (transpose)
    {0,  8,  16, 24, 32, 40, 48, 56, 1,  9,  17, 25, 33, 41, 49, 57,
     2,  10, 18, 26, 34, 42, 50, 58, 3,  11, 19, 27, 35, 43, 51, 59,
     4,  12, 20, 28, 36, 44, 52, 60, 5,  13, 21, 29, 37, 45, 53, 61,
     6,  14, 22, 30, 38, 46, 54, 62, 7,  15, 23, 31, 39, 47, 55, 63},
    // direction 4: reverse, file-major (transpose)
    {63, 55, 47, 39, 31, 23, 15, 7,  62, 54, 46, 38, 30, 22, 14, 6,
     61, 53, 45, 37, 29, 21, 13, 5,  60, 52, 44, 36, 28, 20, 12, 4,
     59, 51, 43, 35, 27, 19, 11, 3,  58, 50, 42, 34, 26, 18, 10, 2,
     57, 49, 41, 33, 25, 17, 9,  1,  56, 48, 40, 32, 24, 16, 8,  0},
    // direction 5: diagonal, forward
    {7,  6,  15, 5,  14, 23, 4,  13, 22, 31, 3,  12, 21, 30, 39, 2,
     11, 20, 29, 38, 47, 1,  10, 19, 28, 37, 46, 55, 0,  9,  18, 27,
     36, 45, 54, 63, 8,  17, 26, 35, 44, 53, 62, 16, 25, 34, 43, 52,
     61, 24, 33, 42, 51, 60, 32, 41, 50, 59, 40, 49, 58, 48, 57, 56},
    // direction 6: diagonal, reverse
    {56, 57, 48, 58, 49, 40, 59, 50, 41, 32, 60, 51, 42, 33, 24, 61,
     52, 43, 34, 25, 16, 62, 53, 44, 35, 26, 17, 8,  63, 54, 45, 36,
     27, 18, 9,  0,  55, 46, 37, 28, 19, 10, 1,  47, 38, 29, 20, 11,
     2,  39, 30, 21, 12, 3,  31, 22, 13, 4,  23, 14, 5,  15, 6,  7},
    // direction 7: anti-diagonal, forward
    {0,  1,  8,  2,  9,  16, 3,  10, 17, 24, 4,  11, 18, 25, 32, 5,
     12, 19, 26, 33, 40, 6,  13, 20, 27, 34, 41, 48, 7,  14, 21, 28,
     35, 42, 49, 56, 15, 22, 29, 36, 43, 50, 57, 23, 30, 37, 44, 51,
     58, 31, 38, 45, 52, 59, 39, 46, 53, 60, 47, 54, 61, 55, 62, 63},
    // direction 8: anti-diagonal, reverse
    {63, 62, 55, 61, 54, 47, 60, 53, 46, 39, 59, 52, 45, 38, 31, 58,
     51, 44, 37, 30, 23, 57, 50, 43, 36, 29, 22, 15, 56, 49, 42, 35,
     28, 21, 14, 7,  48, 41, 34, 27, 20, 13, 6,  40, 33, 26, 19, 12,
     5,  32, 25, 18, 11, 4,  24, 17, 10, 3,  16, 9,  2,  8,  1,  0},
    // Boustrophedon ("serpentine") walks, directions 9-16. Every other
    // rank/file/diagonal is reversed, so consecutive tokens are always
    // board adjacent; the plain walks above jump 7 squares at each rank
    // wrap (12 for the diagonals), which the recurrence cannot represent --
    // it can only weight by recency. Materialised here rather than left as
    // the closed forms 9-12 admit, so every direction is one table read and
    // the GPU kernel's generated table stays uniform.
    // direction 9: rank serpentine
    {0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 15, 14, 13, 12, 11, 10, 9 , 8 ,
     16, 17, 18, 19, 20, 21, 22, 23, 31, 30, 29, 28, 27, 26, 25, 24,
     32, 33, 34, 35, 36, 37, 38, 39, 47, 46, 45, 44, 43, 42, 41, 40,
     48, 49, 50, 51, 52, 53, 54, 55, 63, 62, 61, 60, 59, 58, 57, 56},
    // direction 10: rank serpentine, reverse
    {56, 57, 58, 59, 60, 61, 62, 63, 55, 54, 53, 52, 51, 50, 49, 48,
     40, 41, 42, 43, 44, 45, 46, 47, 39, 38, 37, 36, 35, 34, 33, 32,
     24, 25, 26, 27, 28, 29, 30, 31, 23, 22, 21, 20, 19, 18, 17, 16,
     8 , 9 , 10, 11, 12, 13, 14, 15, 7 , 6 , 5 , 4 , 3 , 2 , 1 , 0 },
    // direction 11: file serpentine
    {0 , 8 , 16, 24, 32, 40, 48, 56, 57, 49, 41, 33, 25, 17, 9 , 1 ,
     2 , 10, 18, 26, 34, 42, 50, 58, 59, 51, 43, 35, 27, 19, 11, 3 ,
     4 , 12, 20, 28, 36, 44, 52, 60, 61, 53, 45, 37, 29, 21, 13, 5 ,
     6 , 14, 22, 30, 38, 46, 54, 62, 63, 55, 47, 39, 31, 23, 15, 7 },
    // direction 12: file serpentine, reverse
    {7 , 15, 23, 31, 39, 47, 55, 63, 62, 54, 46, 38, 30, 22, 14, 6 ,
     5 , 13, 21, 29, 37, 45, 53, 61, 60, 52, 44, 36, 28, 20, 12, 4 ,
     3 , 11, 19, 27, 35, 43, 51, 59, 58, 50, 42, 34, 26, 18, 10, 2 ,
     1 , 9 , 17, 25, 33, 41, 49, 57, 56, 48, 40, 32, 24, 16, 8 , 0 },
    // direction 13: diagonal serpentine
    {7 , 15, 6 , 5 , 14, 23, 31, 22, 13, 4 , 3 , 12, 21, 30, 39, 47,
     38, 29, 20, 11, 2 , 1 , 10, 19, 28, 37, 46, 55, 63, 54, 45, 36,
     27, 18, 9 , 0 , 8 , 17, 26, 35, 44, 53, 62, 61, 52, 43, 34, 25,
     16, 24, 33, 42, 51, 60, 59, 50, 41, 32, 40, 49, 58, 57, 48, 56},
    // direction 14: diagonal serpentine, reverse
    {56, 48, 57, 58, 49, 40, 32, 41, 50, 59, 60, 51, 42, 33, 24, 16,
     25, 34, 43, 52, 61, 62, 53, 44, 35, 26, 17, 8 , 0 , 9 , 18, 27,
     36, 45, 54, 63, 55, 46, 37, 28, 19, 10, 1 , 2 , 11, 20, 29, 38,
     47, 39, 30, 21, 12, 3 , 4 , 13, 22, 31, 23, 14, 5 , 6 , 15, 7 },
    // direction 15: anti-diagonal serpentine
    {0 , 8 , 1 , 2 , 9 , 16, 24, 17, 10, 3 , 4 , 11, 18, 25, 32, 40,
     33, 26, 19, 12, 5 , 6 , 13, 20, 27, 34, 41, 48, 56, 49, 42, 35,
     28, 21, 14, 7 , 15, 22, 29, 36, 43, 50, 57, 58, 51, 44, 37, 30,
     23, 31, 38, 45, 52, 59, 60, 53, 46, 39, 47, 54, 61, 62, 55, 63},
    // direction 16: anti-diagonal serpentine, reverse
    {63, 55, 62, 61, 54, 47, 39, 46, 53, 60, 59, 52, 45, 38, 31, 23,
     30, 37, 44, 51, 58, 57, 50, 43, 36, 29, 22, 15, 7 , 14, 21, 28,
     35, 42, 49, 56, 48, 41, 34, 27, 20, 13, 6 , 5 , 12, 19, 26, 33,
     40, 32, 25, 18, 11, 4 , 3 , 10, 17, 24, 16, 9 , 2 , 1 , 8 , 0 },
};

// Direction 9-16 materialisation: the table is the single source of truth.
// Out-of-range input is clamped (not thrown) so this stays usable from SYCL /
// OpenCL device code, which cannot throw. Callers that need validation should
// check the range themselves before calling.
inline int KdaSquareForToken(int direction, int token) {
  if (direction < 1) direction = 1;
  if (direction > 16) direction = 16;
  return kKdaDirectionOrder[direction - 1][token];
}

}  // namespace lczero
