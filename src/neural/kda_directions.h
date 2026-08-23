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

#include "utils/exception.h"

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
inline int KdaSquareForToken(int direction, int token) {
  switch (direction) {
    case 1:
      return token;
    case 2:
      return 63 - token;
    case 3:
      return (token % 8) * 8 + token / 8;
    case 4: {
      const int reverse = 63 - token;
      return (reverse % 8) * 8 + reverse / 8;
    }
    case 5: {
      static constexpr int kTable[64] = {
          7,  6,  15, 5,  14, 23, 4,  13, 22, 31, 3,  12, 21, 30, 39, 2,
          11, 20, 29, 38, 47, 1,  10, 19, 28, 37, 46, 55, 0,  9,  18, 27,
          36, 45, 54, 63, 8,  17, 26, 35, 44, 53, 62, 16, 25, 34, 43, 52,
          61, 24, 33, 42, 51, 60, 32, 41, 50, 59, 40, 49, 58, 48, 57, 56};
      return kTable[token];
    }
    case 6: {
      static constexpr int kTable[64] = {
          56, 57, 48, 58, 49, 40, 59, 50, 41, 32, 60, 51, 42, 33, 24, 61,
          52, 43, 34, 25, 16, 62, 53, 44, 35, 26, 17, 8,  63, 54, 45, 36,
          27, 18, 9,  0,  55, 46, 37, 28, 19, 10, 1,  47, 38, 29, 20, 11,
          2,  39, 30, 21, 12, 3,  31, 22, 13, 4,  23, 14, 5,  15, 6,  7};
      return kTable[token];
    }
    case 7: {
      static constexpr int kTable[64] = {
          0,  1,  8,  2,  9,  16, 3,  10, 17, 24, 4,  11, 18, 25, 32, 5,
          12, 19, 26, 33, 40, 6,  13, 20, 27, 34, 41, 48, 7,  14, 21, 28,
          35, 42, 49, 56, 15, 22, 29, 36, 43, 50, 57, 23, 30, 37, 44, 51,
          58, 31, 38, 45, 52, 59, 39, 46, 53, 60, 47, 54, 61, 55, 62, 63};
      return kTable[token];
    }
    case 8: {
      static constexpr int kTable[64] = {
          63, 62, 55, 61, 54, 47, 60, 53, 46, 39, 59, 52, 45, 38, 31, 58,
          51, 44, 37, 30, 23, 57, 50, 43, 36, 29, 22, 15, 56, 49, 42, 35,
          28, 21, 14, 7,  48, 41, 34, 27, 20, 13, 6,  40, 33, 26, 19, 12,
          5,  32, 25, 18, 11, 4,  24, 17, 10, 3,  16, 9,  2,  8,  1,  0};
      return kTable[token];
    }
    default:
      throw Exception("Unsupported KDA traversal direction.");
  }
}

}  // namespace lczero
