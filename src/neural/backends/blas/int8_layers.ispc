/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2026 The LCZero Authors

 Leela Chess is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 Leela Chess is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
 */

// q = clamp(rint(x * inverse_scale + zero_point), 0, 255).
export void QuantizeToUint8(uniform const size_t count,
                            const uniform float input[],
                            uniform const float inverse_scale,
                            uniform const float zero_point,
                            uniform unsigned int8 output[]) {
  foreach (i = 0 ... count) {
    const float value =
        clamp(input[i] * inverse_scale + zero_point, 0.0f, 255.0f);
    output[i] = (unsigned int8)(int)round(value);
  }
}

// output[r][o] = scales[o] * (accumulators[r][o] - corrections[o]).
export void DequantizeRows(uniform const size_t rows,
                           uniform const size_t outputs,
                           const uniform int32 accumulators[],
                           const uniform float scales[],
                           const uniform int32 corrections[],
                           uniform float output[]) {
  for (uniform size_t row = 0; row < rows; row++) {
    foreach (i = 0 ... outputs) {
      const uniform size_t base = row * outputs;
      output[base + i] =
          scales[i] * (float)(accumulators[base + i] - corrections[i]);
    }
  }
}
