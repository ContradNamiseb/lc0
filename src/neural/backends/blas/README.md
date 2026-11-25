The files in this directory comprise the BLAS backend of Lc0.

## Structure

As in the CUDA and SYCL backends, the network is a list of layers built once
from the weights (`layers.h`): `ConvLayer`, `ResidualBlock`, `AttentionBody`
with its `EncoderBlock`s, `AttentionPolicyHead`, `PolicyMapLayer`,
`ValueHead`, `DenseLayer` and `EmbeddingLayer`. A computation runs them in
order over three tensors and a workspace that the network pools and reuses.

## INT8 nets

Nets whose layers are stored as W8A8 INT8 (encoding `INT8` in `net.proto`)
run those layers in INT8 when lc0 is built with MKL: the residual blocks' 3x3
convolutions, and the Q, K, V, output and feed-forward projections of the
attention body. Each layer quantizes its input to uint8 with the scale and
zero point stored in the net and multiplies with `cblas_gemm_s8u8s32`.
`--backend-opts=int8=false`, the Eigen backend and other BLAS libraries run
the same nets in FP32 from the dequantized weights.

## License

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

**The source files of this directory are not covered by any additional
permission.**


