// Quick raw-eval probe: compares sycl-fp16 vs blas on the real net.
#define main OriginalKdaTestMain
#include "neural/kda_parity_test.cc"
#undef main
#include <iostream>

int main(int argc, char** argv) {
  using namespace lczero;
  const char* path = std::getenv("LC0_TEST_REAL_NET");
  if (!path) { std::cerr << "set LC0_TEST_REAL_NET\n"; return 2; }
  auto weights = LoadWeights(path);
  if (!weights) { std::cerr << "cannot load\n"; return 2; }
  const char* fens[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4",
    "8/6k1/8/8/2P5/8/2K3P1/8 w - - 0 1",
  };
  for (const char* be : {"sycl", "sycl-fp16"}) {
    try {
      for (int i = 0; i < 3; ++i) {
        ChessBoard board;
        board.SetFromFen(fens[i]);
        PositionHistory history;
        history.Reset(board, 0, 1);
        auto input = EncodePositionForNN(
            weights->format().network_format().input(), history, 8,
            FillEmptyHistory::NO, nullptr);
        std::vector<InputPlanes> batch = {InputPlanes(input)};
        auto ref = RunNetwork("blas", *weights, batch)[0];
        auto got = RunNetwork(be, *weights, batch)[0];
        float worst = 0, scale = 0;
        int argmax_r = 0, argmax_g = 0;
        for (int p = 0; p < 1858; ++p) {
          worst = std::max(worst, std::fabs(got.policy[p] - ref.policy[p]));
          scale = std::max(scale, std::fabs(ref.policy[p]));
          if (ref.policy[p] > ref.policy[argmax_r]) argmax_r = p;
          if (got.policy[p] > got.policy[argmax_g]) argmax_g = p;
        }
        std::cout << be << " pos" << i << ": q=" << ref.q << "/" << got.q
                  << " (abs " << std::fabs(got.q - ref.q) << ")"
                  << " d=" << ref.d << "/" << got.d
                  << " policy_abs=" << worst << " rel=" << (worst / scale)
                  << " argmax " << argmax_g << "/" << argmax_r << "\n";
      }
    } catch (const Exception& e) {
      std::cout << be << ": failed: " << e.what() << "\n";
    }
  }
  return 0;
}
