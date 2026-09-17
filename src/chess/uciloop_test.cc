/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

// Regression coverage for the agora #866 P1-4 finding: a search that
// aborted before any playout ever ran (a worker exception on the very
// first NN batch, review #863/#864) leaves final_bestmove_ at its default-
// constructed Move() -- StringUciResponder::OutputBestMove must serialize
// that as the UCI null move "0000", not let Move::ToString() print the
// syntactically legal-looking but meaningless "a1a1", which a GUI could
// read as a real move and forfeit the game on.

#include "chess/uciloop.h"

#include "gtest/gtest.h"

namespace lczero {
namespace {

// Captures every response instead of printing it, so the test can inspect
// exactly what would have gone out over UCI.
class CapturingUciResponder : public StringUciResponder {
 public:
  void SendRawResponses(const std::vector<std::string>& responses) override {
    for (const auto& r : responses) captured_.push_back(r);
  }
  const std::vector<std::string>& captured() const { return captured_; }

 private:
  std::vector<std::string> captured_;
};

TEST(OutputBestMove, DefaultMoveSerializesAsUciNullMoveNotA1A1) {
  CapturingUciResponder responder;
  // Deliberately not calling PopulateParams(): mirrors a responder used
  // before/without a live OptionsParser, and keeps IsChess960() at its
  // documented null-safe default (false).
  BestMoveInfo info{Move(), Move()};
  ASSERT_TRUE(info.bestmove.is_null());
  responder.OutputBestMove(&info);
  ASSERT_EQ(responder.captured().size(), 1u);
  EXPECT_EQ(responder.captured()[0], "bestmove 0000");
}

TEST(OutputBestMove, RealMoveStillSerializesNormally) {
  CapturingUciResponder responder;
  BestMoveInfo info{Move::White(Square::Parse("e2"), Square::Parse("e4")),
                    Move()};
  responder.OutputBestMove(&info);
  ASSERT_EQ(responder.captured().size(), 1u);
  EXPECT_EQ(responder.captured()[0], "bestmove e2e4");
}

}  // namespace
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
