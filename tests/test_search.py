#!/usr/bin/env python
"""Comprehensive test suite for the AI game-playing agent.

Tests cover:
  - OpenMoveEvalFn correctness
  - CustomEvalFn correctness and feature verification
  - Minimax correctness at multiple depths
  - Alpha-Beta correctness and equivalence to Minimax
  - Timeout / time-management safety
  - Legal-move compliance under iterative deepening
  - Terminal state handling
  - Edge cases (one move, no moves, push scenarios)
"""
import unittest
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game import Board
from test_players import RandomPlayer
from player_submission import (
    CustomPlayer, OpenMoveEvalFn, CustomEvalFn, SearchTimeout
)


# ── helpers ────────────────────────────────────────────────────────── #

class _DummyPlayer:
    """Minimal player stub so Board can be constructed."""
    def __init__(self, name="Dummy"):
        self.name = name
    def move(self, game, legal_moves, time_left):
        return legal_moves[0] if legal_moves else None


def _make_board(p1_pos=(0, 0), p2_pos=(6, 6)):
    """Create a 7×7 Board with queens already placed."""
    p1, p2 = _DummyPlayer("P1"), _DummyPlayer("P2")
    b = Board(p1, p2, 7, 7)
    b.__apply_move__((p1_pos[0], p1_pos[1], False))  # P1
    b.__apply_move__((p2_pos[0], p2_pos[1], False))  # P2
    return b


def _unlimited_time():
    return 100_000


def _low_time():
    return 10


# ── Evaluation function tests ─────────────────────────────────────── #

class TestOpenMoveEvalFn(unittest.TestCase):
    def test_returns_float(self):
        b = _make_board()
        score = OpenMoveEvalFn().score(b, True)
        self.assertIsInstance(score, float)

    def test_symmetric_sign(self):
        """Maximising and minimising perspectives should be negatives."""
        b = _make_board()
        ef = OpenMoveEvalFn()
        self.assertEqual(ef.score(b, True), -ef.score(b, False))

    def test_no_moves_scores_negative(self):
        """If active player has no moves, score should be ≤ 0."""
        # Construct a board where the active player is boxed in.
        p1, p2 = _DummyPlayer("P1"), _DummyPlayer("P2")
        b = Board(p1, p2, 3, 3)  # tiny board
        # Fill everything except one cell
        for r in range(3):
            for c in range(3):
                b.__board_state__[r][c] = Board.BLOCKED
        b.__board_state__[0][0] = Board.BLANK
        b.__board_state__[2][2] = Board.BLANK
        # Place queens
        b.__last_queen_move__[b.__queen_1__] = (0, 0, False)
        b.__last_queen_move__[b.__queen_2__] = (2, 2, False)
        b.__board_state__[0][0] = "Q1"
        b.__board_state__[2][2] = "Q2"
        b.move_count = 2
        # Both should have 0 moves (blocked in)
        score = OpenMoveEvalFn().score(b, True)
        self.assertLessEqual(score, 0.0)


class TestCustomEvalFn(unittest.TestCase):
    def test_returns_float(self):
        b = _make_board()
        score = CustomEvalFn().score(b, True)
        self.assertIsInstance(score, float)

    def test_symmetric_sign(self):
        b = _make_board()
        ef = CustomEvalFn()
        self.assertEqual(ef.score(b, True), -ef.score(b, False))

    def test_central_position_preferred(self):
        """A queen in the center should score higher than one in the corner,
        all else being equal."""
        b_center = _make_board(p1_pos=(3, 3), p2_pos=(0, 0))
        b_corner = _make_board(p1_pos=(0, 0), p2_pos=(3, 3))
        ef = CustomEvalFn()
        self.assertGreater(ef.score(b_center, True),
                           ef.score(b_corner, True))


# ── Minimax tests ─────────────────────────────────────────────────── #

class TestMinimax(unittest.TestCase):
    def setUp(self):
        self.agent = CustomPlayer(eval_fn=OpenMoveEvalFn())

    def test_depth_1(self):
        b = _make_board()
        move, val = self.agent.minimax(b, _unlimited_time, depth=1)
        self.assertIsNotNone(move)
        self.assertIn(move, b.get_legal_moves())

    def test_depth_2(self):
        b = _make_board()
        move, val = self.agent.minimax(b, _unlimited_time, depth=2)
        self.assertIsNotNone(move)
        self.assertIn(move, b.get_legal_moves())

    def test_forced_single_move(self):
        """When only one legal move exists, minimax must return it."""
        p1, p2 = _DummyPlayer("P1"), _DummyPlayer("P2")
        b = Board(p1, p2, 3, 3)
        # Heavily block the board so P1 has exactly one move
        for r in range(3):
            for c in range(3):
                b.__board_state__[r][c] = Board.BLOCKED
        b.__board_state__[0][0] = "Q1"
        b.__board_state__[0][1] = Board.BLANK
        b.__board_state__[2][2] = "Q2"
        b.__last_queen_move__[b.__queen_1__] = (0, 0, False)
        b.__last_queen_move__[b.__queen_2__] = (2, 2, False)
        b.move_count = 2

        legal = b.get_legal_moves()
        self.assertEqual(len(legal), 1, "Board should have exactly 1 legal move")
        move, _ = self.agent.minimax(b, _unlimited_time, depth=3)
        self.assertEqual(move, legal[0])

    def test_no_legal_moves(self):
        """When no legal moves exist, minimax returns None move."""
        p1, p2 = _DummyPlayer("P1"), _DummyPlayer("P2")
        b = Board(p1, p2, 3, 3)
        for r in range(3):
            for c in range(3):
                b.__board_state__[r][c] = Board.BLOCKED
        b.__board_state__[0][0] = "Q1"
        b.__board_state__[2][2] = "Q2"
        b.__last_queen_move__[b.__queen_1__] = (0, 0, False)
        b.__last_queen_move__[b.__queen_2__] = (2, 2, False)
        b.move_count = 2
        move, _ = self.agent.minimax(b, _unlimited_time, depth=3)
        self.assertIsNone(move)


# ── Alpha-Beta tests ──────────────────────────────────────────────── #

class TestAlphaBeta(unittest.TestCase):
    def setUp(self):
        self.agent = CustomPlayer(eval_fn=OpenMoveEvalFn())

    def test_depth_1(self):
        b = _make_board()
        move, val = self.agent.alphabeta(b, _unlimited_time, depth=1)
        self.assertIsNotNone(move)
        self.assertIn(move, b.get_legal_moves())

    def test_depth_3(self):
        b = _make_board()
        move, val = self.agent.alphabeta(b, _unlimited_time, depth=3)
        self.assertIsNotNone(move)
        self.assertIn(move, b.get_legal_moves())


# ── Minimax / Alpha-Beta equivalence ──────────────────────────────── #

class TestSearchEquivalence(unittest.TestCase):
    """Minimax and Alpha-Beta must return the same utility for
    the same state / depth / evaluation function."""

    def test_equivalence_depth_1(self):
        agent = CustomPlayer(eval_fn=OpenMoveEvalFn())
        b = _make_board()
        _, mm_val = agent.minimax(b, _unlimited_time, depth=1)
        _, ab_val = agent.alphabeta(b, _unlimited_time, depth=1)
        self.assertAlmostEqual(mm_val, ab_val, places=6)

    def test_equivalence_depth_2(self):
        agent = CustomPlayer(eval_fn=OpenMoveEvalFn())
        b = _make_board()
        _, mm_val = agent.minimax(b, _unlimited_time, depth=2)
        _, ab_val = agent.alphabeta(b, _unlimited_time, depth=2)
        self.assertAlmostEqual(mm_val, ab_val, places=6)

    def test_equivalence_depth_3(self):
        agent = CustomPlayer(eval_fn=OpenMoveEvalFn())
        b = _make_board((3, 3), (0, 0))
        _, mm_val = agent.minimax(b, _unlimited_time, depth=3)
        _, ab_val = agent.alphabeta(b, _unlimited_time, depth=3)
        self.assertAlmostEqual(mm_val, ab_val, places=6)


# ── Timeout tests ─────────────────────────────────────────────────── #

class TestTimeout(unittest.TestCase):
    def test_minimax_raises_on_low_time(self):
        agent = CustomPlayer(eval_fn=OpenMoveEvalFn())
        b = _make_board()
        with self.assertRaises(SearchTimeout):
            agent.minimax(b, _low_time, depth=5)

    def test_alphabeta_raises_on_low_time(self):
        agent = CustomPlayer(eval_fn=OpenMoveEvalFn())
        b = _make_board()
        with self.assertRaises(SearchTimeout):
            agent.alphabeta(b, _low_time, depth=5)

    def test_move_returns_legal_under_pressure(self):
        """Even under extreme time pressure, move() must return a
        legal move (never crash)."""
        agent = CustomPlayer(eval_fn=OpenMoveEvalFn())
        b = _make_board()
        legal = b.get_legal_moves()

        start = time.time()
        def tight_time():
            elapsed_ms = (time.time() - start) * 1000
            return max(0, 200 - elapsed_ms)

        result = agent.move(b, legal, tight_time)
        self.assertIn(result, legal)


# ── Iterative-deepening / agent-move tests ─────────────────────────── #

class TestAgentMove(unittest.TestCase):
    def test_always_legal(self):
        """Run 20 random boards and verify move() always returns a
        legal move."""
        import random
        random.seed(42)

        for _ in range(20):
            r1, c1 = random.randint(0, 6), random.randint(0, 6)
            r2, c2 = random.randint(0, 6), random.randint(0, 6)
            while (r1, c1) == (r2, c2):
                r2, c2 = random.randint(0, 6), random.randint(0, 6)

            b = _make_board(p1_pos=(r1, c1), p2_pos=(r2, c2))
            legal = b.get_legal_moves()
            if not legal:
                continue
            agent = CustomPlayer(eval_fn=CustomEvalFn())
            # Give each board 500ms so the test completes quickly
            start = time.time()
            def _time_left(s=start):
                return max(0, 500 - (time.time() - s) * 1000)
            result = agent.move(b, legal, _time_left)
            self.assertIn(result, legal,
                          f"Illegal move {result} on board with "
                          f"P1@({r1},{c1}), P2@({r2},{c2})")

    def test_single_legal_move_fast(self):
        """When there is exactly one legal move, move() should be
        near-instant and return it."""
        p1, p2 = _DummyPlayer("P1"), _DummyPlayer("P2")
        b = Board(p1, p2, 3, 3)
        for r in range(3):
            for c in range(3):
                b.__board_state__[r][c] = Board.BLOCKED
        b.__board_state__[0][0] = "Q1"
        b.__board_state__[0][1] = Board.BLANK
        b.__board_state__[2][2] = "Q2"
        b.__last_queen_move__[b.__queen_1__] = (0, 0, False)
        b.__last_queen_move__[b.__queen_2__] = (2, 2, False)
        b.move_count = 2

        legal = b.get_legal_moves()
        agent = CustomPlayer(eval_fn=OpenMoveEvalFn())
        t0 = time.time()
        result = agent.move(b, legal, _unlimited_time)
        elapsed = time.time() - t0
        self.assertEqual(result, legal[0])
        self.assertLess(elapsed, 0.5, "Single-move should be near-instant")


# ── Full-game smoke test ──────────────────────────────────────────── #

class TestFullGame(unittest.TestCase):
    def test_agent_vs_random(self):
        """CustomPlayer should beat RandomPlayer without crashing."""
        import random
        random.seed(12345)

        agent = CustomPlayer(eval_fn=CustomEvalFn())
        rng = RandomPlayer()
        b = Board(agent, rng, 7, 7)
        winner, _, termination = b.play_isolation(
            time_limit=5000, print_moves=False)
        # Just verify it finished without errors
        self.assertIsNotNone(winner)
        self.assertIsNotNone(termination)

    def test_agent_vs_random_as_p2(self):
        """Same but CustomPlayer is player 2."""
        import random
        random.seed(99999)

        agent = CustomPlayer(eval_fn=CustomEvalFn())
        rng = RandomPlayer()
        b = Board(rng, agent, 7, 7)
        winner, _, termination = b.play_isolation(
            time_limit=5000, print_moves=False)
        self.assertIsNotNone(winner)
        self.assertIsNotNone(termination)


if __name__ == '__main__':
    unittest.main()
