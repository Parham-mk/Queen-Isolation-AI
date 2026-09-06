<div align="center">
  <h1> Queen's Isolation: AI Adversarial Agent</h1>
  <p><strong>A high-performance, intelligent game-playing agent for a 7×7 strategy board game.</strong></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Artificial%20Intelligence-FF6F00?style=for-the-badge&logo=openai&logoColor=white" alt="AI">
  </p>
</div>

---

Welcome to **Queen's Isolation**, an advanced adversarial game-playing AI. This project brings to life classical artificial intelligence search algorithms, wrapped in rigorous engineering practices and data-driven performance optimizations, to conquer a highly dynamic board game.

##  The Game: Queen's Isolation

Imagine chess, but every step leaves the board crumbling behind you. 

-  **The Queens**: Two players control one piece each on a 7×7 board, moving exactly like chess queens (horizontally, vertically, diagonally, any number of squares).
-  **The Crumbling Board**: Moving off a square permanently **blocks** it. No piece can pass through or land on it ever again.
-  **The Push**: A player can **push** the opponent's queen one square further in the direction of movement.
-  **Victory Condition**: You win if you push the opponent off the board, or if you strategically trap them so they have absolutely zero legal moves left.

---

##  AI Architecture & Algorithms

This agent is engineered to squeeze the maximum possible depth out of the game tree within strict, unforgiving time limits. It never misses a timeout.

| Concept | Implementation Details |
|---------|------------------------|
| **Minimax Search** | The backbone of the agent, guaranteeing mathematically optimal decision-making when the entire game tree is visible. |
| **Alpha-Beta Pruning** | Exponentially reduces the search space by cutting off branches that can't influence the final decision. |
| **Iterative Deepening** | Progressively searches deeper levels. If the timer runs out mid-search, the agent safely falls back to the best move found in the previous depth. |
| **Move Ordering** | Strategically evaluates **push moves** and **central moves** first. This dramatically increases the frequency of Alpha-Beta cutoffs. |
| **Exception-Based Time Management** | Aborts deep recursive calls instantly and safely when the clock hits a configurable 50ms safety margin. |

---

##  Performance Benchmarks

Through rigorous local tournaments (1000ms time limit per move), this agent thoroughly dominates random baselines. Surprisingly, our data proved that **raw search depth > heuristic sophistication**. A simple, ultra-fast mobility heuristic beat a complex centrality-based heuristic because it allowed the agent to search one ply deeper.

### Win Rate Highlights 

| Agent (Player 1) | Opponent (Player 2) | Win Rate |
|---|---|---:|
| **My Agent** | Random Baseline | **100%** |
| Random Baseline | **My Agent** | **80%** |
| **Agent (Mobility Eval)** | Agent (Centrality Eval) | **60%** |

*Combined win rate against random play: **90%**.*

---

##  Getting Started

Want to see the AI in action or run your own tournaments? 

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run the Test Suite

The project includes 22 rigorous unit tests covering timeout safety and logical correctness.
```bash
python -m unittest tests/test_search.py -v
```

### 3. Run a Tournament Benchmark

Measure the agent's speed and win rate against different heuristics:
```bash
python benchmarks/tournament.py
```

### 4. Play a Match

Watch two agents go head-to-head!
```bash
python player_submission_tests.py
```

---

##  Repository Guide

Navigate the codebase with ease:

-  `player_submission.py` — The core agent logic (Minimax, Alpha-Beta, Iterative Deepening, Heuristics).
-  `game.py` — The game engine and board mechanics.
-  `tests/test_search.py` — The comprehensive automated test suite.
-  `benchmarks/tournament.py` — Automated tournament scripts for data collection.
-  `docs/algorithm.md` — A deep dive into the algorithm design, math, and complexity.
-  `docs/results.md` — Detailed benchmark results and analysis.

