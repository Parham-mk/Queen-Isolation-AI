# Algorithm Documentation

## Game-State Representation

The game state is represented by the `Board` class in `game.py`. Key elements:

- **Grid**: 7×7 array where each cell is `BLANK`, `BLOCKED`, or occupied by a queen (`Q1`/`Q2`).
- **Queen positions**: Tracked via `__last_queen_move__` as `(row, col, was_push)` tuples.
- **Active/Inactive player**: The board alternates after each move.
- **Move format**: `(row, col, is_push)` — a 3-tuple indicating destination and whether a push occurs.

When a queen leaves a square, that square becomes permanently `BLOCKED`. This creates an ever-shrinking board that drives the game toward a terminal state.

## Branching Factor

On an empty 7×7 board, a centrally placed queen has up to ~24 legal moves (8 directions × ~3 squares average). As the board fills with blocked squares, the branching factor decreases. Typical mid-game branching factor: **15–25**.

## Search Algorithms

### Minimax

Standard depth-limited minimax with:
- **Maximizing** on the AI's turn; **minimizing** on the opponent's turn.
- **Depth cutoff**: Evaluates leaf nodes using the selected heuristic.
- **Terminal detection**: Returns `+∞` for wins, `-∞` for losses.

```
function minimax(state, depth, maximizing):
    if depth == 0 or terminal(state):
        return evaluate(state)
    if maximizing:
        best = -∞
        for move in legal_moves(state):
            child = forecast(state, move)
            best = max(best, minimax(child, depth-1, false))
        return best
    else:
        best = +∞
        for move in legal_moves(state):
            child = forecast(state, move)
            best = min(best, minimax(child, depth-1, true))
        return best
```

### Alpha-Beta Pruning

Alpha-Beta eliminates subtrees that provably cannot affect the final decision:

- **Alpha**: Best score the maximizer can guarantee (lower bound).
- **Beta**: Best score the minimizer can guarantee (upper bound).
- **Pruning condition**: `beta ≤ alpha` — the current branch is irrelevant.

```
function alphabeta(state, depth, α, β, maximizing):
    if depth == 0 or terminal(state):
        return evaluate(state)
    if maximizing:
        for move in ordered_moves(state):
            child = forecast(state, move)
            α = max(α, alphabeta(child, depth-1, α, β, false))
            if β ≤ α: break    ← prune
        return α
    else:
        for move in ordered_moves(state):
            child = forecast(state, move)
            β = min(β, alphabeta(child, depth-1, α, β, true))
            if β ≤ α: break    ← prune
        return β
```

**Complexity**: O(b^d) worst case, O(b^(d/2)) best case with perfect move ordering.

### Iterative Deepening

The agent searches at depth 1, then 2, then 3, etc., keeping the best move from the last fully completed iteration:

```
best_move = legal_moves[0]
for depth in 1, 2, 3, ..., 50:
    try:
        move, score = alphabeta(state, depth)
        best_move = move
        if |score| ≥ 900: break    ← decisive result
    except SearchTimeout:
        break
return best_move
```

Benefits:
1. **Anytime behaviour**: Always has a legal move ready, even if interrupted.
2. **Adaptive depth**: Uses all available time without overshooting.

### Timeout Handling

A `SearchTimeout` exception is raised whenever `time_left() ≤ 50ms`. This is checked at the top of every `minimax()` and `alphabeta()` call. The iterative-deepening wrapper catches the exception and returns the best move from the last completed depth.

## Move Ordering

Legal moves are sorted before evaluation to maximize alpha-beta pruning:

1. **Push moves first** — these are aggressive and potentially game-ending.
2. **Central moves next** — sorted by Manhattan distance to center (3,3), ascending.

This ordering is cheap (a single sort per node) and helps the search find good moves earlier, triggering more cutoffs.

## Heuristic Design

### OpenMoveEvalFn (Production)

```
H(s) = my_legal_moves − opponent_legal_moves
```

- **O(1) marginal cost** — legal moves are already computed for the search.
- Dominant signal: the player with more mobility tends to win.
- Used as the production default because it preserves maximum search depth.

### CustomEvalFn (Experimental)

```
H(s) = (my_moves − opp_moves) + 0.1 × (opp_center_dist − my_center_dist)
```

- Adds a tiny centrality tiebreaker (O(1) cost).
- When mobility is equal, prefers positions closer to the board center.
- **Benchmark result**: Loses to OpenMoveEvalFn 3-7 over 10 games, confirming that search depth > heuristic sophistication in this game.

## Terminal Scoring

- **Win** (opponent pushed off board or has no moves): `+∞`
- **Loss**: `-∞`
- Terminal detection is handled by `forecast_move()` returning `is_over=True`.

## Computational Tradeoffs

| Decision | Rationale |
|---|---|
| Simple heuristic over complex one | Preserves search depth, which is the dominant factor |
| Move ordering (sort) | Small per-node cost pays for itself via pruning |
| Exception-based timeout | Clean, reliable, minimal overhead vs polling |
| No transposition table | Avoids memory overhead and hash collision risks for a university project |
| 50ms safety margin | Conservative enough to prevent timeouts on varied hardware |
