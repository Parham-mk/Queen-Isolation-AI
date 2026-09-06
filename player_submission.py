#!/usr/bin/env python
from game import Board, game_as_text
from random import randint


class SearchTimeout(Exception):
    """Exception raised when the search runs out of time."""
    pass


class OpenMoveEvalFn:
    def score(self, game, maximizing_player_turn=True):
        """Score the current game state.

        Returns the difference between the number of legal moves
        available to the active player and the number available to
        the opponent.

        Args:
            game (Board): The board and game state.
            maximizing_player_turn (bool): True if the maximizing
                player is the active player.

        Returns:
            float: my_moves - opp_moves (flipped when minimizing).
        """
        my_moves = len(game.get_legal_moves())
        opp_moves = len(game.get_opponent_moves())

        if maximizing_player_turn:
            return float(my_moves - opp_moves)
        else:
            return float(opp_moves - my_moves)


class CustomEvalFn:
    """Enhanced evaluation: mobility + centrality bonus.

    Critical design choice: in this game, search depth matters far
    more than heuristic sophistication.  Every microsecond spent in
    the evaluation function reduces the depth the agent can reach
    within the time limit.  This heuristic adds only a tiny O(1)
    centrality term on top of the mobility difference so that it
    searches just as deep as OpenMoveEvalFn while making slightly
    better positional choices when mobility scores are tied."""

    def __init__(self):
        pass

    def score(self, game, maximizing_player_turn=True):
        """Evaluate the current game state.

        H(s) = (my_moves - opp_moves)
             + 0.1 * (opp_center_dist - my_center_dist)
        """
        n_my = len(game.get_legal_moves())
        n_opp = len(game.get_opponent_moves())

        active_q = game.get_active_players_queen()
        inactive_q = game.get_inactive_players_queen()
        my_pos = game.__last_queen_move__[active_q]
        opp_pos = game.__last_queen_move__[inactive_q]

        # Centrality: how close to (3,3).  O(1) cost.
        if my_pos[0] != -1:
            my_cd = abs(int(my_pos[0]) - 3) + abs(int(my_pos[1]) - 3)
        else:
            my_cd = 6
        if opp_pos[0] != -1:
            opp_cd = abs(int(opp_pos[0]) - 3) + abs(int(opp_pos[1]) - 3)
        else:
            opp_cd = 6

        raw = float(n_my - n_opp) + 0.1 * (opp_cd - my_cd)

        return raw if maximizing_player_turn else -raw


class CustomPlayer:
    """Player that chooses a move using iterative-deepening alpha-beta
    search with move ordering and safe time management."""

    def __init__(self, eval_fn=OpenMoveEvalFn()):
        """Initializes your player.

        Args:
            eval_fn: Evaluation function instance.
        """
        self.eval_fn = eval_fn
        self.search_depth = 4
        self.time_margin = 50  # ms safety margin

    # ------------------------------------------------------------------ #
    #  Public interface called by the game framework                      #
    # ------------------------------------------------------------------ #
    def move(self, game, legal_moves, time_left):
        """Called to determine one move by your agent.

        Uses iterative-deepening alpha-beta.  The best move from the
        last *fully completed* depth is always kept as a safe fallback.

        Args:
            game (Board): The board and game state.
            legal_moves (list): List of legal moves.
            time_left (function): Returns remaining time in ms.

        Returns:
            tuple: best_move  (row, col, push)
        """
        if not legal_moves:
            return None

        # Fast path: only one legal move
        if len(legal_moves) == 1:
            return legal_moves[0]

        best_move = legal_moves[0]

        try:
            depth = 1
            max_depth = 50  # 7x7 board can't exceed ~47 plies
            while depth <= max_depth:
                move, utility = self.alphabeta(
                    game, time_left, depth=depth)
                if move is not None:
                    best_move = move

                # If we found a decisive result, stop deepening
                if utility >= 900.0 or utility <= -900.0:
                    break

                depth += 1
        except SearchTimeout:
            pass

        return best_move

    # ------------------------------------------------------------------ #
    #  Evaluation helper                                                  #
    # ------------------------------------------------------------------ #
    def utility(self, game, maximizing_player):
        """Can be updated if desired. Not compulsory."""
        return self.eval_fn.score(game, maximizing_player)

    def check_timeout(self, time_left):
        """Raise SearchTimeout if remaining time is too low."""
        if time_left() <= self.time_margin:
            raise SearchTimeout()

    # ------------------------------------------------------------------ #
    #  Move ordering                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _order_moves(moves):
        """Order moves to maximize alpha-beta pruning.

        Priority (descending):
          1. Push moves  (aggressive / potentially winning)
          2. Central moves  (higher strategic mobility)
        """
        return sorted(moves,
                      key=lambda m: (m[2],                       # push first
                                     -(abs(m[0]-3) + abs(m[1]-3))),  # central
                      reverse=True)

    # ------------------------------------------------------------------ #
    #  Minimax (clean reference implementation)                           #
    # ------------------------------------------------------------------ #
    def minimax(self, game, time_left, depth, maximizing_player=True):
        """Implementation of the minimax algorithm.

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout.
            depth: Used to track how deep you are in the search tree.
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, val
        """
        self.check_timeout(time_left)

        legal_moves = game.get_legal_moves()

        if depth == 0 or not legal_moves:
            return None, self.utility(game, maximizing_player)

        best_move = None

        if maximizing_player:
            best_val = float('-inf')
            for move in legal_moves:
                next_game, is_over, winner = game.forecast_move(move)
                if is_over:
                    return move, float('inf')

                _, val = self.minimax(
                    next_game, time_left, depth - 1, False)

                if val > best_val or best_move is None:
                    best_val = val
                    best_move = move
            return best_move, best_val

        else:
            best_val = float('inf')
            for move in legal_moves:
                next_game, is_over, winner = game.forecast_move(move)
                if is_over:
                    return move, float('-inf')

                _, val = self.minimax(
                    next_game, time_left, depth - 1, True)

                if val < best_val or best_move is None:
                    best_val = val
                    best_move = move
            return best_move, best_val

    # ------------------------------------------------------------------ #
    #  Alpha-Beta with move ordering                                      #
    # ------------------------------------------------------------------ #
    def alphabeta(self, game, time_left, depth,
                  alpha=float("-inf"), beta=float("inf"),
                  maximizing_player=True):
        """Implementation of the alphabeta algorithm.

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout.
            depth: Used to track how deep you are in the search tree.
            alpha (float): Alpha value for pruning.
            beta (float): Beta value for pruning.
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, val
        """
        self.check_timeout(time_left)

        legal_moves = game.get_legal_moves()

        if depth == 0 or not legal_moves:
            return None, self.utility(game, maximizing_player)

        # Move ordering for better pruning
        legal_moves = self._order_moves(legal_moves)

        best_move = None

        if maximizing_player:
            best_val = float('-inf')
            for move in legal_moves:
                next_game, is_over, winner = game.forecast_move(move)
                if is_over:
                    return move, float('inf')

                _, val = self.alphabeta(
                    next_game, time_left, depth - 1,
                    alpha, beta, False)

                if val > best_val or best_move is None:
                    best_val = val
                    best_move = move

                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break

            return best_move, best_val

        else:
            best_val = float('inf')
            for move in legal_moves:
                next_game, is_over, winner = game.forecast_move(move)
                if is_over:
                    return move, float('-inf')

                _, val = self.alphabeta(
                    next_game, time_left, depth - 1,
                    alpha, beta, True)

                if val < best_val or best_move is None:
                    best_val = val
                    best_move = move

                beta = min(beta, best_val)
                if beta <= alpha:
                    break

            return best_move, best_val
