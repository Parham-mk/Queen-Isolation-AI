import sys
import os
import time

# Add parent directory to path so we can import from the main project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game import Board
from test_players import RandomPlayer
from player_submission import CustomPlayer, OpenMoveEvalFn, CustomEvalFn

def play_match(p1, p2, time_limit=1000):
    board = Board(p1, p2, 7, 7)
    
    start_time = time.time()
    
    # We patch the time_left function inside play_isolation to provide standard time
    winner, move_history, termination = board.play_isolation(time_limit=time_limit, print_moves=False)
    
    elapsed = time.time() - start_time
    
    # Determine winner
    p1_name = getattr(p1, 'get_name', lambda: None)()
    if winner == p1_name or (hasattr(p1, '__class__') and winner == p1.__class__.__name__ + " - Q1"):
        return 1, elapsed
    else:
        return 2, elapsed

def run_tournament():
    matches = [
        (RandomPlayer("Random"), CustomPlayer(eval_fn=OpenMoveEvalFn())),
        (CustomPlayer(eval_fn=OpenMoveEvalFn()), RandomPlayer("Random")),
        (CustomPlayer(eval_fn=OpenMoveEvalFn()), CustomPlayer(eval_fn=CustomEvalFn())),
        (CustomPlayer(eval_fn=CustomEvalFn()), CustomPlayer(eval_fn=OpenMoveEvalFn()))
    ]
    
    num_games = 5
    time_limit = 1000 # ms
    
    print("Starting Tournament...")
    print(f"{'Matchup':<45} | {'P1 Wins':<8} | {'P2 Wins':<8} | {'Avg Time (s)':<12}")
    print("-" * 80)
    
    for p1, p2 in matches:
        p1_name = p1.name if hasattr(p1, 'name') else p1.__class__.__name__ + " (P1)"
        p2_name = p2.name if hasattr(p2, 'name') else p2.__class__.__name__ + " (P2)"
        
        # Distinguish between eval functions if both are CustomPlayer
        if isinstance(p1, CustomPlayer):
            p1_name = f"CustomPlayer({p1.eval_fn.__class__.__name__})"
        if isinstance(p2, CustomPlayer):
            p2_name = f"CustomPlayer({p2.eval_fn.__class__.__name__})"
            
        p1_wins = 0
        p2_wins = 0
        total_time = 0
        
        for i in range(num_games):
            winner, elapsed = play_match(p1, p2, time_limit)
            if winner == 1:
                p1_wins += 1
            else:
                p2_wins += 1
            total_time += elapsed
            
        avg_time = total_time / num_games
        
        matchup_str = f"{p1_name} vs {p2_name}"
        print(f"{matchup_str:<45} | {p1_wins:<8} | {p2_wins:<8} | {avg_time:<12.2f}")

if __name__ == '__main__':
    run_tournament()
