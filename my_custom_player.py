import random
import time
import csv
import math
from isolation import DebugState
from sample_players import DataPlayer


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only *required* method. You can modify
    the interface for get_action by adding named parameters with default
    values, but the function MUST remain compatible with the default
    interface.

    **********************************************************************
    NOTES:
    - You should **ONLY** call methods defined on your agent class during
      search; do **NOT** add or call functions outside the player class.
      The isolation library wraps each method of this class to interrupt
      search when the time limit expires, but the wrapper only affects
      methods defined on this class.

    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired. 

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        if state.ply_count < 2:
            opp_loc = state.locs[1 - self.player_id]
            corners = [0, 10, 104, 114]
            for corner in corners:
                if opp_loc == corner:
                    corners.remove(corner)
                    break
            choice = random.choice(corners)
            self.queue.put(choice)
            
        else:
            start = int(time.time() * 1000)
            choice, depth = self.iterative_deepening(state, start)
            self.queue.put(choice)
            
    def iterative_deepening(self, state, start):
        depth_limit = 20
        best_score = float("-inf")
        best_move = None
        depth_initial = 2
        for depth in range(1, depth_limit):
            move, score, depth = self.alpha_beta_search(state, depth, start)
            depth_initial = depth
            if score >= best_score:
                best_move = move
                best_score = score
            if self.time_elapsed(start):
                return best_move, depth_initial
        return best_move, depth_initial
    
    def alpha_beta_search(self, gameState, depth, start):
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        
        def min_value(gameState, alpha, beta, depth, start):
            if gameState.terminal_test(): return gameState.utility(self.player_id)
            if depth <= 0: return self.score(gameState)
            v = float("inf")
            for a in gameState.actions():
                if self.time_elapsed(start): return v
                v = min(v, max_value(gameState.result(a), alpha, beta, depth - 1, start))
                if v <= alpha:
                    return v
            beta = min(beta, v)
            if self.time_elapsed(start): return v
            return v
        
        def max_value(gameState, alpha, beta, depth, start):
            if gameState.terminal_test(): return gameState.utility(self.player_id)
            if depth <= 0: return self.score(gameState)
            v = float("-inf")
            for a in gameState.actions():
                if self.time_elapsed(start): return v
                v = max(v, min_value(gameState.result(a), alpha, beta, depth - 1, start))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            if self.time_elapsed(start): return v
            return v
        
        for a in gameState.actions():
            v = min_value(gameState.result(a), alpha, beta, depth - 1, start)
            alpha = max(alpha, v)
            if v >= best_score:
                best_score = v
                best_move = a
            if self.time_elapsed(start): return best_move, best_score, depth
        
        return best_move, best_score, depth

# TODO: modify the function signature to accept an alpha and beta parameter
    
    def time_elapsed(self, start):
        """This function is passed the start time of the turn and returns true if the threshold is reached. Its threshold is 140 ms
        by default."""
        end = int(time.time() * 1000)
        if end - start >= 140:
            return True
        else:
            return False
        
    def score(self, state):
        """This scores a given state with a score dependant upon the Players distance to the corners of the game
        board and the other player. It prefers to stay close to the corners of the game board and stay close to
        the opponent"""
        
        def distance(node1, node2):
            """This function is given two nodes and returns a distance between them based on the game board in 
            integer format"""
            
            dy = int(abs(number - own_loc)/13)
            dx = int(abs(number - 13*dy - own_loc))
            d = math.sqrt(dx**2 + dy**2)
            return d
        
        corners = [0, 10, 104, 114]
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        min_dist = float('-inf')
        for number in corners:
            d = distance(number, own_loc)
            if d > min_dist:
                dist_from_corner = d
        dist_from_opp = -1*distance(own_loc,opp_loc)
        
        return len(own_liberties) - len(opp_liberties) + 0.2*dist_from_corner + 0.2*dist_from_opp