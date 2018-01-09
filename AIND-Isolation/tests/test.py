from random import randint
from game_agent import *
from isolation import *
from sample_players import *
import timeit



if __name__ == "__main__":
    player1 = MinimaxPlayer()
    print(player1.search_depth)
    print(player1.score)
    print(player1.time_left)
    print(player1.TIMER_THRESHOLD)
    ##
    import timeit
    time_millis = lambda: 1000 * timeit.default_timer()
    move_start = time_millis()
    time_left = lambda: player1.TIMER_THRESHOLD - (time_millis() - move_start)
    #

    from isolation import Board
    # create an isolation board (by default 7x7)
    player2 = MinimaxPlayer(search_depth=10, score_fn=custom_score, timeout=10.)
    player1 = AlphaBetaPlayer(search_depth=10, score_fn=custom_score, timeout=10.)
    game = Board(player1, player2)

    # place player 1 on the board at row 2, column 3, then place player 2 on
    # the board at row 0, column 5; display the resulting board state.  Note
    # that the .apply_move() method changes the calling object in-place.
    game.apply_move((2, 3))
    game.apply_move((0, 5))

    #
    assert (player1 == game.active_player)
    # get a list of the legal moves available to the active player
    print(game.get_legal_moves())

    # get a successor of the current state by making a copy of the board and
    # applying a move. Notice that this does NOT change the calling object
    # (unlike .apply_move()).

    # play the remainder of the game automatically -- outcome can be "illegal
    # move", "timeout", or "forfeit"
    winner, history, outcome = game.play(time_limit=15)
    print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
    print(game.to_string())
    print("Move history:\n{!s}".format(history))