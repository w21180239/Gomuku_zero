import enum

from logging import getLogger

from reversi_zero.lib.bitboard import board_to_string, calc_flip, bit_count, find_correct_moves, bit_to_array

logger = getLogger(__name__)
# noinspection PyArgumentList
Player = enum.Enum("Player", "black white")
# noinspection PyArgumentList
Winner = enum.Enum("Winner", "black white draw")


def another_player(player: Player):
    return Player.white if player == Player.black else Player.black


def is_game_over(own, action):
    my_board = bit_to_array(own, 225).reshape(15, 15)
    x = action // 15
    y = action % 15
    limit = 5
    count = 0
    for i in range(15):
        if my_board[x][i] == 1:
            count += 1
    if count == limit:
        return True
    count = 0
    for i in range(15):
        if my_board[i][y] == 1:
            count += 1
    if count == limit:
        return True

    count = 1
    for i in range(1, 15):
        xx = x + i
        yy = y + i
        if 0 <= xx < 15 and 0 <= yy < 15 and my_board[xx][yy] == 1:
            count += 1
    for i in range(1, 15):
        xx = x - i
        yy = y - i
        if 0 <= xx < 15 and 0 <= yy < 15 and my_board[xx][yy] == 1:
            count += 1
    if count == limit:
        return True

    count = 1
    for i in range(1, 15):
        xx = x + i
        yy = y - i
        if 0 <= xx < 15 and 0 <= yy < 15 and my_board[xx][yy] == 1:
            count += 1
    for i in range(1, 15):
        xx = x - i
        yy = y + i
        if 0 <= xx < 15 and 0 <= yy < 15 and my_board[xx][yy] == 1:
            count += 1
    if count == limit:
        return True

    return False


class ReversiEnv:
    def __init__(self):
        self.board = None
        self.next_player = None  # type: Player
        self.turn = 0
        self.done = False
        self.winner = None  # type: Winner

    def reset(self):
        self.board = Board()
        self.next_player = Player.black
        self.turn = 0
        self.done = False
        self.winner = None
        return self

    def update(self, black, white, next_player):
        self.board = Board(black, white)
        self.next_player = next_player
        self.turn = sum(self.board.number_of_black_and_white)
        self.done = False
        self.winner = None
        return self

    def step(self, action):
        """

        :param int|None action: move pos=0 ~ 224 (0=top left, 14 top right, 224 bottom right), None is resign
        :return:
        """
        assert action is None or 0 <= action <= 224, f"Illegal action={action}"

        if action is None:
            self._resigned()
            return self.board, {}

        own, enemy = self.get_own_and_enemy()

        # flipped = calc_flip(action, own, enemy)
        # if bit_count(flipped) == 0:
        #     self.illegal_move_to_lose(action)
        #     return self.board, {}
        # own ^= flipped
        # enemy ^= flipped
        own |= 1 << action

        self.set_own_and_enemy(own, enemy)
        self.turn += 1



        # if bit_count(find_correct_moves(enemy, own)) > 0:  # there are legal moves for enemy.
        #     self.change_to_next_player()
        # elif bit_count(find_correct_moves(own, enemy)) > 0:  # there are legal moves for me but enemy.
        #     pass
        # else:  # there is no legal moves for me and enemy.
        #     self._game_over()

        if is_game_over(own, action) or sum(self.board.number_of_black_and_white) == 225:
            self._game_over()
        else:
            self.change_to_next_player()


        return self.board, {}

    def _game_over(self):
        self.done = True
        if self.winner is None:
            if sum(self.board.number_of_black_and_white) == 225:
                self.winner = Winner.draw
                return
            if self.next_player == Player.black:
                self.winner = Winner.black
            else:
                self.winner = Winner.white

    def change_to_next_player(self):
        self.next_player = another_player(self.next_player)

    def illegal_move_to_lose(self, action):
        logger.warning(f"Illegal action={action}, No Flipped!")
        self._win_another_player()
        self._game_over()

    def _resigned(self):
        self._win_another_player()
        self._game_over()

    def _win_another_player(self):
        win_player = another_player(self.next_player)  # type: Player
        if win_player == Player.black:
            self.winner = Winner.black
        else:
            self.winner = Winner.white

    def get_own_and_enemy(self):
        if self.next_player == Player.black:
            own, enemy = self.board.black, self.board.white
        else:
            own, enemy = self.board.white, self.board.black
        return own, enemy

    def set_own_and_enemy(self, own, enemy):
        if self.next_player == Player.black:
            self.board.black, self.board.white = own, enemy
        else:
            self.board.white, self.board.black = own, enemy

    def render(self):
        b, w = self.board.number_of_black_and_white
        print(f"next={self.next_player.name} turn={self.turn} B={b} W={w}")
        print(board_to_string(self.board.black, self.board.white, with_edge=True))

    @property
    def observation(self):
        """

        :rtype: Board
        """
        return self.board


class Board:
    def __init__(self, black=None, white=None, init_type=0):
        self.black = black or (0)
        self.white = white or (0)

        if init_type:
            self.black, self.white = self.white, self.black

    @property
    def number_of_black_and_white(self):
        return bit_count(self.black), bit_count(self.white)
