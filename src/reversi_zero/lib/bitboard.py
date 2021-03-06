# 有巨大问题！！！！
import numpy as np

BLACK_CHR = "O"
WHITE_CHR = "X"
EXTRA_CHR = "*"


def get_num(length):
    re = 1
    for i in range(length - 1):
        re <<= 1
        re += 1
    return re


def board_to_string(black, white, with_edge=True, extra=None):
    """
     0  1  2  3  4  5  6  7
     8  9 10 11 12 13 14 15
    ..
    56 57 58 59 60 61 62 63

    0: Top Left, LSB
    63: Bottom Right

    :param black: bitboard
    :param white: bitboard
    :param with_edge:
    :param extra: bitboard
    :return:
    """
    array = [" "] * 225
    extra = extra or 0
    for i in range(225):
        if black & 1:
            array[i] = BLACK_CHR
        elif white & 1:
            array[i] = WHITE_CHR
        elif extra & 1:
            array[i] = EXTRA_CHR
        black >>= 1
        white >>= 1
        extra >>= 1

    ret = ""
    if with_edge:
        ret = "#" * 10 + "\n"
    for y in range(15):
        if with_edge:
            ret += "#"
        ret += "".join(array[y * 15:y * 15 + 15])
        if with_edge:
            ret += "#"
        ret += "\n"
    if with_edge:
        ret += "#" * 10 + "\n"
    return ret


def find_correct_moves(own, enemy):
    """return legal moves"""

    return get_num(225) ^ own ^ enemy


# def calc_flip(pos, own, enemy):
#     """return flip stones of enemy by bitboard when I place stone at pos.
#
#     :param pos: 0~63
#     :param own: bitboard (0=top left, 63=bottom right)
#     :param enemy: bitboard
#     :return: flip stones of enemy when I place stone at pos.
#     """
#     assert 0 <= pos <= 224, f"pos={pos}"
#     f1 = _calc_flip_half(pos, own, enemy)
#     f2 = _calc_flip_half(224 - pos, rotate180(own), rotate180(enemy))
#     return f1 | rotate180(f2)
#
#
# def _calc_flip_half(pos, own, enemy):
#     el = [enemy, enemy & 0xfff9fff3ffe7ffcfff9fff3ffe7ffcfff9fff3ffe7ffcfff9fff3ffe,
#           enemy & 0xfff9fff3ffe7ffcfff9fff3ffe7ffcfff9fff3ffe7ffcfff9fff3ffe,
#           enemy & 0xfff9fff3ffe7ffcfff9fff3ffe7ffcfff9fff3ffe7ffcfff9fff3ffe]
#     masks = [0x40008001000200040008001000200040008001000200040008000, 0x7ffe,
#              0x10004001000400100040010004001000400100040010004000,
#              0x100010001000100010001000100010001000100010001000100010000]
#     masks = [b225(m << pos) for m in masks]
#     flipped = 0
#     for e, mask in zip(el, masks):
#         outflank = mask & ((e | ~mask) + 1) & own
#         flipped |= (outflank - (outflank != 0)) & mask
#     return flipped


# def search_offset_left(own, enemy, mask, offset):
#     e = enemy & mask
#     blank = ~(own | enemy)
#     t = e & (own >> offset)
#     t |= e & (t >> offset)
#     t |= e & (t >> offset)
#     t |= e & (t >> offset)
#     t |= e & (t >> offset)
#     t |= e & (t >> offset)  # Up to six stones can be turned at once
#     return blank & (t >> offset)  # Only the blank squares can be started
#
#
# def search_offset_right(own, enemy, mask, offset):
#     e = enemy & mask
#     blank = ~(own | enemy)
#     t = e & (own << offset)
#     t |= e & (t << offset)
#     t |= e & (t << offset)
#     t |= e & (t << offset)
#     t |= e & (t << offset)
#     t |= e & (t << offset)  # Up to six stones can be turned at once
#     return blank & (t << offset)  # Only the blank squares can be started


def flip_vertical(x):
    # test_x= bit_to_array(x,225).reshape((15,15))
    x = np.flipud(bit_to_array(x,225).reshape(15,15))
    x = array_to_bit(x)
    # test_h = bit_to_array(x, 225).reshape((15, 15))
    return x


def b225(x):
    return x & get_num(225)


def bit_count(x):
    return bin(x).count('1')


def bit_to_array(x, size):
    """bit_to_array(0b0010, 4) -> array([0, 1, 0, 0])"""
    return np.array(list(reversed((("0" * size) + bin(x)[2:])[-size:])), dtype=np.uint8)



def rotate90(x):
    h = bit_to_array(x, 225).reshape((15, 15))
    h = np.rot90(h)
    h = array_to_bit(h)
    return h


def rotate180(x):
    return rotate90(rotate90(x))


def dirichlet_noise_of_mask(mask, alpha):
    num_1 = bit_count(mask)
    noise = list(np.random.dirichlet([alpha] * num_1))
    ret_list = []
    for i in range(225):
        if (1 << i) & mask:
            ret_list.append(noise.pop(0))
        else:
            ret_list.append(0)
    return np.array(ret_list)


def array_to_bit(x):
    n = len(x[0])
    ss = '0b'
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            ss += str(x[i][j])
    ss = int(ss, 2)
    return ss
