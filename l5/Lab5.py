import sys
import numpy as np
from typing import Union, Iterable


def print_path(actions: dict, messages: dict, cur_cell: tuple):
    if cur_cell == (0, 0):
        return 0
    else:
        print_path(actions, messages, actions[cur_cell])
        print(messages[cur_cell])


def print_parens(s_mat: np.ndarray, i: int, j: int, name: str = 'A'):
    if i == j:
        print(name + chr(0x2080 + i + 1), end='')
    else:
        print('(', end='')
        print_parens(s_mat, i, s_mat[i, j])
        print_parens(s_mat, s_mat[i, j] + 1, j)
        print(')', end='')


def levenshtein_distance(s: str, t: str, verbose: bool = False) -> int:
    if not(isinstance(s, str) or isinstance(t, str)):
        raise TypeError("The input is of the wrong type.")

    len_s = len(s)
    len_t = len(t)
    if s == '':
        if t == '':
            if verbose:
                print("The strings are empty.")
            return 0
        else:
            if verbose:
                print("Insert all symbols of the second string into the first string.")
            return len_t
    elif t == '':
        if verbose:
            print("Delete all symbols from the first string.")
        return len_s

    d_mat = np.zeros((len_s + 1, len_t + 1), dtype=np.int64)
    d_mat[0, :] = np.arange(len_t + 1, dtype=np.int64)
    d_mat[:, 0] = np.arange(len_s + 1, dtype=np.int64)

    actions = {}
    messages = {}
    if not verbose:
        del actions
        del messages
    else:
        for i in range(1, len_s + 1):
            actions[(i, 0)] = (i - 1, 0)
            messages[(i, 0)] = f"Delete symbol {s[i - 1]}."

        for j in range(1, len_t + 1):
            actions[(0, j)] = (0, j - 1)
            messages[(0, j)] = f"Insert symbol {t[j - 1]}."

    for i in range(1, len_s + 1):
        for j in range(1, len_t + 1):
            if s[i - 1] == t[j - 1]:
                d_mat[i, j] = d_mat[i - 1, j - 1]

                if verbose:
                    actions[(i, j)] = (i - 1, j - 1)
                    messages[(i, j)] = f"Leave symbol {s[i - 1]}."
            else:
                insert = d_mat[i, j - 1] + 1
                delete = d_mat[i - 1, j] + 1
                replace = d_mat[i - 1, j - 1] + 2
                d_mat[i, j] = min(insert, delete, replace)

                if verbose:
                    action_idx = np.argmin([insert, delete, replace])
                    actions[(i, j)] = [(i, j - 1), (i - 1, j), (i - 1, j - 1)][action_idx]
                    messages[(i, j)] = [f"Insert symbol {t[j - 1]}.", f"Delete symbol {s[i - 1]}.",
                                        f"Delete symbol {s[i - 1]}.\nInsert symbol {t[j - 1]}."][action_idx]

    if verbose:
        print_path(actions, messages, (len_s, len_t))

    return d_mat[-1, -1]


def damerau_levenshtein_distance(s: str, t: str, verbose: bool = False) -> int:
    if not (isinstance(s, str) or isinstance(t, str)):
        raise TypeError("The input is of the wrong type.")

    len_s = len(s)
    len_t = len(t)
    if s == '':
        if t == '':
            if verbose:
                print("The strings are empty.")
            return 0
        else:
            if verbose:
                print("Insert all symbols of the second string into the first string.")
            return len_t
    elif t == '':
        if verbose:
            print("Delete all symbols from the first string.")
        return len_s

    d_mat = np.zeros((len_s + 1, len_t + 1), dtype=np.int64)
    d_mat[0, :] = np.arange(len_t + 1, dtype=np.int64)
    d_mat[:, 0] = np.arange(len_s + 1, dtype=np.int64)

    actions = {}
    messages = {}
    if not verbose:
        del actions
        del messages
    else:
        for i in range(1, len_s + 1):
            actions[(i, 0)] = (i - 1, 0)
            messages[(i, 0)] = f"Delete symbol {s[i - 1]}."

        for j in range(1, len_t + 1):
            actions[(0, j)] = (0, j - 1)
            messages[(0, j)] = f"Insert symbol {t[j - 1]}."

    for i in range(1, len_s + 1):
        for j in range(1, len_t + 1):
            if s[i - 1] == t[j - 1]:
                d_mat[i, j] = d_mat[i - 1, j - 1]

                if verbose:
                    actions[(i, j)] = (i - 1, j - 1)
                    messages[(i, j)] = f"Leave symbol {s[i - 1]}."
            else:
                insert = d_mat[i, j - 1] + 1
                delete = d_mat[i - 1, j] + 1
                replace = d_mat[i - 1, j - 1] + 2
                d_mat[i, j] = min(insert, delete, replace)

                if verbose:
                    action_idx = np.argmin([insert, delete, replace])
                    actions[(i, j)] = [(i, j - 1), (i - 1, j), (i - 1, j - 1)][action_idx]
                    messages[(i, j)] = [f"Insert symbol {t[j - 1]}.", f"Delete symbol {s[i - 1]}.",
                                        f"Delete symbol {s[i - 1]}.\nInsert symbol {t[j - 1]}."][action_idx]

            if (i > 1) and (j > 1) and (s[i - 1] == t[j - 2]) and (s[i - 2] == t[j - 1]):
                transpose = d_mat[i - 2, j - 2] + 1

                if transpose < d_mat[i, j]:
                    d_mat[i, j] = transpose

                    if verbose:
                        actions[(i, j)] = (i - 2, j - 2)
                        messages[(i, j)] = f"Swap symbols {s[i - 2]} and {s[i - 1]}."

    if verbose:
        print_path(actions, messages, (len_s, len_t))

    return d_mat[-1, -1]


def matrix_chain(matrix_sizes: list, verbose: bool = True) -> Union[bool, int]:
    if not isinstance(matrix_sizes, Iterable):
        raise TypeError("The input must be of the Iterable type.")

    matrix_sizes = list(matrix_sizes)
    n = len(matrix_sizes)

    if n == 0:
        raise ValueError("The input is empty.")

    for i in range(n):
        if not isinstance(matrix_sizes[i], (list, tuple)):
            raise TypeError("The input must be an iterable over lists or tuples containing matrix dimensions.")
        if not len(matrix_sizes[i]) == 2:
            raise TypeError("The input must contain pairs of matrix dimensions from the matrix chain.")

    if n == 1:
        if verbose:
            print("Only one matrix passed for the matrix chain multiplication.")
            print("Total 0 multiplication operations.")
        return 0
    elif n == 2:
        if matrix_sizes[0][1] == matrix_sizes[1][0]:
            mul_num = matrix_sizes[0][0] * matrix_sizes[0][1] * matrix_sizes[1][1]
            if verbose:
                print('(A' + chr(0x2081) + 'A' + chr(0x2082) + ')')
                print(f"Total {mul_num} multiplication operations.")
            return mul_num
        else:
            if verbose:
                print("Impossible to multiply two matrices.")
            return False

    p = [matrix_sizes[0][0]]
    for i in range(n - 1):
        if matrix_sizes[i][1] == matrix_sizes[i + 1][0]:
            p.append(matrix_sizes[i][1])
        else:
            del p
            if verbose:
                print("Impossible to multiply.")
                if i == 0:
                    num_str1 = f'{i + 1}st'
                    num_str2 = f'{i + 2}nd'
                elif i == 1:
                    num_str1 = f'{i + 1}nd'
                    num_str2 = f'{i + 2}rd'
                elif i == 2:
                    num_str1 = f'{i + 1}rd'
                    num_str2 = f'{i + 2}th'
                else:
                    num_str1 = f'{i + 1}th'
                    num_str2 = f'{i + 2}th'

                print("The " + num_str1 + f" matrix has {matrix_sizes[i][1]} columns, but the "
                      + num_str2 + f" matrix has {matrix_sizes[i + 1][0]} rows.")

            return False
    p.append(matrix_sizes[-1][1])

    m_mat = np.zeros((n, n), dtype=np.int64)
    s_mat = np.zeros((n, n), dtype=np.int64)
    for k in range(1, n):
        for i in range(n - k):
            j = i + k
            m_mat[i, j] = sys.maxsize
            for h in range(i, j):
                mul = m_mat[i, h] + m_mat[h + 1, j] + p[i] * p[h + 1] * p[j + 1]
                if m_mat[i, j] > mul:
                    m_mat[i, j] = mul
                    s_mat[i, j] = h

    mul_num = m_mat[0, -1]
    if verbose:
        print_parens(s_mat, 0, n - 1)
        print(f"\nTotal {mul_num} multiplication operations.")
    return mul_num


if __name__ == "__main__":
    # i = int(input()) % 2 + 1
    i = 1
    if i == 1:
        print("Test 1 (Levenstein):")
        input_1 = ['abcd', 'badc']
        dist_l = levenshtein_distance(*input_1, verbose=True)
        print(f"\nLevenstein distance between {input_1[0]} and {input_1[1]} equals {dist_l}.")
        print('\n')

        print("Test 1 (Damerau-Levenstein):")
        dist_dl = damerau_levenshtein_distance(*input_1, verbose=True)
        print(f"\nDamerau-Levenstein distance between {input_1[0]} and {input_1[1]} equals {dist_dl}.")
        print('-' * 50)

        print("Test 2 (Levenstein):")
        input_2 = ['abcd', 'bcda']
        dist_l = levenshtein_distance(*input_2, verbose=True)
        print(f"\nLevenstein distance between {input_2[0]} and {input_2[1]} equals {dist_l}.")
        print('\n')

        print("Test 2 (Damerau-Levenstein):")
        dist_dl = damerau_levenshtein_distance(*input_2, verbose=True)
        print(f"\nDamerau-Levenstein distance between {input_2[0]} and {input_2[1]} equals {dist_dl}.")
    else:
        print("Test 1:")
        input_1 = [(20, 20)]
        matrix_chain(input_1)

        print("\nTest 2:")
        input_2 = [(20, 20), (20, 1)]
        matrix_chain(input_2)

        print("\nTest 3:")
        input_3 = [(30, 35), (35, 15), (15, 5), (5, 10), (10, 20), (20, 25)]
        matrix_chain(input_3)

        print("\nTest 4:")
        input_4 = [(5, 5), (4, 4)]
        matrix_chain(input_4)

        print("\nTest 5:")
        input_5 = [(5, 4), (4, 6), (5, 2), (2, 7)]
        matrix_chain(input_5)
