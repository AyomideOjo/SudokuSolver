"""
Further improvement is possible; however, such improvement is more suited to someone less prone to boredom compared to me.
Potential implementations include: 'X-wing', 'XY-wing', 'XYZ-wing', 'Sue de Coq', etc.

This code was based on the Code from:
https://github.com/hhc97/sudoku_solver/blob/master/sudoku_solver.py
"""

from __future__ import annotations

from re import sub
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

class SudokuPuzzle:
    """
    A representation of the current state of a Sudoku puzzle.
    """

    def __init__(self, n: int, symbols: List[List[str]],
                 symbol_set: Set[str], grid_map=None) -> None:
        """
        Initializes the puzzle.

        Empty spaces are denoted by the '*' symbol, and the grid symbols
        are represented as letters or numerals.

        ===== Preconditions =====
        - n is an integer that is a perfect square
        - the given symbol_set must contain n unique symbols
        - for the puzzle to print properly, symbols should be 1 character long
        - there are n lists in symbols, and each list has n strings as elements
        """
        # ===== Private Attributes =====
        # _n: The number of rows and columns in the puzzle.
        # _symbols: A list of lists representing
        #           the current state of the puzzle.
        # _symbol_set: The set of symbols that each row, column, and subsquare
        #              must have exactly one of for this puzzle to be solved.
        # _map: A dictionary mapping each unfilled position to the possible
        #       symbols that can still be filled in at that position.
        # _set_map: A dictionary that maps the unfilled symbols for each
        #           row, column, and subsquare set to the possible positions
        #           that they could occupy within that section.

        _n: int
        _symbols: List[List[str]]
        _symbol_set: Set[str]
        _map: Dict[Tuple[int, int], Set[str]]
        _set_map: Dict[str, Dict[str, Set[Tuple[int, int]]]]

        assert n == len(symbols), 'length of symbols not equal to value of n'
        self._n, self._symbols, self._symbol_set, self._set_map \
            = n, symbols, symbol_set, {}
        if grid_map is None:
            self._map = {}
            self._populate_map()
        else:
            self._map = grid_map

    def _populate_map(self) -> None:
        # updates _map with possible symbols for each unfilled position
        for r in range(self._n):
            for c in range(self._n):
                if self._symbols[r][c] == '*':
                    subset = self._row_set(r) | self._column_set(c) | \
                             self._subsquare_set(r, c)
                    allowed_symbols = self._symbol_set - subset
                    self._map[(r, c)] = allowed_symbols

    def _populate_set_map(self) -> None:
        # updates _set_map with missing symbols for each set
        # and the positions they could possibly occupy within the set
        for r in range(self._n):
            set_name = f'row{r}'
            self._set_map[set_name] = {}
            row_set = self._row_set(r)
            missing_symbols = self._symbol_set - row_set
            for sym in missing_symbols:
                self._set_map[set_name][sym] = set()
                for key, value in self._map.items():
                    if key[0] == r and sym in value:
                        self._set_map[set_name][sym].add(key)
        if self._n > 9:
            # these computations take up too much time for the regular 9x9 puzzles
            for c in range(self._n):
                set_name = f'col{c}'
                self._set_map[set_name] = {}
                col_set = self._column_set(c)
                missing_symbols = self._symbol_set - col_set
                for sym in missing_symbols:
                    self._set_map[set_name][sym] = set()
                    for key, value in self._map.items():
                        if key[1] == c and sym in value:
                            self._set_map[set_name][sym].add(key)
            n = round(self._n ** (1 / 2))
            for r in range(0, self._n, n):
                for c in range(0, self._n, n):
                    set_name = f'ss{r // n}{c // n}'
                    self._set_map[set_name] = {}
                    subsq_set = self._subsquare_set(r, c)
                    missing_symbols = self._symbol_set - subsq_set
                    for sym in missing_symbols:
                        self._set_map[set_name][sym] = set()
                        for key, value in self._map.items():
                            if key[0] // n == r // n and key[1] // n == c // n \
                                    and sym in value:
                                self._set_map[set_name][sym].add(key)

    def get_symbols(self) -> List[List[str]]:
        """
        Returns a copy of symbols, for use during testing.
        """
        return [row[:] for row in self._symbols]

    def __str__(self) -> str:
        """
        Returns an easily readable string representation of the current puzzle.
        """
        string_repr, n = [], round(self._n ** (1 / 2))
        div = '--' * n + ('+' + '-' + '--' * n) * (n - 2) + '+' + '--' * n
        for i in range(self._n):
            if i > 0 and i % n == 0:
                string_repr.append(div)
            row_lst = self._symbols[i][:]
            for index in range(n, self._n, n + 1):
                row_lst.insert(index, '|')
            string_repr.append(' '.join(row_lst))
        return '\n'.join(string_repr)

    def is_solved(self) -> bool:
        """
        Returns whether the current puzzle is solved.
        """
        return not any('*' in row for row in self._symbols) \
            and self._check_row_and_col() and self._check_subsquares()

    def _check_row_and_col(self) -> bool:
        # (helper for is_solved)
        # checks that all rows and columns are filled in properly
        return all(self._row_set(i) == self._symbol_set and
                   self._column_set(i) == self._symbol_set
                   for i in range(self._n))

    def _check_subsquares(self) -> bool:
        # (helper for is_solved)
        # checks that all subsquares are filled in properly
        n = round(self._n ** (1 / 2))
        return all(self._subsquare_set(r, c) == self._symbol_set
                   for r in range(0, self._n, n) for c in range(0, self._n, n))

    def extensions(self) -> List[SudokuPuzzle]:
        """
        Returns a list of SudokuPuzzle objects that have the position
        with the least number of possibilities filled in.

        This method checks for naked singles first, and if none are found,
        checks for hidden singles. Again, if none are found, it fills in the
        spot with the least number of naked/hidden possibilities.
        """
        if not self._map:
            return []
        extensions = []
        position, possible = None, self._symbol_set | {'*'}
        for pos, values in self._map.items():
            if len(values) < len(possible):
                position, possible = pos, values
        symbol, possible_positions = None, None
        if len(possible) > 1:
            self._populate_set_map()
            for d in self._set_map.values():
                for sym, positions in d.items():
                    if len(positions) < len(possible):
                        symbol, possible_positions, = sym, positions
        if symbol:
            for pos in possible_positions:
                new_symbols = [row[:] for row in self._symbols]
                new_symbols[pos[0]][pos[1]] = symbol
                new_map = self._map.copy()
                for key in self._get_positions(pos):
                    new_map[key] = self._map[key] - {symbol}
                del new_map[pos]
                extensions.append(SudokuPuzzle(self._n, new_symbols,
                                               self._symbol_set, new_map))
        else:
            for value in possible:
                new_symbols = [row[:] for row in self._symbols]
                new_symbols[position[0]][position[1]] = value
                new_map = self._map.copy()
                for key in self._get_positions(position):
                    new_map[key] = self._map[key] - {value}
                del new_map[position]
                extensions.append(SudokuPuzzle(self._n, new_symbols,
                                               self._symbol_set, new_map))
        return extensions

    def _get_positions(self, pos: tuple) -> List[Tuple[int, int]]:
        # returns the keys of sets in _map that may need to be altered
        n = round(self._n ** (1 / 2))
        return [key for key in self._map if key[0] == pos[0] or
                key[1] == pos[1] or (key[0] // n == pos[0] // n and
                                     key[1] // n == pos[1] // n)]

    def _row_set(self, r: int) -> Set[str]:
        # returns the set of symbols of row r
        return set(self._symbols[r])

    def _column_set(self, c: int) -> Set[str]:
        # returns the set of symbols of column c
        return set(row[c] for row in self._symbols)

    def _subsquare_set(self, r: int, c: int) -> Set[str]:
        # returns the set of symbols of the subsquare that [r][c] belongs to
        n, symbols = self._n, self._symbols
        ss = round(n ** (1 / 2))
        ul_row = (r // ss) * ss
        ul_col = (c // ss) * ss
        return set(symbols[ul_row + i][ul_col + j]
                   for i in range(ss) for j in range(ss))

class SudokuBox:
    def __init__(self, element, row, value):
        self.value = element
        self.index = (row, value)
        self.possibilities = [i for i in range(1, 10)] if element == "*" else [int(element)]

def initialize_box(puzzle: SudokuPuzzle):
    dictionary = {}
    symbols = puzzle.get_symbols()
    puzzle = []

    for row, symbol_row in enumerate(symbols):
        temp = []
        for val, symbol in enumerate(symbol_row):
            box = SudokuBox(symbol, row, val)
            dictionary[(row, val)] = box

            temp.append(box)
        puzzle.append(temp)
    return dictionary, np.array(puzzle)

def find_box(row, col):
    box_row = row // 3
    box_col = col // 3
    return box_row + 3 * box_col

def prune_possibilities(dictionary, puzzle):
    for (a, b), value in dictionary.items():
        if value.value != "*":
            continue
        row_set = puzzle._row_set(a)
        column_set = puzzle._column_set(b)
        subsquare_set = puzzle._subsquare_set(a, b)

        pruned_list = [x for x in value.possibilities if
            str(x) not in row_set and str(x) not in column_set and
            str(x) not in subsquare_set]

        value.possibilities = pruned_list
    return dictionary, puzzle

def hidden_single(dictionary, puzzle):
    for (a, b), value in dictionary.items():
        if len(dictionary[(a, b)].possibilities) == 1:
            continue
        box_number = find_box(a, b)
        box_row = (box_number // 3) * 3
        box_col = (box_number % 3) * 3

        pruned = dictionary[(a, b)].possibilities
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if (i, j) == (a, b):
                    continue
                pruned = list(filter(lambda x: x not in dictionary[(i, j)].possibilities, pruned))

        if len(pruned) == 1:
            dictionary[(a, b)].possibilities = pruned
            dictionary[(a, b)].value = dictionary[(a, b)].possibilities[0]

            for i in range(9):
                if (a, i) == (a, b) or (i, b) == (a, b):
                    continue
                dictionary[(a, i)].possibilities = list(filter(lambda x: x not in pruned, dictionary[(a, i)].possibilities))
                dictionary[(i, b)].possibilities = list(filter(lambda x: x not in pruned, dictionary[(i, b)].possibilities))

        #print(find_box(a, b), (a, b), dictionary[(a, b)].possibilities)
    return dictionary, puzzle


def copy_sublists(input_list):
    two_element_count = {}
    three_element_count = {}
    result_list = []

    for sublist in input_list:
        if len(sublist) == 2:
            # Count occurrences of 2-element sublists
            key = tuple(sublist)
            two_element_count[key] = two_element_count.get(key, 0) + 1

            # Check if a 2-element sublist appears twice
            if two_element_count[key] == 2:
                result_list.append(sublist)

        elif len(sublist) == 3:
            # Count occurrences of 3-element sublists
            key = tuple(sublist)
            three_element_count[key] = three_element_count.get(key, 0) + 1

            # Check if a 3-element sublist appears three times
            if three_element_count[key] == 3:
                result_list.append(sublist)

    return result_list

def naked_helper(lists):
    duplicates = copy_sublists(lists)

    elements_to_remove = set()
    for sublist in duplicates:
        elements_to_remove.update(sublist)

    for sublist in lists:
        if sublist not in duplicates:
            sublist[:] = [x for x in sublist if x not in elements_to_remove]
    return lists

def naked_solver(dictionary, puzzle):
    for i in range(puzzle.shape[0]):
        naked_helper([obj.possibilities for obj in puzzle[i]])
        naked_helper([obj.possibilities for obj in puzzle[:, i]])
    return dictionary, puzzle

def constraint_solve(puzzle: SudokuPuzzle) -> Optional[SudokuPuzzle]:
    dictionary, puzzleArray = initialize_box(puzzle)
    while not puzzle.is_solved():
        dictionary, puzzle = prune_possibilities(dictionary, puzzle)
        dictionary, puzzle = hidden_single(dictionary, puzzle)
        dictionary, puzzleArray = naked_solver(dictionary, puzzleArray)

        for key, value in dictionary.items():
            pass
            #print(key, ":", value.value, value.possibilities)
        quit()

def is_valid_grid(lst: list, symbol_set: set) -> bool:
    """
    Returns True if this is a valid Sudoku grid.
    """
    return not any(lst[r].count(sym) > 1 or
                   [row[c] for row in lst].count(sym) > 1
                   or _sbsq_lst(r, c, lst).count(sym) > 1
                   for r in range(len(lst)) for c in range(len(lst[r]))
                   for sym in symbol_set)

def _sbsq_lst(r: int, c: int, symbols: list) -> list:
    # (helper for is_valid_grid)
    # returns the list of symbols in the subsquare containing [r][c]
    ss = round(len(symbols) ** (1 / 2))
    return [symbols[(r // ss) * ss + i][(c // ss) * ss + j]
            for i in range(ss) for j in range(ss)]

def make_9x9_sudoku_puzzle() -> SudokuPuzzle:
    """
    Takes user input to build and return a SudokuPuzzle object.
    """
    symbol_set = {str(i) for i in range(1, 10)}
    symbols = [[n for n in sub('[^1-9]', '*',
                               input(f'Please type in row {r}:')[:9].ljust(9, '*'))]
               for r in range(1, 10)]
    if is_valid_grid(symbols, symbol_set):
        return SudokuPuzzle(9, symbols, symbol_set)
    else:
        print(f'\nGrid entered:\n{SudokuPuzzle(9, symbols, symbol_set)}')
        print('\nInvalid grid entered, please retry.\n')
        return make_9x9_sudoku_puzzle()

puzzle = SudokuPuzzle(9, [
    ["*", "*", "*", "7", "*", "8", "*", "1", "*"],
    ["*", "*", "7", "*", "9", "*", "*", "*", "6"],
    ["9", "*", "3", "1", "*", "*", "*", "*", "*"],
    ["3", "5", "*", "8", "*", "*", "6", "*", "1"],
    ["*", "*", "*", "*", "*", "*", "*", "*", "*"],
    ["1", "*", "6", "*", "*", "9", "*", "4", "8"],
    ["*", "*", "*", "*", "*", "1", "2", "*", "7"],
    ["8", "*", "*", "*", "7", "*", "4", "*", "*"],
    ["*", "6", "*", "3", "*", "2", "*", "*", "*"]],
    {"1", "2", "3", "4", "5", "6", "7", "8", "9"})

solved = constraint_solve(puzzle)

