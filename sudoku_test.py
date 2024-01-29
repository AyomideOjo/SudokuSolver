"""
A set of basic unittests to ensure that every revision of the solver works.
"""

import unittest

from sudoku_solver import SudokuPuzzle, depth_first_solve


class TestSudoku(unittest.TestCase):
    """
    A class that contains a few sanity checks for the solver, followed by a few
    test puzzles to verify that it can come up with a correct solution.
    """

    def test_solved_completed_puzzle(self):
        puzzle = SudokuPuzzle(9, [['1', '2', '3', '7', '8', '9', '4', '5', '6'],
                                  ['4', '5', '6', '1', '2', '3', '7', '8', '9'],
                                  ['7', '8', '9', '4', '5', '6', '1', '2', '3'],
                                  ['3', '1', '2', '9', '7', '8', '6', '4', '5'],
                                  ['6', '4', '5', '3', '1', '2', '9', '7', '8'],
                                  ['9', '7', '8', '6', '4', '5', '3', '1', '2'],
                                  ['2', '3', '1', '8', '9', '7', '5', '6', '4'],
                                  ['5', '6', '4', '2', '3', '1', '8', '9', '7'],
                                  ['8', '9', '7', '5', '6', '4', '2', '3', '1']],
                              {"1", "2", "3", "4", "5", "6", "7", "8", "9"})
        self.assertTrue(puzzle.is_solved(), 'is_solved returning False when should be True')

    def test_solved_duplicate_row(self):
        puzzle = SudokuPuzzle(9, [['1', '2', '3', '7', '8', '9', '4', '5', '1'],
                                  ['4', '5', '6', '1', '2', '3', '7', '8', '9'],
                                  ['7', '8', '9', '4', '5', '6', '1', '2', '3'],
                                  ['3', '1', '2', '9', '7', '8', '6', '4', '5'],
                                  ['6', '4', '5', '3', '1', '2', '9', '7', '8'],
                                  ['9', '7', '8', '6', '4', '5', '3', '1', '2'],
                                  ['2', '3', '1', '8', '9', '7', '5', '6', '4'],
                                  ['5', '6', '4', '2', '3', '1', '8', '9', '7'],
                                  ['8', '9', '7', '5', '6', '4', '2', '3', '1']],
                              {"1", "2", "3", "4", "5", "6", "7", "8", "9"})
        self.assertFalse(puzzle.is_solved(), 'is_solved returning True when should be False')

    def test_solved_duplicate_column(self):
        puzzle = SudokuPuzzle(9, [['1', '2', '3', '7', '8', '9', '4', '5', '6'],
                                  ['4', '5', '6', '1', '2', '3', '7', '8', '9'],
                                  ['7', '8', '9', '4', '5', '6', '1', '2', '3'],
                                  ['3', '1', '2', '9', '7', '8', '6', '4', '5'],
                                  ['6', '4', '5', '3', '1', '2', '9', '7', '8'],
                                  ['9', '7', '8', '6', '4', '5', '3', '1', '2'],
                                  ['2', '3', '1', '8', '9', '7', '5', '6', '4'],
                                  ['5', '6', '4', '2', '3', '1', '8', '9', '7'],
                                  ['1', '9', '7', '5', '6', '4', '2', '3', '1']],
                              {"1", "2", "3", "4", "5", "6", "7", "8", "9"})
        self.assertFalse(puzzle.is_solved(), 'is_solved returning True when should be False')

    def test_solved_duplicate_subsquare(self):
        puzzle = SudokuPuzzle(9, [['1', '2', '3', '7', '8', '9', '4', '5', '6'],
                                  ['4', '5', '6', '1', '2', '3', '7', '8', '9'],
                                  ['3', '1', '2', '9', '7', '8', '6', '4', '5'],
                                  ['7', '8', '9', '4', '5', '6', '1', '2', '3'],
                                  ['6', '4', '5', '3', '1', '2', '9', '7', '8'],
                                  ['9', '7', '8', '6', '4', '5', '3', '1', '2'],
                                  ['2', '3', '1', '8', '9', '7', '5', '6', '4'],
                                  ['5', '6', '4', '2', '3', '1', '8', '9', '7'],
                                  ['8', '9', '7', '5', '6', '4', '2', '3', '1']],
                              {"1", "2", "3", "4", "5", "6", "7", "8", "9"})
        self.assertFalse(puzzle.is_solved(), 'is_solved returning True when should be False')

    def test_puzzle_empty(self):
        puzzle = SudokuPuzzle(9, [['*', '*', '*', '*', '*', '*', '*', '*', '*'],
                                  ['*', '*', '*', '*', '*', '*', '*', '*', '*'],
                                  ['*', '*', '*', '*', '*', '*', '*', '*', '*'],
                                  ['*', '*', '*', '*', '*', '*', '*', '*', '*'],
                                  ['*', '*', '*', '*', '*', '*', '*', '*', '*'],
                                  ['*', '*', '*', '*', '*', '*', '*', '*', '*'],
                                  ['*', '*', '*', '*', '*', '*', '*', '*', '*'],
                                  ['*', '*', '*', '*', '*', '*', '*', '*', '*'],
                                  ['*', '*', '*', '*', '*', '*', '*', '*', '*']],
                              {"1", "2", "3", "4", "5", "6", "7", "8", "9"})
        solved = depth_first_solve(puzzle)
        if not solved:
            self.fail("No solution found when there should be a solution.")
        self.assertTrue(solved.is_solved(), 'Solve empty puzzle failed.')

    def test_puzzle1(self):
        puzzle = SudokuPuzzle(9, [["*", "*", "*", "7", "*", "8", "*", "1", "*"],
                                  ["*", "*", "7", "*", "9", "*", "*", "*", "6"],
                                  ["9", "*", "3", "1", "*", "*", "*", "*", "*"],
                                  ["3", "5", "*", "8", "*", "*", "6", "*", "1"],
                                  ["*", "*", "*", "*", "*", "*", "*", "*", "*"],
                                  ["1", "*", "6", "*", "*", "9", "*", "4", "8"],
                                  ["*", "*", "*", "*", "*", "1", "2", "*", "7"],
                                  ["8", "*", "*", "*", "7", "*", "4", "*", "*"],
                                  ["*", "6", "*", "3", "*", "2", "*", "*", "*"]],
                              {"1", "2", "3", "4", "5", "6", "7", "8", "9"})
        solved = depth_first_solve(puzzle)
        if not solved:
            self.fail("No solution found when there should be a solution.")
        solution = [['6', '4', '5', '7', '3', '8', '9', '1', '2'], ['2', '1', '7', '5', '9', '4', '8', '3', '6'],
                    ['9', '8', '3', '1', '2', '6', '5', '7', '4'], ['3', '5', '2', '8', '4', '7', '6', '9', '1'],
                    ['4', '9', '8', '6', '1', '3', '7', '2', '5'], ['1', '7', '6', '2', '5', '9', '3', '4', '8'],
                    ['5', '3', '9', '4', '6', '1', '2', '8', '7'], ['8', '2', '1', '9', '7', '5', '4', '6', '3'],
                    ['7', '6', '4', '3', '8', '2', '1', '5', '9']]
        self.assertEqual(solved.get_symbols(), solution, 'Solution is wrong.')

    def test_puzzle2(self):
        puzzle = SudokuPuzzle(9, [["*", "*", "*", "9", "*", "2", "*", "*", "*"],
                                  ["*", "9", "1", "*", "*", "*", "6", "3", "*"],
                                  ["*", "3", "*", "*", "7", "*", "*", "8", "*"],
                                  ["3", "*", "*", "*", "*", "*", "*", "*", "8"],
                                  ["*", "*", "9", "*", "*", "*", "2", "*", "*"],
                                  ["5", "*", "*", "*", "*", "*", "*", "*", "7"],
                                  ["*", "7", "*", "*", "8", "*", "*", "4", "*"],
                                  ["*", "4", "5", "*", "*", "*", "8", "1", "*"],
                                  ["*", "*", "*", "3", "*", "6", "*", "*", "*"]],
                              {"1", "2", "3", "4", "5", "6", "7", "8", "9"})
        solved = depth_first_solve(puzzle)
        if not solved:
            self.fail("No solution found when there should be a solution.")
        solution = [['8', '5', '6', '9', '3', '2', '4', '7', '1'], ['7', '9', '1', '4', '5', '8', '6', '3', '2'],
                    ['4', '3', '2', '6', '7', '1', '5', '8', '9'], ['3', '6', '7', '5', '2', '4', '1', '9', '8'],
                    ['1', '8', '9', '7', '6', '3', '2', '5', '4'], ['5', '2', '4', '8', '1', '9', '3', '6', '7'],
                    ['2', '7', '3', '1', '8', '5', '9', '4', '6'], ['6', '4', '5', '2', '9', '7', '8', '1', '3'],
                    ['9', '1', '8', '3', '4', '6', '7', '2', '5']]
        self.assertEqual(solved.get_symbols(), solution, 'Solution is wrong.')

    def test_puzzle3(self):
        puzzle = SudokuPuzzle(9, [["5", "6", "*", "*", "*", "7", "*", "*", "9"],
                                  ["*", "7", "*", "*", "4", "8", "*", "3", "1"],
                                  ["*", "*", "*", "*", "*", "*", "*", "*", "*"],
                                  ["4", "3", "*", "*", "*", "*", "*", "*", "*"],
                                  ["*", "8", "*", "*", "*", "*", "*", "9", "*"],
                                  ["*", "*", "*", "*", "*", "*", "*", "2", "6"],
                                  ["*", "*", "*", "*", "*", "*", "*", "*", "*"],
                                  ["1", "9", "*", "3", "6", "*", "*", "7", "*"],
                                  ["7", "*", "*", "1", "*", "*", "*", "4", "2"]],
                              {"1", "2", "3", "4", "5", "6", "7", "8", "9"})
        solved = depth_first_solve(puzzle)
        if not solved:
            self.fail("No solution found when there should be a solution.")
        solution = [['5', '6', '1', '2', '3', '7', '4', '8', '9'], ['2', '7', '9', '6', '4', '8', '5', '3', '1'],
                    ['3', '4', '8', '5', '9', '1', '2', '6', '7'], ['4', '3', '2', '9', '1', '6', '7', '5', '8'],
                    ['6', '8', '5', '7', '2', '3', '1', '9', '4'], ['9', '1', '7', '8', '5', '4', '3', '2', '6'],
                    ['8', '2', '6', '4', '7', '5', '9', '1', '3'], ['1', '9', '4', '3', '6', '2', '8', '7', '5'],
                    ['7', '5', '3', '1', '8', '9', '6', '4', '2']]
        self.assertEqual(solved.get_symbols(), solution, 'Solution is wrong.')

    def test_puzzle4(self):
        puzzle = SudokuPuzzle(9, [['*', '*', '*', '*', '*', '*', '*', '*', '*'],
                                  ['*', '*', '*', '*', '*', '3', '*', '8', '5'],
                                  ['*', '*', '1', '*', '2', '*', '*', '*', '*'],
                                  ['*', '*', '*', '5', '*', '7', '*', '*', '*'],
                                  ['*', '*', '4', '*', '*', '*', '1', '*', '*'],
                                  ['*', '9', '*', '*', '*', '*', '*', '*', '*'],
                                  ['5', '*', '*', '*', '*', '*', '*', '7', '3'],
                                  ['*', '*', '2', '*', '1', '*', '*', '*', '*'],
                                  ['*', '*', '*', '*', '4', '*', '*', '*', '9']],
                              {"1", "2", "3", "4", "5", "6", "7", "8", "9"})
        solved = depth_first_solve(puzzle)
        if not solved:
            self.fail("No solution found when there should be a solution.")
        solution = [['9', '8', '7', '6', '5', '4', '3', '2', '1'], ['2', '4', '6', '1', '7', '3', '9', '8', '5'],
                    ['3', '5', '1', '9', '2', '8', '7', '4', '6'], ['1', '2', '8', '5', '3', '7', '6', '9', '4'],
                    ['6', '3', '4', '8', '9', '2', '1', '5', '7'], ['7', '9', '5', '4', '6', '1', '8', '3', '2'],
                    ['5', '1', '9', '2', '8', '6', '4', '7', '3'], ['4', '7', '2', '3', '1', '9', '5', '6', '8'],
                    ['8', '6', '3', '7', '4', '5', '2', '1', '9']]
        self.assertEqual(solved.get_symbols(), solution, 'Solution is wrong.')

    def test_puzzle5(self):
        puzzle = SudokuPuzzle(9, [['8', '*', '*', '*', '*', '*', '*', '*', '*'],
                                  ['*', '*', '3', '6', '*', '*', '*', '*', '*'],
                                  ['*', '7', '*', '*', '9', '*', '2', '*', '*'],
                                  ['*', '5', '*', '*', '*', '7', '*', '*', '*'],
                                  ['*', '*', '*', '*', '4', '5', '7', '*', '*'],
                                  ['*', '*', '*', '1', '*', '*', '*', '3', '*'],
                                  ['*', '*', '1', '*', '*', '*', '*', '6', '8'],
                                  ['*', '*', '8', '5', '*', '*', '*', '1', '*'],
                                  ['*', '9', '*', '*', '*', '*', '4', '*', '*']],
                              {"1", "2", "3", "4", "5", "6", "7", "8", "9"})
        solved = depth_first_solve(puzzle)
        if not solved:
            self.fail("No solution found when there should be a solution.")
        solution = [['8', '1', '2', '7', '5', '3', '6', '4', '9'], ['9', '4', '3', '6', '8', '2', '1', '7', '5'],
                    ['6', '7', '5', '4', '9', '1', '2', '8', '3'], ['1', '5', '4', '2', '3', '7', '8', '9', '6'],
                    ['3', '6', '9', '8', '4', '5', '7', '2', '1'], ['2', '8', '7', '1', '6', '9', '5', '3', '4'],
                    ['5', '2', '1', '9', '7', '4', '3', '6', '8'], ['4', '3', '8', '5', '2', '6', '9', '1', '7'],
                    ['7', '9', '6', '3', '1', '8', '4', '5', '2']]
        self.assertEqual(solved.get_symbols(), solution, 'Solution is wrong.')


if __name__ == '__main__':
    unittest.main()