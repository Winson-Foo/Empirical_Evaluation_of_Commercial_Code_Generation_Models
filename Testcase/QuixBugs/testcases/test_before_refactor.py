# %%
### To run this file type pytest Testcase in python
import pytest
from .load_testdata import load_json_testcases
from .test_node import Node

import sys

sys.path.append('C:/Users/fatty/Desktop/Empirical_Evaluation_of_Commercial_Code_Generation_Models/Before_Refactor/QuixBugs')  

###
# Testing bitcount
import bitcount
testdata = load_json_testcases(bitcount.__name__)

# parameterize the test case from the json file
@pytest.mark.parametrize("input_data,expected", testdata)
def test_bitcount(input_data, expected):
    assert bitcount.bitcount(*input_data) == expected

####
# Testing breadth first search
import breadth_first_search
def test1():
    """Case 1: Strongly connected graph
    Output: Path found!
    """

    station1 = Node("Westminster")
    station2 = Node("Waterloo", None, [station1])
    station3 = Node("Trafalgar Square", None, [station1, station2])
    station4 = Node("Canary Wharf", None, [station2, station3])
    station5 = Node("London Bridge", None, [station4, station3])
    station6 = Node("Tottenham Court Road", None, [station5, station4])

    path_found = breadth_first_search.breadth_first_search(station6, station1)

    assert path_found

def test2():
    """Case 2: Branching graph
    Output: Path found!
    """

    nodef = Node("F")
    nodee = Node("E")
    noded = Node("D")
    nodec = Node("C", None, [nodef])
    nodeb = Node("B", None, [nodee])
    nodea = Node("A", None, [nodeb, nodec, noded])

    path_found = breadth_first_search.breadth_first_search(nodea, nodee)

    assert path_found


def test3():
    """Case 3: Two unconnected nodes in graph
    Output: Path not found
    """

    nodef = Node("F")
    nodee = Node("E")

    path_found = breadth_first_search.breadth_first_search(nodef, nodee)

    assert not path_found


def test4():
    """Case 4: One node graph
    Output: Path found!
    """

    nodef = Node("F")

    path_found = breadth_first_search.breadth_first_search(nodef, nodef)

    assert path_found


def test5():
    """Case 5: Graph with cycles
    Output: Path found!
    """

    nodef = Node("F")
    nodee = Node("E")
    noded = Node("D")
    nodec = Node("C", None, [nodef])
    nodeb = Node("B", None, [nodee])
    nodea = Node("A", None, [nodeb, nodec, noded])

    nodee.successors = [nodea]

    path_found = breadth_first_search.breadth_first_search(nodea, nodef)

    assert path_found

####
# Testing bucketsort
import bucketsort
# bucket sort
testdata = load_json_testcases(bucketsort.__name__)

@pytest.mark.parametrize("input_data,expected", testdata)
def test_bucketsort(input_data, expected):
    assert bucketsort.bucketsort(*input_data) == expected

####
# Testing depth first search
import depth_first_search

def test1():
    """Case 1: Strongly connected graph
    Output: Path found!
    """

    station1 = Node("Westminster")
    station2 = Node("Waterloo", None, [station1])
    station3 = Node("Trafalgar Square", None, [station1, station2])
    station4 = Node("Canary Wharf", None, [station2, station3])
    station5 = Node("London Bridge", None, [station4, station3])
    station6 = Node("Tottenham Court Road", None, [station5, station4])

    path_found = depth_first_search.is_reachable(station6, station1)

    assert path_found


def test2():
    """Case 2: Branching graph
    Output: Path found!
    """

    nodef = Node("F")
    nodee = Node("E")
    noded = Node("D")
    nodec = Node("C", None, [nodef])
    nodeb = Node("B", None, [nodee])
    nodea = Node("A", None, [nodeb, nodec, noded])

    path_found = depth_first_search.is_reachable(nodea, nodee)

    assert path_found


def test3():
    """Case 3: Two unconnected nodes in graph
    Output: Path not found
    """

    nodef = Node("F")
    nodee = Node("E")

    path_found = depth_first_search.is_reachable(nodef, nodee)

    assert not path_found


def test4():
    """Case 4: One node graph
    Output: Path found!
    """

    nodef = Node("F")

    path_found = depth_first_search.is_reachable(nodef, nodef)

    assert path_found


def test5():
    """Case 5: Graph with cycles
    Output: Path found!
    """

    nodef = Node("F")
    nodee = Node("E")
    noded = Node("D")
    nodec = Node("C", None, [nodef])
    nodeb = Node("B", None, [nodee])
    nodea = Node("A", None, [nodeb, nodec, noded])

    nodee.successors = [nodea]

    path_found = depth_first_search.is_reachable(nodea, nodef)

    assert path_found
###
import detect_cycle

node1_dc = Node(1)
node2_dc = Node(2, node1_dc)
node3_dc = Node(3, node2_dc)
node4_dc = Node(4, node3_dc)
node5_dc = Node(5, node4_dc)


def test1():
    """Case 1: 5-node list input with no cycle
    Expected Output: Cycle not detected!
    """

    detected = detect_cycle.detect_cycle(node5_dc)

    assert not detected


def test2():
    """Case 2: 5-node list input with cycle
    Expected Output: Cycle detected!
    """

    node1.successor = node5_dc

    detected = detect_cycle.detect_cycle(node5_dc)

    assert detected


def test3():
    """Case 3: 2-node list with cycle
    Expected Output: Cycle detected!
    """

    node1.successor = node2_dc

    detected = detect_cycle.detect_cycle(node2_dc)

    assert detected


def test4():
    """Case 4: 2-node list with no cycle
    Expected Output: Cycle not detected!
    """

    node6 = Node(6)
    node7 = Node(7, node6)

    detected = detect_cycle.detect_cycle(node7)

    assert not detected


def test5():
    """Case 5: 1-node list
    Expected Output: Cycle not detected
    """

    node = Node(0)
    detected = detect_cycle.detect_cycle(node)

    assert not detected


def test6():
    """Case 6: 5 nodes in total. the last 2 nodes form a cycle. input the first node
    Expected Output: Cycle detected!
    """

    node1_dc.successor = node2_dc

    detected = detect_cycle.detect_cycle(node5_dc)

    assert detected
###
import find_first_in_sorted

testdata = load_json_testcases(find_first_in_sorted.__name__)


@pytest.mark.parametrize("input_data,expected", testdata)
def test_find_first_in_sorted(input_data, expected):
    assert find_first_in_sorted.find_first_in_sorted(*input_data) == expected

###
import find_in_sorted

testdata = load_json_testcases(find_in_sorted.__name__)


@pytest.mark.parametrize("input_data,expected", testdata)
def test_find_in_sorted(input_data, expected):
    assert find_in_sorted.find_in_sorted(*input_data) == expected

###
import flatten

testdata = load_json_testcases(flatten.__name__)


@pytest.mark.parametrize("input_data,expected", testdata)
def test_flatten(input_data, expected):
    assert list(flatten.flatten(*input_data)) == expected

###
import gcd

testdata = load_json_testcases(gcd.__name__)


@pytest.mark.parametrize("input_data,expected", testdata)
def test_gcd(input_data, expected):
    assert gcd.gcd(*input_data) == expected

###
import get_factors

testdata = load_json_testcases(get_factors.__name__)

@pytest.mark.parametrize("input_data,expected", testdata)
def test_get_factors(input_data, expected):
    assert get_factors.get_factors(*input_data) == expected

###
import hanoi
testdata = load_json_testcases(hanoi.__name__)
testdata = [[inp, [tuple(x) for x in out]] for inp, out in testdata]


@pytest.mark.parametrize("input_data,expected", testdata)
def test_hanoi(input_data, expected):
    assert hanoi.hanoi(*input_data) == expected

###
import is_valid_parenthesization

testdata = load_json_testcases(is_valid_parenthesization.__name__)


@pytest.mark.parametrize("input_data,expected", testdata)
def test_is_valid_parenthesization(input_data, expected):
    assert is_valid_parenthesization.is_valid_parenthesization(*input_data) == expected

###
import kheapsort

testdata = load_json_testcases(kheapsort.__name__)


@pytest.mark.parametrize("input_data,expected", testdata)
def test_kheapsort(input_data, expected):
    assert list(kheapsort.kheapsort(*input_data)) == expected

###
import knapsack

testdata = load_json_testcases(knapsack.__name__)

@pytest.mark.parametrize("input_data,expected", testdata)
def test_knapsack(input_data, expected):
    assert knapsack.knapsack(*input_data) == expected

###
import kth

testdata = load_json_testcases(kth.__name__)


@pytest.mark.parametrize("input_data,expected", testdata)
def test_kth(input_data, expected):
    assert kth.kth(*input_data) == expected

###
import lcs_length

testdata = load_json_testcases(lcs_length.__name__)


@pytest.mark.parametrize("input_data,expected", testdata)
def test_lcs_length(input_data, expected):
    assert lcs_length.lcs_length(*input_data) == expected

###
import levenshtein


testdata = load_json_testcases(levenshtein.__name__)


@pytest.mark.parametrize("input_data,expected", testdata)
def test_levenshtein(input_data, expected):
    assert levenshtein.levenshtein(*input_data) == expected

###
import lis

testdata = load_json_testcases(lis.__name__)

@pytest.mark.parametrize("input_data,expected", testdata)
def test_lis(input_data, expected):
    assert lis.lis(*input_data) == expected

###
import longest_common_subsequence


testdata = load_json_testcases(longest_common_subsequence.__name__)


@pytest.mark.parametrize("input_data,expected", testdata)
def test_longest_common_subsequence(input_data, expected):
    assert longest_common_subsequence.longest_common_subsequence(*input_data) == expected

###
import max_sublist_sum


testdata = load_json_testcases(max_sublist_sum.__name__)


@pytest.mark.parametrize("input_data,expected", testdata)
def test_max_sublist_sum(input_data, expected):
    assert max_sublist_sum.max_sublist_sum(*input_data) == expected

### 
import mergesort

testdata = load_json_testcases(mergesort.__name__)

@pytest.mark.parametrize("input_data,expected", testdata)
def test_mergesort(input_data, expected):
    assert mergesort.mergesort(*input_data) == expected

###
import minimum_spanning_tree


def test1():
    """Case 1: Simple tree input.
    Output: (1, 2) (3, 4) (1, 4)
    """

    result = minimum_spanning_tree.minimum_spanning_tree(
        {
            (1, 2): 10,
            (2, 3): 15,
            (3, 4): 10,
            (1, 4): 10,
        }
    )

    assert result == {(1, 2), (3, 4), (1, 4)}


def test2():
    """Case 2: Strongly connected tree input.
    Output: (2, 5) (1, 3) (2, 3) (4, 6) (3, 6)
    """

    result = minimum_spanning_tree.minimum_spanning_tree(
        {
            (1, 2): 6,
            (1, 3): 1,
            (1, 4): 5,
            (2, 3): 5,
            (2, 5): 3,
            (3, 4): 5,
            (3, 5): 6,
            (3, 6): 4,
            (4, 6): 2,
            (5, 6): 6,
        }
    )

    assert result == {(2, 5), (1, 3), (2, 3), (4, 6), (3, 6)}


def test3():
    """Case 3: Minimum spanning tree input.
    Output: (1, 2) (1, 3) (2, 4)
    """

    result = minimum_spanning_tree.minimum_spanning_tree(
        {
            (1, 2): 6,
            (1, 3): 1,
            (2, 4): 2,
        }
    )

    assert result == {(1, 2), (1, 3), (2, 4)}

###
import next_palindrome

testdata = load_json_testcases(next_palindrome.__name__)


@pytest.mark.parametrize("input_data,expected", testdata)
def test_next_palindrome(input_data, expected):
    assert next_palindrome.next_palindrome(*input_data) == expected

###
import next_permutation


testdata = load_json_testcases(next_permutation.__name__)


@pytest.mark.parametrize("input_data,expected", testdata)
def test_next_permutation(input_data, expected):
    assert next_permutation.next_permutation(*input_data) == expected

###
import pascal

testdata = load_json_testcases(pascal.__name__)

@pytest.mark.parametrize("input_data,expected", testdata)
def test_pascal(input_data, expected):
    assert pascal.pascal(*input_data) == expected

### 
import possible_change

testdata = load_json_testcases(possible_change.__name__)

@pytest.mark.parametrize("input_data,expected", testdata)
def test_possible_change(input_data, expected):
    assert possible_change.possible_change(*input_data) == expected

###
import powerset

testdata = load_json_testcases(powerset.__name__)

@pytest.mark.parametrize("input_data,expected", testdata)
def test_powerset(input_data, expected):
    assert powerset.powerset(*input_data) == expected

###
import quicksort

testdata = load_json_testcases(quicksort.__name__)

@pytest.mark.parametrize("input_data,expected", testdata)
def test_quicksort(input_data, expected):
    assert quicksort.quicksort(*input_data) == expected

###
import reverse_linked_list


def test1():
    """Case 1: 5-node list input
    Expected Output: 1 2 3 4 5
    """

    node1 = Node(1)
    node2 = Node(2, node1)
    node3 = Node(3, node2)
    node4 = Node(4, node3)
    node5 = Node(5, node4)

    result = reverse_linked_list.reverse_linked_list(node5)
    assert result == node1

    output = []
    while result:
        output.append(result.value)
        result = result.successor
    assert output == [1, 2, 3, 4, 5]


def test2():
    """Case 2: 1-node list input
    Expected Output: 0
    """

    node = Node(0)

    result = reverse_linked_list.reverse_linked_list(node)
    assert result == node

    output = []
    while result:
        output.append(result.value)
        result = result.successor
    assert output == [0]


def test3():
    """Case 3: None input
    Expected Output: None
    """

    result = reverse_linked_list.reverse_linked_list(None)
    assert not result

    output = []
    while result:
        output.append(result.value)
        result = result.successor
    assert not output

###
import rpn_eval

testdata = load_json_testcases(rpn_eval.__name__)

@pytest.mark.parametrize("input_data,expected", testdata)
def test_rpn_eval(input_data, expected):
    assert rpn_eval.rpn_eval(*input_data) == expected

###
import shortest_path_length

node1_spl = Node("1")
node5_spl = Node("5")
node4_spl = Node("4", None, [node5_spl])
node3_spl = Node("3", None, [node4_spl])
node2_spl = Node("2", None, [node1_spl, node3_spl, node4_spl])
node0_spl = Node("0", None, [node2_spl, node5_spl])

length_by_edge = {
    (node0_spl, node2_spl): 3,
    (node0_spl, node5_spl): 10,
    (node2_spl, node1_spl): 1,
    (node2_spl, node3_spl): 2,
    (node2_spl, node4_spl): 4,
    (node3_spl, node4_spl): 1,
    (node4_spl, node5_spl): 1,
}


def test1():
    """Case 1: One path
    Output: 4
    """

    result = shortest_path_length.shortest_path_length(length_by_edge, node0_spl, node1_spl)
    assert result == 4


def test2():
    """Case 2: Multiple path
    Output: 7
    """

    result = shortest_path_length.shortest_path_length(length_by_edge, node0_spl, node5_spl)
    assert result == 7


def test3():
    """Case 3: Start point is same as end point
    Output: 0
    """

    result = shortest_path_length.shortest_path_length(length_by_edge, node2_spl, node2_spl)
    assert result == 0


def test4():
    """Case 4: Unreachable path
    Output: INT_MAX
    """

    result = shortest_path_length.shortest_path_length(length_by_edge, node1_spl, node5_spl)
    assert result == float("inf")

###
import shortest_path_lengths


def test1():
    """Case 1: Basic graph input."""

    graph = {
        (0, 2): 3,
        (0, 5): 5,
        (2, 1): -2,
        (2, 3): 7,
        (2, 4): 4,
        (3, 4): -5,
        (4, 5): -1,
    }
    result = shortest_path_lengths.shortest_path_lengths(6, graph)

    expected = {
        (0, 0): 0,
        (1, 1): 0,
        (2, 2): 0,
        (3, 3): 0,
        (4, 4): 0,
        (5, 5): 0,
        (0, 2): 3,
        (0, 5): 4,
        (2, 1): -2,
        (2, 3): 7,
        (2, 4): 2,
        (3, 4): -5,
        (4, 5): -1,
        (0, 1): 1,
        (0, 3): 10,
        (0, 4): 5,
        (1, 0): float("inf"),
        (1, 2): float("inf"),
        (1, 3): float("inf"),
        (1, 4): float("inf"),
        (1, 5): float("inf"),
        (2, 0): float("inf"),
        (2, 5): 1,
        (3, 0): float("inf"),
        (3, 1): float("inf"),
        (3, 2): float("inf"),
        (3, 5): -6,
        (4, 0): float("inf"),
        (4, 1): float("inf"),
        (4, 2): float("inf"),
        (4, 3): float("inf"),
        (5, 0): float("inf"),
        (5, 1): float("inf"),
        (5, 2): float("inf"),
        (5, 3): float("inf"),
        (5, 4): float("inf"),
    }

    assert result == expected


def test2():
    """Case 2: Linear graph input."""

    graph = {
        (0, 1): 3,
        (1, 2): 5,
        (2, 3): -2,
        (3, 4): 7,
    }
    result = shortest_path_lengths.shortest_path_lengths(5, graph)

    expected = {
        (0, 0): 0,
        (1, 1): 0,
        (2, 2): 0,
        (3, 3): 0,
        (4, 4): 0,
        (0, 1): 3,
        (1, 2): 5,
        (2, 3): -2,
        (3, 4): 7,
        (0, 2): 8,
        (0, 3): 6,
        (0, 4): 13,
        (1, 0): float("inf"),
        (1, 3): 3,
        (1, 4): 10,
        (2, 0): float("inf"),
        (2, 1): float("inf"),
        (2, 4): 5,
        (3, 0): float("inf"),
        (3, 1): float("inf"),
        (3, 2): float("inf"),
        (4, 0): float("inf"),
        (4, 1): float("inf"),
        (4, 2): float("inf"),
        (4, 3): float("inf"),
    }

    assert result == expected


def test3():
    """Case 3: Disconnected graphs input."""

    graph = {
        (0, 1): 3,
        (2, 3): 5,
    }
    result = shortest_path_lengths.shortest_path_lengths(4, graph)

    expected = {
        (0, 0): 0,
        (1, 1): 0,
        (2, 2): 0,
        (3, 3): 0,
        (0, 1): 3,
        (2, 3): 5,
        (0, 2): float("inf"),
        (0, 3): float("inf"),
        (1, 0): float("inf"),
        (1, 2): float("inf"),
        (1, 3): float("inf"),
        (2, 0): float("inf"),
        (2, 1): float("inf"),
        (3, 0): float("inf"),
        (3, 1): float("inf"),
        (3, 2): float("inf"),
    }

    assert result == expected


def test4():
    """Case 4: Strongly connected graph input."""

    graph = {
        (0, 1): 3,
        (1, 2): 5,
        (2, 0): -1,
    }
    result = shortest_path_lengths.shortest_path_lengths(3, graph)

    expected = {
        (0, 0): 0,
        (1, 1): 0,
        (2, 2): 0,
        (0, 1): 3,
        (1, 2): 5,
        (2, 0): -1,
        (0, 2): 8,
        (1, 0): 4,
        (2, 1): 2,
    }

    assert result == expected

###
import shortest_paths


def test1():
    """Case 1: Graph with multiple paths
    Output: {'A': 0, 'C': 3, 'B': 1, 'E': 5, 'D': 10, 'F': 4}
    """

    graph = {
        ("A", "B"): 3,
        ("A", "C"): 3,
        ("A", "F"): 5,
        ("C", "B"): -2,
        ("C", "D"): 7,
        ("C", "E"): 4,
        ("D", "E"): -5,
        ("E", "F"): -1,
    }
    result = shortest_paths.shortest_paths("A", graph)

    expected = {"A": 0, "C": 3, "B": 1, "E": 5, "D": 10, "F": 4}
    assert result == expected


def test2():
    """Case 2: Graph with one path
    Output: {'A': 0, 'C': 3, 'B': 1, 'E': 5, 'D': 6, 'F': 9}
    """

    graph2 = {
        ("A", "B"): 1,
        ("B", "C"): 2,
        ("C", "D"): 3,
        ("D", "E"): -1,
        ("E", "F"): 4,
    }
    result = shortest_paths.shortest_paths("A", graph2)

    expected = {"A": 0, "C": 3, "B": 1, "E": 5, "D": 6, "F": 9}
    assert result == expected


def test3():
    """Case 3: Graph with cycle
    Output: {'A': 0, 'C': 3, 'B': 1, 'E': 5, 'D': 6, 'F': 9}
    """

    graph3 = {
        ("A", "B"): 1,
        ("B", "C"): 2,
        ("C", "D"): 3,
        ("D", "E"): -1,
        ("E", "D"): 1,
        ("E", "F"): 4,
    }
    result = shortest_paths.shortest_paths("A", graph3)

    expected = {"A": 0, "C": 3, "B": 1, "E": 5, "D": 6, "F": 9}
    assert result == expected

###
import shunting_yard

testdata = load_json_testcases(shunting_yard.__name__)

@pytest.mark.parametrize("input_data,expected", testdata)
def test_shunting_yard(input_data, expected):
    assert shunting_yard.shunting_yard(*input_data) == expected

###
import sieve

testdata = load_json_testcases(sieve.__name__)

@pytest.mark.parametrize("input_data,expected", testdata)
def test_sieve(input_data, expected):
    assert sieve.sieve(*input_data) == expected

###
import sqrt

testdata = load_json_testcases(sqrt.__name__)

@pytest.mark.parametrize("input_data,expected", testdata)
def test_sqrt(input_data, expected):
    assert sqrt.sqrt(*input_data) == pytest.approx(expected, abs=input_data[-1])

###
import subsequences

testdata = load_json_testcases(subsequences.__name__)

@pytest.mark.parametrize("input_data,expected", testdata)
def test_subsequences(input_data, expected):
    assert subsequences.subsequences(*input_data) == expected

###
import to_base

testdata = load_json_testcases(to_base.__name__)

@pytest.mark.parametrize("input_data,expected", testdata)
def test_to_base(input_data, expected):
    assert to_base.to_base(*input_data) == expected

###
import topological_ordering

def test1():
    """Case 1: Wikipedia graph
    Output: 5 7 3 11 8 10 2 9
    """

    five = Node(5)
    seven = Node(7)
    three = Node(3)
    eleven = Node(11)
    eight = Node(8)
    two = Node(2)
    nine = Node(9)
    ten = Node(10)

    five.outgoing_nodes = [eleven]
    seven.outgoing_nodes = [eleven, eight]
    three.outgoing_nodes = [eight, ten]
    eleven.incoming_nodes = [five, seven]
    eleven.outgoing_nodes = [two, nine, ten]
    eight.incoming_nodes = [seven, three]
    eight.outgoing_nodes = [nine]
    two.incoming_nodes = [eleven]
    nine.incoming_nodes = [eleven, eight]
    ten.incoming_nodes = [eleven, three]

    result = [
        x.value
        for x in topological_ordering.topological_ordering(
            [five, seven, three, eleven, eight, two, nine, ten]
        )
    ]

    assert result == [5, 7, 3, 11, 8, 10, 2, 9]


def test2():
    """Case 2: GeekforGeeks example
    Output: 4 5 0 2 3 1
    """

    five = Node(5)
    zero = Node(0)
    four = Node(4)
    one = Node(1)
    two = Node(2)
    three = Node(3)

    five.outgoing_nodes = [two, zero]
    four.outgoing_nodes = [zero, one]
    two.incoming_nodes = [five]
    two.outgoing_nodes = [three]
    zero.incoming_nodes = [five, four]
    one.incoming_nodes = [four, three]
    three.incoming_nodes = [two]
    three.outgoing_nodes = [one]

    result = [
        x.value for x in topological_ordering.topological_ordering([zero, one, two, three, four, five])
    ]

    assert result == [4, 5, 0, 2, 3, 1]


def test3():
    """Case 3: Cooking with InteractivePython"""

    milk = Node("3/4 cup milk")
    egg = Node("1 egg")
    oil = Node("1 Tbl oil")
    mix = Node("1 cup mix")
    syrup = Node("heat syrup")
    griddle = Node("heat griddle")
    pour = Node("pour 1/4 cup")
    turn = Node("turn when bubbly")
    eat = Node("eat")

    milk.outgoing_nodes = [mix]
    egg.outgoing_nodes = [mix]
    oil.outgoing_nodes = [mix]
    mix.incoming_nodes = [milk, egg, oil]
    mix.outgoing_nodes = [syrup, pour]
    griddle.outgoing_nodes = [pour]
    pour.incoming_nodes = [mix, griddle]
    pour.outgoing_nodes = [turn]
    turn.incoming_nodes = [pour]
    turn.outgoing_nodes = [eat]
    syrup.incoming_nodes = [mix]
    syrup.outgoing_nodes = [eat]
    eat.incoming_nodes = [syrup, turn]

    result = [
        x.value
        for x in topological_ordering.topological_ordering(
            [milk, egg, oil, mix, syrup, griddle, pour, turn, eat]
        )
    ]

    expected = [
        "3/4 cup milk",
        "1 egg",
        "1 Tbl oil",
        "heat griddle",
        "1 cup mix",
        "pour 1/4 cup",
        "heat syrup",
        "turn when bubbly",
        "eat",
    ]
    assert result == expected

###
import wrap

testdata = load_json_testcases(wrap.__name__)

@pytest.mark.parametrize("input_data,expected", testdata)
def test_wrap(input_data, expected):
    assert wrap.wrap(*input_data) == expected