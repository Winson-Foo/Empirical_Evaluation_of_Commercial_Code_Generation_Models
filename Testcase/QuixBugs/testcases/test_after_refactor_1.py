# # %%
# import pytest
# from .load_testdata import load_json_testcases
# from .test_node import Node

# import sys

# sys.path.append('C:/Users/fatty/Desktop/Empirical_Evaluation_of_Commercial_Code_Generation_Models/After_Refactor_1/QuixBugs')  

# ###
# # Testing bitcount
# import bitcount
# testdata = load_json_testcases(bitcount.__name__)

# # parameterize the test case from the json file
# @pytest.mark.parametrize("input_data,expected", testdata)
# def test_bitcount(input_data, expected):
#     assert bitcount.bitcount(*input_data) == expected

# ####
# # Testing breadth first search
# import breadth_first_search
# def test1():
#     """Case 1: Strongly connected graph
#     Output: Path found!
#     """

#     station1 = Node("Westminster")
#     station2 = Node("Waterloo", None, [station1])
#     station3 = Node("Trafalgar Square", None, [station1, station2])
#     station4 = Node("Canary Wharf", None, [station2, station3])
#     station5 = Node("London Bridge", None, [station4, station3])
#     station6 = Node("Tottenham Court Road", None, [station5, station4])

#     path_found = breadth_first_search.breadth_first_search(station6, station1)

#     assert path_found

# def test2():
#     """Case 2: Branching graph
#     Output: Path found!
#     """

#     nodef = Node("F")
#     nodee = Node("E")
#     noded = Node("D")
#     nodec = Node("C", None, [nodef])
#     nodeb = Node("B", None, [nodee])
#     nodea = Node("A", None, [nodeb, nodec, noded])

#     path_found = breadth_first_search.breadth_first_search(nodea, nodee)

#     assert path_found


# def test3():
#     """Case 3: Two unconnected nodes in graph
#     Output: Path not found
#     """

#     nodef = Node("F")
#     nodee = Node("E")

#     path_found = breadth_first_search.breadth_first_search(nodef, nodee)

#     assert not path_found


# def test4():
#     """Case 4: One node graph
#     Output: Path found!
#     """

#     nodef = Node("F")

#     path_found = breadth_first_search.breadth_first_search(nodef, nodef)

#     assert path_found


# def test5():
#     """Case 5: Graph with cycles
#     Output: Path found!
#     """

#     nodef = Node("F")
#     nodee = Node("E")
#     noded = Node("D")
#     nodec = Node("C", None, [nodef])
#     nodeb = Node("B", None, [nodee])
#     nodea = Node("A", None, [nodeb, nodec, noded])

#     nodee.successors = [nodea]

#     path_found = breadth_first_search.breadth_first_search(nodea, nodef)

#     assert path_found

# ####
# # Testing bucketsort
# import bucketsort
# # bucket sort
# testdata = load_json_testcases(bucketsort.__name__)

# @pytest.mark.parametrize("input_data,expected", testdata)
# def test_bucketsort(input_data, expected):
#     assert bucketsort.count_sort(*input_data) == expected

# ####
# # Testing depth first search
# import depth_first_search

# def test1():
#     """Case 1: Strongly connected graph
#     Output: Path found!
#     """

#     station1 = Node("Westminster")
#     station2 = Node("Waterloo", None, [station1])
#     station3 = Node("Trafalgar Square", None, [station1, station2])
#     station4 = Node("Canary Wharf", None, [station2, station3])
#     station5 = Node("London Bridge", None, [station4, station3])
#     station6 = Node("Tottenham Court Road", None, [station5, station4])

#     path_found = depth_first_search.is_reachable(station6, station1)

#     assert path_found


# def test2():
#     """Case 2: Branching graph
#     Output: Path found!
#     """

#     nodef = Node("F")
#     nodee = Node("E")
#     noded = Node("D")
#     nodec = Node("C", None, [nodef])
#     nodeb = Node("B", None, [nodee])
#     nodea = Node("A", None, [nodeb, nodec, noded])

#     path_found = depth_first_search.is_reachable(nodea, nodee)

#     assert path_found


# def test3():
#     """Case 3: Two unconnected nodes in graph
#     Output: Path not found
#     """

#     nodef = Node("F")
#     nodee = Node("E")

#     path_found = depth_first_search.is_reachable(nodef, nodee)

#     assert not path_found


# def test4():
#     """Case 4: One node graph
#     Output: Path found!
#     """

#     nodef = Node("F")

#     path_found = depth_first_search.is_reachable(nodef, nodef)

#     assert path_found


# def test5():
#     """Case 5: Graph with cycles
#     Output: Path found!
#     """

#     nodef = Node("F")
#     nodee = Node("E")
#     noded = Node("D")
#     nodec = Node("C", None, [nodef])
#     nodeb = Node("B", None, [nodee])
#     nodea = Node("A", None, [nodeb, nodec, noded])

#     nodee.successors = [nodea]

#     path_found = depth_first_search.is_reachable(nodea, nodef)

#     assert path_found
# ###
# import detect_cycle

# node1 = Node(1)
# node2 = Node(2, node1)
# node3 = Node(3, node2)
# node4 = Node(4, node3)
# node5 = Node(5, node4)


# def test1():
#     """Case 1: 5-node list input with no cycle
#     Expected Output: Cycle not detected!
#     """

#     detected = detect_cycle.detect_cycle(node5)

#     assert not detected


# def test2():
#     """Case 2: 5-node list input with cycle
#     Expected Output: Cycle detected!
#     """

#     node1.successor = node5

#     detected = detect_cycle.detect_cycle(node5)

#     assert detected


# def test3():
#     """Case 3: 2-node list with cycle
#     Expected Output: Cycle detected!
#     """

#     node1.successor = node2

#     detected = detect_cycle.detect_cycle(node2)

#     assert detected


# def test4():
#     """Case 4: 2-node list with no cycle
#     Expected Output: Cycle not detected!
#     """

#     node6 = Node(6)
#     node7 = Node(7, node6)

#     detected = detect_cycle.detect_cycle(node7)

#     assert not detected


# def test5():
#     """Case 5: 1-node list
#     Expected Output: Cycle not detected
#     """

#     node = Node(0)
#     detected = detect_cycle.detect_cycle(node)

#     assert not detected


# def test6():
#     """Case 6: 5 nodes in total. the last 2 nodes form a cycle. input the first node
#     Expected Output: Cycle detected!
#     """

#     node1.successor = node2

#     detected = detect_cycle.detect_cycle(node5)

#     assert detected
# ###
# import find_first_in_sorted

# testdata = load_json_testcases(find_first_in_sorted.__name__)


# @pytest.mark.parametrize("input_data,expected", testdata)
# def test_find_first_in_sorted(input_data, expected):
#     assert find_first_in_sorted.find_first(*input_data) == expected