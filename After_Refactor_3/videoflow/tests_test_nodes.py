import pytest

from videoflow.core.graph import GraphEngine
from videoflow.core.node import TaskModuleNode, ProcessorNode
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.consumers import CommandlineConsumer

# FUNCTION TO CREATE A TASK MODULE NODE WITH PROVIDED NODES
def create_task_module_node(*nodes):
    return TaskModuleNode(*nodes)

# FUNCTION TO CREATE AN IDENTITY PROCESSOR NODE AND ATTACH TO PROVIDED NODE
def create_identity_processor_node(node):
    return IdentityProcessor()(node)

# FUNCTION TO CREATE A JOINER PROCESSOR NODE AND ATTACH TO PROVIDED NODES
def create_joiner_processor_node(*nodes):
    return JoinerProcessor()(*nodes)

# TEST FUNCTION TO TEST CREATION OF SIMPLE TASK MODULE AND TESTING IT IN A FLOW
def test_taskmodule_node():

    # create a simple module
    zero = IntProducer()
    identity_processor_a = create_identity_processor_node(zero)
    identity_processor_b = create_identity_processor_node(identity_processor_a)
    identity_processor_c = create_identity_processor_node(identity_processor_b)
    joiner_processor_d = create_joiner_processor_node(identity_processor_b, identity_processor_c)
    identity_processor_e = create_identity_processor_node(joiner_processor_d)
    joiner_processor_f = create_joiner_processor_node(joiner_processor_d, identity_processor_e, identity_processor_c, identity_processor_b)
    module = create_task_module_node(identity_processor_a, joiner_processor_f)

    # test error is raised when trying to use module outside of flow
    with pytest.raises(RuntimeError):
        out1 = CommandlineConsumer()(joiner_processor_f)

    # create graph engine and test topological order is correct
    out = CommandlineConsumer()(module)
    graph_engine = GraphEngine([zero], [out])
    tsort = graph_engine.topological_sort()
    assert len(tsort) == 3

# TEST FUNCTION TO TEST THAT TASK MODULE CAN CREATE ITS OWN PARENTS WITHOUT NEEDING THE ENTRY NODE
def test_taskmodule_node_1():

    # create a task module with its own parents
    zero = IntProducer()
    identity_processor_a = create_identity_processor_node(zero)
    identity_processor_b = create_identity_processor_node(identity_processor_a)
    identity_processor_c = create_identity_processor_node(identity_processor_b)
    task_module = create_task_module_node(identity_processor_a, identity_processor_c)(zero)
    
    # test graph engine and topological sort is correct
    out = CommandlineConsumer()(task_module)
    graph_engine = GraphEngine([zero], [out])
    tsort = graph_engine.topological_sort()
    assert len(tsort) == 3

# TEST FUNCTION TO TEST THAT TASK MODULE CAN TAKE CHILD NODES FROM ITS EXIT_ENTRY
def test_taskmodule_node_2():
    
    # create a task module and attach a consumer
    zero = IntProducer()
    identity_processor_a = create_identity_processor_node(zero)
    identity_processor_b = create_identity_processor_node(identity_processor_a)
    identity_processor_c = create_identity_processor_node(identity_processor_b)
    out = CommandlineConsumer()(identity_processor_c)
    task_module = create_task_module_node(identity_processor_a, identity_processor_c)

    # test graph engine and topological sort is correct
    graph_engine = GraphEngine([zero], [out])
    tsort = graph_engine.topological_sort()
    assert len(tsort) == 3
    assert task_module in tsort

# TEST FUNCTION TO TEST ERROR WHEN TRYING TO PUT MODULE INSIDE OF MODULE
def test_taskmodule_node_3():

    # create a task module and another module to put inside of it, then raise error
    zero = IntProducer()
    identity_processor_a = create_identity_processor_node(zero)
    identity_processor_b = create_identity_processor_node(identity_processor_a)
    identity_processor_c = create_identity_processor_node(identity_processor_b)
    joiner_processor_d = create_joiner_processor_node(identity_processor_b, identity_processor_c)
    identity_processor_e = create_identity_processor_node(joiner_processor_d)
    joiner_processor_f = create_joiner_processor_node(joiner_processor_d, identity_processor_e, identity_processor_c, identity_processor_b)
    module = create_task_module_node(identity_processor_a, joiner_processor_f)

    identity_processor_g = create_identity_processor_node(module)
    identity_processor_h = create_identity_processor_node(identity_processor_g)
    identity_processor_i = create_identity_processor_node(identity_processor_h)
    with pytest.raises(ValueError):
        module1 = create_task_module_node(module, identity_processor_i)

# TEST FUNCTION TO TEST THE PROCESS ALGORITHM OF THE TASKMODULE NODE
def test_taskmodule_node_4():
    pass

# RUN ALL TEST FUNCTIONS
if __name__ == "__main__":
    pytest.main([__file__])