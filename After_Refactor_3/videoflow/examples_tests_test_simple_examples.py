import pytest
from videoflow.core import Flow
from videoflow.core.node import TaskModuleNode
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.processors.aggregators import SumAggregator
from videoflow.consumers import CommandlineConsumer

def test_simple_example():
    producer = IntProducer(start=0, stop=40, interval=0.1)
    identity = IdentityProcessor()(producer)
    identity1 = IdentityProcessor()(identity)
    joined = JoinerProcessor()(identity, identity1)
    printer = CommandlineConsumer()(joined)
    flow = Flow([producer], [printer])
    flow.run()
    flow.join()

def test_mp_example():
    producer = IntProducer(start=0, stop=40, interval=0.1)
    identity = IdentityProcessor(nb_tasks=5)(producer)
    identity1 = IdentityProcessor(nb_tasks=5)(identity)
    joined = JoinerProcessor(nb_tasks=5)(identity, identity1)
    printer = CommandlineConsumer()(joined)
    flow = Flow([producer], [printer])
    flow.run()
    flow.join()

def test_taskmodulenode_example():
    producer = IntProducer(start=0, stop=40, interval=0.05)
    identity = IdentityProcessor(nb_tasks=1)(producer)
    identity1 = IdentityProcessor(nb_tasks=1)(identity)
    joined = JoinerProcessor(nb_tasks=1)(identity, identity1)
    task_module = TaskModuleNode(identity, joined)
    printer = CommandlineConsumer()(task_module)
    flow = Flow([producer], [printer])
    flow.run()
    flow.join()

def test_graph_with_deadend_processor():
    producer = IntProducer(start=0, stop=40, interval=0.05)
    identity = IdentityProcessor(nb_tasks=1)(producer)
    identity1 = IdentityProcessor(nb_tasks=1)(identity)
    joined = JoinerProcessor(nb_tasks=1)(identity, identity1)
    task_module = TaskModuleNode(identity, joined)
    dead_end = IdentityProcessor()(task_module)
    printer = CommandlineConsumer()(task_module)
    flow = Flow([producer], [printer])
    flow.run()
    flow.join()

def test_graph_with_no_consumer():
    producer = IntProducer(start=0, stop=40, interval=0.05)
    identity = IdentityProcessor(nb_tasks=1)(producer)
    identity1 = IdentityProcessor(nb_tasks=1)(identity)
    joined = JoinerProcessor(nb_tasks=1)(identity, identity1)
    task_module = TaskModuleNode(identity, joined)
    flow = Flow([producer], [])
    flow.run()
    flow.join()

def test_sum_aggregator():
    producer = IntProducer(start=0, stop=40, interval=0.01)
    sum_agg = SumAggregator()(producer)
    printer = CommandlineConsumer()(sum_agg)
    flow = Flow([producer], [printer])
    flow.run()
    flow.join()

if __name__ == "__main__":
    pytest.main([__file__])