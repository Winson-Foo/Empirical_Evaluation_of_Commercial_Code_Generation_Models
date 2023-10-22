import pytest

from videoflow.core import Flow
from videoflow.core.node import TaskModuleNode
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.processors.aggregators import SumAggregator
from videoflow.consumers import CommandlineConsumer


def run_flow(producer, consumer):
    flow = Flow([producer], [consumer])
    flow.run()
    flow.join()


def test_simple_examples():
    producer = IntProducer(0, 40, 0.1)
    identity1 = IdentityProcessor()(producer)
    identity2 = IdentityProcessor()(identity1)
    joined = JoinerProcessor()(identity1, identity2)
    printer = CommandlineConsumer()(joined)
    run_flow(producer, printer)

    producer = IntProducer(0, 40, 0.01)
    sum_agg = SumAggregator()(producer)
    printer = CommandlineConsumer()(sum_agg)
    run_flow(producer, printer)


def test_mp_example():
    producer = IntProducer(0, 40, 0.1)
    identity1 = IdentityProcessor(nb_tasks=5)(producer)
    identity2 = IdentityProcessor(nb_tasks=5)(identity1)
    joined = JoinerProcessor(nb_tasks=5)(identity1, identity2)
    printer = CommandlineConsumer()(joined)
    run_flow(producer, printer)


def test_taskmodulenode_example():
    producer = IntProducer(0, 40, 0.05)
    identity1 = IdentityProcessor(nb_tasks=1)(producer)
    identity2 = IdentityProcessor(nb_tasks=1)(identity1)
    joined = JoinerProcessor(nb_tasks=1)(identity1, identity2)
    task_module = TaskModuleNode(identity1, joined)
    printer = CommandlineConsumer()(task_module)
    run_flow(producer, printer)


def test_graph_with_deadend_processor():
    producer = IntProducer(0, 40, 0.05)
    identity1 = IdentityProcessor(nb_tasks=1)(producer)
    identity2 = IdentityProcessor(nb_tasks=1)(identity1)
    joined = JoinerProcessor(nb_tasks=1)(identity1, identity2)
    task_module = TaskModuleNode(identity1, joined)
    dead_end = IdentityProcessor()(task_module)
    printer = CommandlineConsumer()(task_module)
    run_flow(producer, printer)


def test_graph_with_no_consumer():
    producer = IntProducer(0, 40, 0.05)
    identity1 = IdentityProcessor(nb_tasks=1)(producer)
    identity2 = IdentityProcessor(nb_tasks=1)(identity1)
    joined = JoinerProcessor(nb_tasks=1)(identity1, identity2)
    task_module = TaskModuleNode(identity1, joined)
    run_flow(producer, None)


if __name__ == "__main__":
    pytest.main([__file__])