# import modules
import pytest
from videoflow.core import Flow
from videoflow.core.node import TaskModuleNode
from videoflow.producers import IntProducer, ProducerBase
from videoflow.processors import IdentityProcessor, JoinerProcessor, ProcessorBase
from videoflow.processors.aggregators import SumAggregator
from videoflow.consumers import CommandlineConsumer, ConsumerBase

# define test classes
class VideoFlowTestBase:
    producer: ProducerBase
    processor_list: list[ProcessorBase]
    consumer: ConsumerBase

    def run_test(self):
        flow = Flow([self.producer], [self.consumer])
        for processor in self.processor_list:
            processor(flow)
        flow.join()

class TestSimpleExample1(VideoFlowTestBase):
    def __init__(self):
        self.producer = IntProducer(0, 40, 0.1)
        self.processor_list = [
            IdentityProcessor(),
            IdentityProcessor(),
            JoinerProcessor()
        ]
        self.consumer = CommandlineConsumer()

class TestSimpleExample2(VideoFlowTestBase):
    def __init__(self):
        self.producer = IntProducer(0, 40, 0.01)
        self.processor_list = [SumAggregator()]
        self.consumer = CommandlineConsumer()

class TestMpExample1(VideoFlowTestBase):
    def __init__(self):
        self.producer = IntProducer(0, 40, 0.1)
        self.processor_list = [
            IdentityProcessor(nb_tasks = 5),
            IdentityProcessor(nb_tasks = 5),
            JoinerProcessor(nb_tasks = 5)
        ]
        self.consumer = CommandlineConsumer()

class TestTaskModuleNodeExample1(VideoFlowTestBase):
    def __init__(self):
        self.producer = IntProducer(0, 40, 0.05)
        identity = IdentityProcessor(nb_tasks = 1)
        identity1 = IdentityProcessor(nb_tasks = 1)
        joined = JoinerProcessor(nb_tasks = 1)
        task_module = TaskModuleNode(identity, joined)
        self.processor_list = [identity, identity1, joined, task_module]
        self.consumer = CommandlineConsumer()

class TestGraphWithDeadEndProcessor(VideoFlowTestBase):
    def __init__(self):
        self.producer = IntProducer(0, 40, 0.05)
        identity = IdentityProcessor(nb_tasks = 1)
        identity1 = IdentityProcessor(nb_tasks = 1)
        joined = JoinerProcessor(nb_tasks = 1)
        task_module = TaskModuleNode(identity, joined)
        dead_end = IdentityProcessor()
        self.processor_list = [identity, identity1, joined, task_module, dead_end]
        self.consumer = CommandlineConsumer()

class TestGraphWithNoConsumer(VideoFlowTestBase):
    def __init__(self):
        self.producer = IntProducer(0, 40, 0.05)
        identity = IdentityProcessor(nb_tasks = 1)
        identity1 = IdentityProcessor(nb_tasks = 1)
        joined = JoinerProcessor(nb_tasks = 1)
        task_module = TaskModuleNode(identity, joined)
        self.processor_list = [identity, identity1, joined, task_module]
        self.consumer = None

    def run_test(self):
        flow = Flow([self.producer], [])
        for processor in self.processor_list:
            processor(flow)
        flow.join()

# define pytest functions
@pytest.mark.timeout(30)
def test_simple_example1():
    TestSimpleExample1().run_test()

@pytest.mark.timeout(30)
def test_simple_example2():
    TestSimpleExample2().run_test()

@pytest.mark.timeout(30)
def test_mp_example1():
    TestMpExample1().run_test()

@pytest.mark.timeout(30)
def test_taskmodulenode_example1():
    TestTaskModuleNodeExample1().run_test()

@pytest.mark.timeout(30)
def test_graph_with_deadend_processor():
    TestGraphWithDeadEndProcessor().run_test()

@pytest.mark.timeout(30)
def test_graph_with_no_consumer():
    TestGraphWithNoConsumer().run_test()

# call pytest main
if __name__ == "__main__":
    pytest.main([__file__])
