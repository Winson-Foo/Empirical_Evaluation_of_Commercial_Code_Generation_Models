'''
Tests multiple situations regarding allocation to gpu.
'''
import pytest
import time

from videoflow.core.flow import task_data_from_node_tsort
from videoflow.engines.realtime import RealtimeExecutionEngine
from videoflow.utils.graph import topological_sort
from videoflow.core.node import TaskModuleNode, ProcessorNode
from videoflow.core.constants import CPU, GPU
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.consumers import CommandlineConsumer

import videoflow.utils.system

# Constants
GPU_AVAILABLE = [0, 1, 3]
FPS = 25

# Fixtures
@pytest.fixture(scope='module')
def execution_engine():
    return RealtimeExecutionEngine()

@pytest.fixture(scope='module')
def gpu_identity_processor():
    return IdentityProcessorGpuOnly(fps=FPS)

@pytest.fixture(scope='module')
def cpu_identity_processor():
    return IdentityProcessor(device_type=CPU)

# Tests
def test_number_of_tasks_created():
    # Test that the number of tasks created is equal to number of nodes
    producer = IntProducer()
    identity1 = IdentityProcessor()(producer)
    identity2 = IdentityProcessor()(identity1)
    joiner = JoinerProcessor()(identity1, identity2)
    final_joiner = JoinerProcessor()(producer, identity1, identity2, joiner)

    tsort = topological_sort([producer])
    tasks_data = task_data_from_node_tsort(tsort)

    engine = RealtimeExecutionEngine()
    engine._al_create_processes(tasks_data)
    assert len(tsort) == len(engine._procs)

def test_number_of_tasks_created_with_module_processor():
    # Test that number of tasks created is different than number of
    # nodes, in the case of TaskModuleProcessor
    zero = IntProducer()
    a = IdentityProcessor()(zero)
    b = IdentityProcessor()(a)
    c = IdentityProcessor()(b)
    d = JoinerProcessor()(b, c)
    e = IdentityProcessor()(d)
    f = JoinerProcessor()(d, e, c, b)
    module = TaskModuleNode(a, f)
    out = CommandlineConsumer()(module)

    tsort = topological_sort([zero])
    tasks_data = task_data_from_node_tsort(tsort)

    engine = RealtimeExecutionEngine()
    engine._al_create_processes(tasks_data)
    assert len(engine._procs) == 3

def test_gpu_nodes_accepted(monkeypatch, execution_engine, gpu_identity_processor):
    # Test that gpu nodes are accepted by having same number of gpu 
    # processes as gpus in the system
    def gpus_mock():
        return GPU_AVAILABLE

    monkeypatch.setattr(videoflow.engines.realtime, 'get_gpus_available_to_process', gpus_mock)

    producer = IntProducer()
    identity1 = gpu_identity_processor(producer)
    identity2 = gpu_identity_processor(identity1)
    identity3 = gpu_identity_processor(identity2)
    joiner = JoinerProcessor()(identity3)
    final_joiner = JoinerProcessor()(identity1, identity2, identity3, joiner)

    tsort = topological_sort([producer])
    tasks_data = task_data_from_node_tsort(tsort)

    engine = execution_engine
    engine._al_create_processes(tasks_data)

    # Test that gpu nodes are accepted by having nodes not thrown an 
    #error if gpu is not available
    producer = IntProducer()
    identity1 = IdentityProcessor(device_type=GPU)(producer)
    identity2 = IdentityProcessor(device_type=GPU)(identity1)
    identity3 = IdentityProcessor(device_type=GPU)(producer)
    joiner = JoinerProcessor(device_type=GPU)(identity2)
    final_joiner = JoinerProcessor(device_type=GPU)(identity1, identity2, identity3, joiner)

    tsort = topological_sort([producer])
    tasks_data = task_data_from_node_tsort(tsort)

    engine = execution_engine
    engine._al_create_processes(tasks_data)

def test_gpu_nodes_not_accepted(monkeypatch, gpu_identity_processor):
    # Test that gpu node rejects because already all gpus were allocated
    # to other nodes.
    def gpus_mock():
        return [0, 1]

    monkeypatch.setattr(videoflow.utils.system, 'get_gpus_available_to_process', gpus_mock)

    producer = IntProducer()
    identity1 = gpu_identity_processor(producer)
    identity2 = gpu_identity_processor(identity1)
    identity3 = gpu_identity_processor(identity2)
    joiner = JoinerProcessor()(identity3)
    final_joiner = JoinerProcessor()(identity1, identity2, identity3, joiner)

    tsort = topological_sort([producer])
    tasks_data = task_data_from_node_tsort(tsort)

    engine = RealtimeExecutionEngine()
    with pytest.raises(RuntimeError):
        engine._al_create_processes(tasks_data)

# Helper classes
class IdentityProcessorGpuOnly(ProcessorNode):
    def __init__(self, fps=-1):
        super().__init__(device_type=GPU)
        if fps > 0:
            self._wts = 1.0 / fps # wait time in seconds
        else:
            self._wts = 0

    def process(self, inp):
        if self._wts > 0:
            time.sleep(self._wts)
        return inp
    
    def change_device(self, device_type):
        if device_type == CPU:
            raise ValueError('Cannot allocate to CPU')

if __name__ == "__main__":
    pytest.main([__file__])