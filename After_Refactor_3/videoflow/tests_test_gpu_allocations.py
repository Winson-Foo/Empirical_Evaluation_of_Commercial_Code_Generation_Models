import pytest
import time

from videoflow.core.flow import _task_data_from_node_tsort
from videoflow.engines.realtime import RealtimeExecutionEngine
from videoflow.utils.graph import topological_sort
from videoflow.core.node import TaskModuleNode, ProcessorNode
from videoflow.core.constants import CPU, GPU
from videoflow.producers import IntProducer
from videoflow.processors import IdentityProcessor, JoinerProcessor
from videoflow.consumers import CommandlineConsumer
import videoflow.utils.system

def test_number_of_tasks_created():
    # Test that the number of tasks created is equal to the number of nodes
    producer = IntProducer()
    identity1 = IdentityProcessor()(producer)
    identity2 = IdentityProcessor()(identity1)
    joiner = JoinerProcessor()(identity1, identity2)

    tsort = topological_sort([producer])
    tasks_data = _task_data_from_node_tsort(tsort)

    ee = RealtimeExecutionEngine()
    ee._al_create_processes(tasks_data)

    assert len(tsort) == len(ee._procs)

def test_number_of_tasks_created_with_task_module_processor():
    # Test that the number of tasks created is different than the number of nodes, in the case of TaskModuleProcessor
    zero = IntProducer()
    identity1 = IdentityProcessor()(zero)
    identity2 = IdentityProcessor()(identity1)
    identity3 = IdentityProcessor()(identity2)
    joiner = JoinerProcessor()(identity2, identity3)
    identity4 = IdentityProcessor()(joiner)
    joiner2 = JoinerProcessor()(joiner, identity4, identity3, identity2)
    module_node = TaskModuleNode(identity1, joiner2)
    consumer = CommandlineConsumer()(module_node)

    tsort = topological_sort([zero])
    tasks_data = _task_data_from_node_tsort(tsort)

    ee = RealtimeExecutionEngine()
    ee._al_create_processes(tasks_data)

    assert len(ee._procs) == 3

class IdentityProcessorGpuOnly(ProcessorNode):
    def __init__(self, fps=-1):
        super().__init__(device_type=GPU)
        if fps > 0:
            self._wts = 1.0 / fps  # wait time in seconds
        else:
            self._wts = 0

    def process(self, inp):
        if self._wts > 0:
            time.sleep(self._wts)
        return inp

    def change_device(self, device_type):
        if device_type == CPU:
            raise ValueError('Cannot allocate to CPU')

@pytest.fixture
def available_gpus(monkeypatch):
    def mock_gpus():
        return [0, 1, 3]

    monkeypatch.setattr(videoflow.engines.realtime, 'get_gpus_available_to_process', mock_gpus)

def test_gpu_nodes_accepted(available_gpus):
    # Test that gpu nodes are accepted by having same number of gpu processes as gpus in the system
    producer = IntProducer()
    identity_gpu1 = IdentityProcessorGpuOnly()(producer)
    identity_gpu2 = IdentityProcessorGpuOnly()(identity_gpu1)
    identity_gpu3 = IdentityProcessorGpuOnly()(identity_gpu2)
    joiner_gpu = JoinerProcessor()(identity_gpu3)
    joiner_cpu = JoinerProcessor()(identity_gpu1, identity_gpu2, joiner_gpu, identity_gpu3)

    tsort = topological_sort([producer])
    tasks_data = _task_data_from_node_tsort(tsort)

    ee = RealtimeExecutionEngine()
    ee._al_create_processes(tasks_data)

    assert len(ee._procs) == 3

    # Test that gpu nodes are accepted by having nodes not thrown an error if gpu is not available
    producer = IntProducer()
    identity1 = IdentityProcessor(device_type=GPU)(producer)
    identity2 = IdentityProcessor(device_type=GPU)(identity1)
    identity3 = IdentityProcessor(device_type=GPU)(producer)
    joiner = JoinerProcessor(device_type=GPU)(identity2, identity3)
    joiner_cpu = JoinerProcessor(device_type=CPU)(identity1, identity2, joiner)

    tsort = topological_sort([producer])
    tasks_data = _task_data_from_node_tsort(tsort)

    ee = RealtimeExecutionEngine()
    ee._al_create_processes(tasks_data)

def test_gpu_nodes_not_accepted():
    # Test that gpu node rejects because already all gpus were allocated to other nodes
    def mock_gpus():
        return [0, 1]

    A = IntProducer()
    B = IdentityProcessorGpuOnly()(A)
    C = IdentityProcessorGpuOnly()(B)
    D = IdentityProcessorGpuOnly()(C)
    E = JoinerProcessor()(D)
    F = JoinerProcessor()(B, C, D, E)

    tsort = topological_sort([A])
    tasks_data = _task_data_from_node_tsort(tsort)

    ee = RealtimeExecutionEngine()
    with pytest.raises(RuntimeError):
        ee._al_create_processes(tasks_data)

    monkeypatch.setattr(videoflow.utils.system, 'get_gpus_available_to_process', mock_gpus)