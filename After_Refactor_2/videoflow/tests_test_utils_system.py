import pytest
import videoflow.utils.system as system

@pytest.fixture
def monkeypatched_gpus_available(monkeypatch, available_devices):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', available_devices)
    return system.get_gpus_available_to_process()

def test_no_gpus(monkeypatched_gpus_available):
    assert len(monkeypatched_gpus_available) == 0

def test_single_gpu(monkeypatched_gpus_available):
    assert len(monkeypatched_gpus_available) == 1
    assert 0 in monkeypatched_gpus_available

def test_multiple_gpus(monkeypatched_gpus_available):
    assert len(monkeypatched_gpus_available) == 1
    assert 1 in monkeypatched_gpus_available

def test_invalid_devices(monkeypatched_gpus_available):
    assert len(monkeypatched_gpus_available) == 2

if __name__ == "__main__":
    pytest.main([__file__])
