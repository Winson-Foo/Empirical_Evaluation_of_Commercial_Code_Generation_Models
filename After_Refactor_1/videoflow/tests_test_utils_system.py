import pytest
import videoflow.utils.system as system

@pytest.mark.parametrize("environment, available_gpus", [
    ("", []),
    ("0", []),
    ("1, 2", [1]),
    ("2, 3", []),
    ("asdfa, 1, 0, asdf", [1, 0])
])
def test_gpus_available(monkeypatch, environment, available_gpus):
    def get_system_gpus_mock():
        return set([0, 1])

    monkeypatch.setattr(system, 'get_system_gpus', get_system_gpus_mock)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', environment)

    gpus = system.get_gpus_available_to_process()

    assert len(gpus) == len(available_gpus)
    for gpu in available_gpus:
        assert gpu in gpus

if __name__ == "__main__":
    pytest.main([__file__])
