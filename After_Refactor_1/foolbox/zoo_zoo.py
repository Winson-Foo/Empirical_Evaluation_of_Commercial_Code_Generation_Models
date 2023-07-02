from typing import Any

from ..models import Model
from .git_cloner import clone
from .model_loader import ModelLoader


def get_model(url: str, module_name: str = "foolbox_model", overwrite: bool = False, **model_params: Any) -> Model:
    """Download a Foolbox-compatible model from the given Git repository URL.

    Args:
        url: URL of the git repository.
        module_name: Name of the module to import.
        overwrite: Whether to overwrite an existing repository if it exists.
        **model_params: Optional set of parameters to be used by the instantiated model.

    Returns:
        A Foolbox-wrapped model instance.
    """
    repo_path: str = clone(url, overwrite=overwrite)
    model_loader = ModelLoader.get()
    model = model_loader.load(repo_path, module_name=module_name, **model_params)
    return model