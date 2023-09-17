from typing import Union, Any, Tuple, Generator
import eagerpy as ep

from ..criteria import Criterion
from .base import Model, Attack, T
from .spatial_attack_transformations import rotate_and_shift
from .base import get_is_adversarial, get_criterion
from .base import raise_if_kwargs, verify_input_bounds


class SpatialAttack(Attack):
    def __init__(
        self,
        max_translation: float = 3,
        max_rotation: float = 30,
        num_translations: int = 5,
        num_rotations: int = 5,
        grid_search: bool = True,
        random_steps: int = 100,
    ):
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.num_translations = num_translations
        self.num_rotations = num_rotations
        self.grid_search = grid_search
        self.random_steps = random_steps

    def __call__(
        self,
        model: Model,
        inputs: T,
        criterion: Any,
        **kwargs: Any,
    ) -> Tuple[T, T, T]:
        x, restore_type = ep.astensor_(inputs)
        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        if x.ndim != 4:
            raise NotImplementedError(
                "only implemented for inputs with two spatial dimensions (and one channel and one batch dimension)"
            )

        xp = self.run(model, x, criterion)
        success = is_adversarial(xp)

        xp_ = restore_type(xp)
        return xp_, xp_, restore_type(success)

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        verify_input_bounds(x, model)
        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)
        found = is_adversarial(x)
        results = x

        def grid_search_generator() -> Generator[Any, Any, Any]:
            d_phis = ep.linspace(-self.max_rotation, self.max_rotation, self.num_rotations)
            dxs = ep.linspace(-self.max_translation, self.max_translation, self.num_translations)
            dys = ep.linspace(-self.max_translation, self.max_translation, self.num_translations)
            for d_phi in d_phis:
                for dx in dxs:
                    for dy in dys:
                        yield d_phi, dx, dy

        def random_search_generator() -> Generator[Any, Any, Any]:
            d_phis = ep.uniform(-self.max_rotation, self.max_rotation, self.random_steps)
            dxs = ep.uniform(-self.max_translation, self.max_translation, self.random_steps)
            dys = ep.uniform(-self.max_translation, self.max_translation, self.random_steps)
            for d_phi, dx, dy in zip(d_phis, dxs, dys):
                yield d_phi, dx, dy

        generator = grid_search_generator() if self.grid_search else random_search_generator()
        for d_phi, dx, dy in generator:
            x_p = rotate_and_shift(x, translation=(dx, dy), rotation=d_phi)
            is_adv = is_adversarial(x_p)
            new_adv = ep.logical_and(is_adv, found.logical_not())

            results = ep.where(atleast_kd(new_adv, x_p.ndim), x_p, results)
            found = ep.logical_or(new_adv, found)
            if found.all():
                break
        return restore_type(results)

    def repeat(self, times: int) -> Attack:
        if self.grid_search:
            raise ValueError("repeat is not supported if attack is deterministic")
        else:
            random_steps = self.random_steps * times
            return SpatialAttack(
                max_translation=self.max_translation,
                max_rotation=self.max_rotation,
                num_translations=self.num_translations,
                num_rotations=self.num_rotations,
                grid_search=self.grid_search,
                random_steps=random_steps,
            )