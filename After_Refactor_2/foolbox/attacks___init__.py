from .base import Attack
from .contrast import L2ContrastReductionAttack
from .virtual_adversarial_attack import VirtualAdversarialAttack
from .ddn import DDNAttack
from .projected_gradient_descent import (
    L1ProjectedGradientDescentAttack,
    L2ProjectedGradientDescentAttack,
    LinfProjectedGradientDescentAttack,
    L1AdamProjectedGradientDescentAttack,
    L2AdamProjectedGradientDescentAttack,
    LinfAdamProjectedGradientDescentAttack,
)
from .basic_iterative_method import (
    L1BasicIterativeAttack,
    L2BasicIterativeAttack,
    LinfBasicIterativeAttack,
    L1AdamBasicIterativeAttack,
    L2AdamBasicIterativeAttack,
    LinfAdamBasicIterativeAttack,
)
from .fast_gradient_method import (
    L1FastGradientAttack,
    L2FastGradientAttack,
    LinfFastGradientAttack,
)
from .additive_noise import (
    L2AdditiveGaussianNoiseAttack,
    L2AdditiveUniformNoiseAttack,
    L2ClippingAwareAdditiveGaussianNoiseAttack,
    L2ClippingAwareAdditiveUniformNoiseAttack,
    LinfAdditiveUniformNoiseAttack,
    L2RepeatedAdditiveGaussianNoiseAttack,
    L2RepeatedAdditiveUniformNoiseAttack,
    L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack,
    L2ClippingAwareRepeatedAdditiveUniformNoiseAttack,
    LinfRepeatedAdditiveUniformNoiseAttack,
)
from .sparse_l1_descent_attack import SparseL1DescentAttack
from .inversion import InversionAttack
from .contrast_min import (
    BinarySearchContrastReductionAttack,
    LinearSearchContrastReductionAttack,
)
from .carlini_wagner import L2CarliniWagnerAttack
from .newtonfool import NewtonFoolAttack
from .ead import EADAttack
from .blur import GaussianBlurAttack
from .spatial_attack import SpatialAttack
from .deepfool import L2DeepFoolAttack, LinfDeepFoolAttack
from .saltandpepper import SaltAndPepperNoiseAttack
from .blended_noise import LinearSearchBlendedUniformNoiseAttack
from .binarization import BinarizationRefinementAttack
from .dataset_attack import DatasetAttack
from .boundary_attack import BoundaryAttack
from .hop_skip_jump import HopSkipJumpAttack
from .brendel_bethge import (
    L0BrendelBethgeAttack,
    L1BrendelBethgeAttack,
    L2BrendelBethgeAttack,
    LinfinityBrendelBethgeAttack,
)
from .fast_minimum_norm import (
    L0FMNAttack,
    L1FMNAttack,
    L2FMNAttack,
    LInfFMNAttack,
)
from .gen_attack import GenAttack
from .pointwise import PointwiseAttack

# Assign aliases for easier access
FGM = L2FastGradientAttack
FGSM = LinfFastGradientAttack
L1PGD = L1ProjectedGradientDescentAttack
L2PGD = L2ProjectedGradientDescentAttack
LinfPGD = LinfProjectedGradientDescentAttack
PGD = LinfPGD

L1AdamPGD = L1AdamProjectedGradientDescentAttack
L2AdamPGD = L2AdamProjectedGradientDescentAttack
LinfAdamPGD = LinfAdamProjectedGradientDescentAttack
AdamPGD = LinfAdamPGD