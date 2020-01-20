from .base import *
from .vanilla_vae import *
from .gamma_vae import *
from .beta_vae import *
from .wae_mmd import *
from .cvae import *
from .hvae import *
from .vampvae import *

# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE
CVAE = ConditionalVAE

vae_models = {'VanillaVAE':VanillaVAE,
              'WAE_MMD':WAE_MMD,
              'ConditionalVAE':ConditionalVAE,
              'BetaVAE':BetaVAE,
              'GammaVAE':GammaVAE,
              'HVAE':HVAE}