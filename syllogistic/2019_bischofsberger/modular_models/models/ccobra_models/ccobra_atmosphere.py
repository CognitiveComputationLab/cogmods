import ccobra
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.models.basic_models import Atmosphere
from modular_models.models.ccobra_models.interface import CCobraWrapper
from modular_models.util import sylutil

ATMOSPHERE_CACHE_DAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cached_atmosphere_predictions.dat")


class CCobraAtmosphere(CCobraWrapper, ccobra.CCobraModel):
    def __init__(self):
        CCobraWrapper.__init__(self, model=Atmosphere)
        ccobra.CCobraModel.__init__(self, "Atmosphere", ["syllogistic"], ["single-choice"])

    @sylutil.persistent_memoize(ATMOSPHERE_CACHE_DAT)
    def generate_predictions(self):
        return CCobraWrapper.generate_predictions(self)
