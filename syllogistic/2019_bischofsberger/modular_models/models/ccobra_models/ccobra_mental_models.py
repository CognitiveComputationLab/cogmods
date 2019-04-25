import ccobra
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.models.basic_models import MentalModels
from modular_models.models.ccobra_models.interface import CCobraWrapper
from modular_models.util import sylutil

MM_CACHE_DAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cached_mm_prediction.dat")


class CCobraMentalModels(CCobraWrapper, ccobra.CCobraModel):
    def __init__(self):
        CCobraWrapper.__init__(self, model=MentalModels)
        ccobra.CCobraModel.__init__(self, "Mental Models", ["syllogistic"], ["single-choice"])

    @sylutil.persistent_memoize(MM_CACHE_DAT)
    def generate_predictions(self):
        return CCobraWrapper.generate_predictions(self)
