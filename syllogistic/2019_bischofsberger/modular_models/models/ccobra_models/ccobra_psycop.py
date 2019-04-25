import ccobra
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.models.basic_models import PSYCOP
from modular_models.models.ccobra_models.interface import CCobraWrapper
from modular_models.util import sylutil

PSYCOP_CACHE_DAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cached_psycop_prediction.dat")


class CCobraPSYCOP(CCobraWrapper, ccobra.CCobraModel):
    def __init__(self):
        CCobraWrapper.__init__(self, model=PSYCOP)
        ccobra.CCobraModel.__init__(self, "PSYCOP", ["syllogistic"], ["single-choice"])

    @sylutil.persistent_memoize(PSYCOP_CACHE_DAT)
    def generate_predictions(self):
        return CCobraWrapper.generate_predictions(self)
