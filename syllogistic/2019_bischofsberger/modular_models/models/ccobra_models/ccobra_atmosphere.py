import ccobra
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.models.basic_models import Atmosphere
from modular_models.models.ccobra_models.interface import CCobraWrapper


class CCobraAtmosphere(CCobraWrapper, ccobra.CCobraModel):
    def __init__(self):
        CCobraWrapper.__init__(self, model=Atmosphere)
        ccobra.CCobraModel.__init__(self, "Atmosphere", ["syllogistic"], ["single-choice"])
