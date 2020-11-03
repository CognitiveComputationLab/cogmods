import os
import sys

import ccobra

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.models.basic_models import MReasoner
from modular_models.models.ccobra_models.interface import CCobraWrapper


class CCobraMReasoner(CCobraWrapper, ccobra.CCobraModel):
    def __init__(self):
        CCobraWrapper.__init__(self, model=MReasoner)
        ccobra.CCobraModel.__init__(self, "MReasoner", ["syllogistic-generalized"], ["single-choice"])
