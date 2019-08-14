import os
import sys

import ccobra

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")))
from modular_models.models.ccobra_models.tree_model_base import AbstractModelBase, BASIC_OPERATIONS


class AbstractModelComplete(AbstractModelBase, ccobra.CCobraModel):
    def __init__(self, name="Tree_Model (all operations)"):
        ccobra.CCobraModel.__init__(self, name, ["syllogistic"], ["single-choice"])
        AbstractModelBase.__init__(self, operations=BASIC_OPERATIONS, name=name)
