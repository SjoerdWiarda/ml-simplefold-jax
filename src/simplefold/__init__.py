#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

__version__ = "0.1.0"
import sys

import simplefold.model as _model

# Needed for hydra and the config definitions when running tests
sys.modules["model"] = _model
