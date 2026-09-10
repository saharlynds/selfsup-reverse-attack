"""Use pretrained RobustBench classifiers as backbones."""
import os

import torch.nn as nn

ROBUSTBENCH_PIP = 'pip install "setuptools<82" git+https://github.com/RobustBench/robustbench.git'
ROBUSTBENCH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "robustbench"
)


class LogitsOnlyWrapper(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if not linears:
            raise ValueError(f"{type(model).__name__} has no nn.Linear to read penultimate features from")
        self.model = model
        self.feat_dim = linears[-1].in_features
        self._feats = None
        linears[-1].register_forward_pre_hook(self._capture)

    def _capture(self, module, inputs):
        self._feats = inputs[0]

    def forward(self, x):
        self._feats = None
        logits = self.model(x)
        feats, self._feats = self._feats, None
        if feats is None:
            raise RuntimeError("the last nn.Linear did not run in forward(); cannot capture features")
        return logits, feats


def load_robustbench(model_name: str, dataset: str = "cifar10", threat_model: str = "Linf"):
    """Load a pretrained RobustBench model, wrapped and in eval mode."""
    try:
        from robustbench.utils import load_model
    except ImportError as e:
        raise ImportError(
            f"'robustbench:' backbones need the optional robustbench package, which could not be "
            f"imported ({e}).\nInstall it with:  {ROBUSTBENCH_PIP}"
        ) from e
    model = load_model(model_name=model_name, model_dir=ROBUSTBENCH_DIR, dataset=dataset,
                       threat_model=threat_model)
    return LogitsOnlyWrapper(model).eval()


def robustbench_arch(model_name: str, threat_model: str = "Linf") -> str:
    """Arch string for the checkpoint; the threat model is added only if it is not Linf."""
    return f"robustbench:{model_name}" + ("" if threat_model == "Linf" else f":{threat_model}")


def build_robustbench_backbone(arch: str):
    """Rebuild the wrapped model from a 'robustbench:<model>[:<threat_model>]' arch string."""
    _, model_name, *threat = arch.split(":")
    return load_robustbench(model_name, "cifar10", threat[0] if threat else "Linf")
