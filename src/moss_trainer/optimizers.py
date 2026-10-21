"""Native Muon/AdamW with one scheduler and an ordered checkpoint state."""
from __future__ import annotations

import torch


HYBRID_FIELDS = (
    "muon_momentum", "muon_nesterov", "muon_ns_steps", "muon_adjust_lr",
    "hybrid_adam_beta1", "hybrid_adam_beta2", "hybrid_adam_eps",
)


class HybridMuonAdamW(torch.optim.Optimizer):
    """Keep contiguous role groups so adapter parameter order never changes.

    Children share the parent's group dictionaries and state. After loading,
    rebind both: Optimizer.load_state_dict replaces these objects.
    """

    def __init__(self, named_parameters, training):
        groups = []
        seen = set()
        matrix_names = {"lokr_w1", "lokr_w2", "lokr_w1_a", "lokr_w1_b",
                        "lokr_w2_a", "lokr_w2_b"}
        for name, parameter in named_parameters:
            if not parameter.requires_grad:
                continue
            if id(parameter) in seen:
                raise ValueError(f"Duplicate optimizer parameter: {name}")
            seen.add(id(parameter))
            matrix = parameter.ndim == 2 and (
                name.rsplit(".", 1)[-1] in matrix_names
                or name.endswith((".lora_up.weight", ".lora_down.weight")))
            auxiliary = name.rsplit(".", 1)[-1] in {"w_norm", "scalar", "dora_scale", "bias"}
            if not matrix and not auxiliary:
                raise ValueError(f"Unsupported hybrid adapter parameter: {name} {tuple(parameter.shape)}")
            role = "muon" if matrix else "adamw"
            if not groups or groups[-1]["algorithm"] != role:
                groups.append({"params": [], "param_names": [], "algorithm": role,
                               "lr": training.learning_rate,
                               "weight_decay": training.weight_decay if matrix else 0.0})
            groups[-1]["params"].append(parameter)
            groups[-1]["param_names"].append(name)
        if not any(g["algorithm"] == "muon" for g in groups):
            raise ValueError("Hybrid Muon requires trainable LoRA/LoKr matrices")
        super().__init__(groups, {"lr": training.learning_rate})
        self._children = {}
        for role in ("muon", "adamw"):
            selected = [g for g in self.param_groups if g["algorithm"] == role]
            if not selected:
                continue
            if role == "muon":
                child = torch.optim.Muon(
                    selected, momentum=training.muon_momentum,
                    nesterov=training.muon_nesterov, ns_steps=training.muon_ns_steps,
                    adjust_lr_fn=training.muon_adjust_lr)
            else:
                child = torch.optim.AdamW(
                    selected, betas=(training.hybrid_adam_beta1, training.hybrid_adam_beta2),
                    eps=training.hybrid_adam_eps, foreach=False)
            child.state = self.state
            self._children[role] = child

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for child in self._children.values():
            child.step()
        return loss

    def load_state_dict(self, state_dict):
        saved = state_dict.get("param_groups", [])
        if len(saved) != len(self.param_groups) or any(
            old.get("algorithm") != new["algorithm"]
            or old.get("param_names") != new["param_names"]
            for old, new in zip(saved, self.param_groups)
        ):
            raise ValueError("Hybrid optimizer parameter roles or names changed")
        super().load_state_dict(state_dict)
        for role, child in self._children.items():
            child.param_groups = [g for g in self.param_groups if g["algorithm"] == role]
            child.state = self.state
