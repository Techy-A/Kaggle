"""Check: features from our LoRALinear wrapper == features after the 0.592 inference code's merge_lora on the export.

Builds a tiny random Qwen3, wraps it with the training notebook's LoRALinear (random A and B), exports the adapter in
the same peft key layout as the notebook, then merges it into a fresh copy of the base with merge_lora copied from the
0.592 inference notebook, and compares pooled hidden states. Usage: python check_lora_merge.py
"""
import json, math, os, tempfile
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers import AutoModel, Qwen3Config

R, ALPHA = 16, 32
torch.manual_seed(0)
mc = Qwen3Config(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                 num_key_value_heads=2, head_dim=16)
base_sd = AutoModel.from_config(mc).state_dict()


class LoRALinear(nn.Module):   # as in ntl_twostage_lora_train.py
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(R, base.in_features) * 0.1)
        self.B = nn.Parameter(torch.randn(base.out_features, R) * 0.1)
        self.drop, self.scale = nn.Dropout(0.0), ALPHA / R

    def forward(self, x):
        return self.base(x) + (self.drop(x).to(self.A.dtype) @ self.A.T @ self.B.T).to(x.dtype) * self.scale


def fresh():
    m = AutoModel.from_config(mc)
    m.load_state_dict(base_sd)
    m.norm = nn.Identity()
    return m.eval()


wrapped, lora = fresh(), {}
for name, mod in list(wrapped.named_modules()):
    for child, sub in list(mod.named_children()):
        if isinstance(sub, nn.Linear) and name.startswith("layers."):
            setattr(mod, child, LoRALinear(sub))
            lora[f"{name}.{child}"] = getattr(mod, child)
d = tempfile.mkdtemp()
save_file({**{f"base_model.model.{k}.lora_A.weight": m.A.detach().contiguous() for k, m in lora.items()},
           **{f"base_model.model.{k}.lora_B.weight": m.B.detach().contiguous() for k, m in lora.items()}},
          os.path.join(d, "adapter_model.safetensors"))
json.dump({"r": R, "lora_alpha": ALPHA}, open(os.path.join(d, "adapter_config.json"), "w"))


@torch.no_grad()
def merge_lora(model, adapter_dir):   # verbatim from the 0.592 inference notebook
    acfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    scale = acfg["lora_alpha"] / acfg["r"]
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    modules = dict(model.named_modules())
    for k, A in sd.items():
        if ".lora_A." in k:
            name = k.split(".lora_A.")[0]
            lin = modules[name[name.index("layers."):]]
            Bm = sd[k.replace(".lora_A.", ".lora_B.")]
            lin.weight.add_(((Bm.float().to(lin.weight.device) @ A.float().to(lin.weight.device)) * scale).to(lin.weight.dtype))


merged = fresh()
merge_lora(merged, d)
x = torch.randint(0, 1000, (2, 12))
with torch.no_grad():
    a = wrapped(input_ids=x, output_hidden_states=True).hidden_states
    b = merged(input_ids=x, output_hidden_states=True).hidden_states
    c = fresh()(input_ids=x, output_hidden_states=True).hidden_states
err = max((u - v).abs().max().item() for u, v in zip(a, b))
moved = (a[-1] - c[-1]).abs().max().item()
print(f"{len(lora)} LoRA layers; max |wrapped - merged| = {err:.2e}; LoRA effect on last layer = {moved:.2e}")
assert err < 1e-4 and moved > 1e-2, "merge does not reproduce the trained model"
print("OK: the export merges exactly into the base the way the 0.592 inference code does it")
