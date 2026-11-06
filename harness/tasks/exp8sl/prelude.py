# The published Experiment 8 kernel helpers, on top of the OpenWiSH kernel (every write goes through the edit gate).
import json
import os

import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer

_np_load, _gate_save, _gate_drop = load, save_edit, drop_edit
_EDITED_DIR = "/work/edited"


def edited():
    """Names of the tensors you have edited (their current values are mirrored in /work/edited/)."""
    return sorted(f[:-4] for f in os.listdir(_EDITED_DIR) if f.endswith(".npy")) if os.path.isdir(_EDITED_DIR) else []


def load(name):
    """A float32 torch tensor: your current edit if there is one, else the original."""
    return torch.from_numpy(_np_load(name))


def load_original(name):
    with safe_open("/model/model.safetensors", framework="pt") as f:
        return f.get_tensor(name).float()


def save_edit(name, tensor, reason=""):
    r = _gate_save(name, tensor, reason)
    os.makedirs(_EDITED_DIR, exist_ok=True)
    np.save(os.path.join(_EDITED_DIR, name + ".npy"), _np_load(name))
    return r


def drop_edit(name, reason=""):
    r = _gate_drop(name, reason)
    p = os.path.join(_EDITED_DIR, name + ".npy")
    if os.path.exists(p):
        os.remove(p)
    return r


tok = Tokenizer.from_file("/model/tokenizer.json")


# --- scale-ladder addition: every save_edit needs a reason (kept in the ledger with the edit) ---
_save_edit_v3 = save_edit
MIN_REASON_WORDS = 8


def save_edit(name, tensor, reason=""):
    """Save an edit of one tensor. reason: which component you change, how, and the evidence (required)."""
    if len(str(reason).split()) < MIN_REASON_WORDS:
        raise ValueError(f"save_edit needs reason='...' of at least {MIN_REASON_WORDS} words: which component you change, "
                         "how, and what evidence makes you expect it to help. Nothing was saved.")
    return _save_edit_v3(name, tensor, reason)
