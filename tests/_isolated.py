"""Load production Python helpers without importing optional model/Triton stacks.

AST selection executes the real source, never a copied implementation. CUDA tests
use real torch CUDA; protocol tests explicitly inject a fake DMA API.
"""
import ast
import functools
import hashlib
import json
from pathlib import Path
import sys
import types
import weakref

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def selected(path, names, **namespace):
    source = ROOT / 'src' / 'moss_trainer' / path
    tree = ast.parse(source.read_text(encoding='utf-8'), str(source))
    tree.body = [n for n in tree.body if getattr(n, 'name', None) in names]
    scope = dict(torch=torch, functools=functools, weakref=weakref, **namespace)
    exec(compile(tree, str(source), 'exec'), scope)
    return types.SimpleNamespace(**scope)


def offload(**namespace):
    return selected('compilation.py', {
        '_cache_checkpoint_input', '_PooledCheckpointInput', '_offload_checkpoint_input'
    }, **namespace)


def checkpoint_helpers():
    import dataclasses
    import os
    import random
    import re
    import shutil
    import tempfile
    from safetensors.torch import load_file, save_file
    from moss_trainer.scheduling import checkpoint_name, training_plan

    # Only the adapter codec is replaced: a tiny nn.Module stands in for LyCORIS.
    def save_adapter(network, config, path, *, model_fingerprint, step, epoch):
        save_file(network.state_dict(), str(path), metadata={
            'moss_adapter_format': 'lycoris-native-v1',
            'moss_network': json.dumps(dataclasses.asdict(config)),
            'moss_step': str(step), 'moss_model_fingerprint': model_fingerprint,
        })

    source = ROOT / 'src/moss_trainer/checkpoints.py'
    names = {n.name for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef)}
    return selected('checkpoints.py', names, dataclasses=dataclasses,
                    hashlib=hashlib, json=json, os=os, Path=Path, random=random,
                    re=re, shutil=shutil, tempfile=tempfile,
                    CHECKPOINT_VERSION=2,
                    PAYLOAD_FILES=('adapter.safetensors', 'trainer-state.pt', 'config.json'),
                    checkpoint_name=checkpoint_name, training_plan=training_plan,
                    digest=lambda v: hashlib.sha256(json.dumps(v, sort_keys=True,
                                                    ensure_ascii=False).encode()).hexdigest(),
                    save_adapter=save_adapter,
                    load_adapter_weights=lambda net, path: net.load_state_dict(load_file(str(path))))
