"""Searchable CNN architecture space.

An architecture is a plain JSON-serialisable dict:
{
  "resolution": 32,                 # variable input resolution
  "stem_channels": 24,
  "stages": [                       # variable depth = number of stages x blocks per stage
     {"blocks": 2, "channels": 32, "kernel": 3, "block": "dwsep", "act": "relu6", "stride": 1},
     ...
  ],
  "head_channels": 256              # 0 = no 1x1 head conv
}
`sample_arch(trial, space)` draws one from an Optuna trial (TPE-guided);
`random_arch(rng, space)` draws one uniformly (used for smoke tests / profiling sweeps).
"""


def _stage_strides(n):
    return [1] + [2] * (n - 1)


def sample_arch(trial, space):
    n_stages = trial.suggest_int("n_stages", *space["n_stages"])
    arch = {
        "resolution": trial.suggest_categorical("resolution", space["resolution"]),
        "stem_channels": trial.suggest_categorical("stem_channels", space["stem_channels"]),
        "head_channels": trial.suggest_categorical("head_channels", space["head_channels"]),
        "stages": [],
    }
    for i, stride in enumerate(_stage_strides(n_stages)):
        arch["stages"].append({
            "blocks": trial.suggest_int(f"s{i}_blocks", *space["blocks_per_stage"]),
            "channels": trial.suggest_categorical(f"s{i}_channels", space["channels"][i]),
            "kernel": trial.suggest_categorical(f"s{i}_kernel", space["kernels"]),
            "block": trial.suggest_categorical(f"s{i}_block", space["block_types"]),
            "act": trial.suggest_categorical(f"s{i}_act", space["activations"]),
            "stride": stride,
        })
    return arch


def random_arch(rng, space):
    n_stages = rng.randint(*space["n_stages"])
    return {
        "resolution": rng.choice(space["resolution"]),
        "stem_channels": rng.choice(space["stem_channels"]),
        "head_channels": rng.choice(space["head_channels"]),
        "stages": [{
            "blocks": rng.randint(*space["blocks_per_stage"]),
            "channels": rng.choice(space["channels"][i]),
            "kernel": rng.choice(space["kernels"]),
            "block": rng.choice(space["block_types"]),
            "act": rng.choice(space["activations"]),
            "stride": stride,
        } for i, stride in enumerate(_stage_strides(n_stages))],
    }


def describe(arch):
    s = " | ".join(f"{st['blocks']}x{st['block']}{st['kernel']}-{st['channels']}-{st['act']}"
                   for st in arch["stages"])
    return f"res{arch['resolution']} stem{arch['stem_channels']} [{s}] head{arch['head_channels']}"
