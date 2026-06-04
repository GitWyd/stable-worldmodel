---
title: 5. Planning
summary: Latent shooting from first principles, then CEM + a real success rate
sidebar_title: 5. Planning
---

# NanoJEPA — 5. Planning

A world model earns its name by *planning*: imagining the consequences of action
sequences and choosing the best one — entirely in latent space, without touching
the simulator. This chapter follows NanoJEPA's "first principles, then graduate"
arc (`wm/nanojepa/planning.py`): a hand-written shooting planner first, then the
real CEM loop wired into `stable_worldmodel`.

## Where planning happens

Planning lives in the **predictor's layer-normed target patch space** — the same
space the model was *trained* to predict (chapter 2). It is **not** the fused
`encode()` space used by the probes in chapter 4. The internals that make this
work:

- `model._encode_goal_z(goal_images)` — encode a goal image with the EMA target
  encoder, layer-norm, mean-pool → `[B, D]`.
- `model._latent_rollout(context_tokens, proprio_token, actions)` —
  autoregressively feed the predictor's `[B, N, D]` output back as the next
  context, conditioned on each action in turn; layer-norm + mean-pool the final
  state → `[B, D]`, comparable to an encoded goal.

The **goal is a rendered future frame**. We do not hand the planner a symbolic
target — we give it the *image* the camera would show at the goal, encode it into
the same latent space, and minimise the squared latent distance to it. That is
why both the current observation and the goal go through the same image
transform.

## First principles: `latent_shooting`

Random shooting is model-predictive control written out in ~20 lines: sample many
random action sequences, roll each forward in latent space, keep the one whose
predicted final state is closest to the goal embedding.

```python
import torch
import stable_worldmodel as swm
from stable_worldmodel.wm.nanojepa import NanoJEPA, latent_shooting

device = torch.device('cpu')
model, _ = NanoJEPA.from_checkpoint('checkpoints/nanojepa/epoch_10.pt', device)

# grab a current frame + a goal frame from the dataset (the goal is just a
# later frame of the same episode rendered as an image)
ds = swm.data.load_dataset(
    'nanojepa_fetchpush_quick.lance', num_steps=2,
    keys_to_load=['pixels', 'proprio', 'action'],
)
w = ds[0]
image = w['pixels'][0]          # CHW uint8 — latent_shooting accepts HWC or CHW
proprio = w['proprio'][0]
goal_image = ds[10]['pixels'][0]

env = swm.World('swm/FetchPush-v3', num_envs=1, image_shape=(64, 64))

best_actions = latent_shooting(
    model,
    image,
    proprio,
    goal_image,
    action_sampler=env.action_space.sample,   # callable() -> action [A]
    num_candidates=256,
    horizon=10,
    device=device,
)
best_actions.shape    # (10, 4) np.ndarray — the chosen action sequence
env.close()
```

This is the whole idea of latent planning. But random shooting is sample-hungry
and never refines its guesses, and we scored against the dataset's goal frame
rather than in a live environment. To get a *real* success rate we graduate to
CEM and let `stable_worldmodel` drive the environment.

## Graduating: `build_cem_policy` + `WorldModelPolicy`

NanoJEPA implements the `Costable` interface — `get_cost(info_dict,
action_candidates) -> [B, S]` — which encodes the goal, rolls the predictor
forward over every candidate, and returns the squared latent distance to the
goal. That single method is the bridge: it lets a real
`stable_worldmodel.solver.CEMSolver` optimise the exact same latent cost.

`build_cem_policy` wires it together:

```python
from stable_worldmodel.wm.nanojepa import build_cem_policy

policy = build_cem_policy(
    model,
    horizon=5,
    receding_horizon=1,
    num_samples=300,
    n_steps=10,
    topk=30,
    image_size=64,
    device=device,
)
```

Under the hood this builds a `CEMSolver(model=model, num_samples=..., n_steps=...,
topk=...)` and a `WorldModelPolicy(solver, PlanConfig(horizon, receding_horizon),
transform={'pixels': tf, 'goal': tf})`. The transform is a plain scale+resize
(`ToImage -> ToDtype(float32, scale=True) -> Resize`) with **no ImageNet
normalisation** — NanoJEPA trains from scratch on `[0, 1]` 64x64 images, unlike
the pretrained backbones in `scripts/plan/eval_wm.py`. Proprio is passed raw
(`process={}`).

CEM improves on random shooting by *fitting a distribution* to the top-`k`
candidates each iteration and resampling from it (`n_steps` times), so the search
concentrates around promising action sequences instead of guessing blindly.

## A real success rate with `World.evaluate`

`World.evaluate` has a dataset-driven mode: seed each env from a dataset episode,
start at `start_steps[i]`, target the rendered frame at `start_steps[i] +
goal_offset`, and cap each rollout at `eval_budget` steps. It injects that
rendered goal frame into the planner's `info_dict['goal']` for you, then reports
the success rate.

```python
num_envs = 4
world = swm.World(
    'swm/FetchPush-v3',
    num_envs=num_envs,
    image_shape=(64, 64),
    max_episode_steps=2 * 50,     # >= 2 * eval_budget
)
world.set_policy(policy)

results = world.evaluate(
    dataset=ds,
    episodes_idx=list(range(num_envs)),   # one dataset episode per env
    start_steps=[0] * num_envs,
    goal_offset=10,                       # goal = frame 10 steps after start
    eval_budget=50,
)
world.close()
results['success_rate']        # percent solved
```

Requirements worth remembering: `num_envs == len(episodes_idx)`, and `info_dict`
tensors arrive as `[B, S, T, ...]` (B envs, S CEM samples, T history) — `get_cost`
reads `[:, 0, -1]` to grab the current frame and the goal frame. `get_cost`
returns a 2-D `[B, S]` cost, which the solver asserts.

## From the command line

The eval script ties this together behind `--plan`:

```bash
python scripts/eval/nanojepa.py \
    --checkpoint checkpoints/nanojepa/epoch_10.pt \
    --config scripts/train/config/nanojepa_flat/quick.yaml \
    --plan
```

```
=== CEM planning ===
  success_rate: ...
```

## The throughline

The collapse experiment (chapter 3) is not academic: planning *depends* on an
uncollapsed latent space. If the embeddings collapse, every goal and every
predicted rollout map to nearly the same vector, the latent distance is flat, and
CEM has no gradient of cost to follow — the success rate goes to zero. Train with
VICReg, verify with the probe, then plan. That is the whole NanoJEPA curriculum.

← Back to [chapter 1](1_env_and_data.md) · [the model](2_model.md) ·
[training & collapse](3_train_and_collapse.md) · [evaluate](4_evaluate.md).
