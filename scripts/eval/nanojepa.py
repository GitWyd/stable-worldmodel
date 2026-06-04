"""Evaluate a NanoJEPA checkpoint: probes, collapse metrics, and CEM planning.

    # representation quality + collapse metrics
    python scripts/eval/nanojepa.py --checkpoint checkpoints/nanojepa/epoch_10.pt \
        --config scripts/train/config/nanojepa_flat/quick.yaml

    # the collapse experiment side-by-side
    python scripts/eval/nanojepa.py --config .../quick.yaml \
        --compare checkpoints/nanojepa/epoch_10.pt \
                  checkpoints/nanojepa_no_vicreg/epoch_10.pt

    # plan with CEM and report a real success rate
    python scripts/eval/nanojepa.py --checkpoint .../epoch_10.pt \
        --config .../quick.yaml --plan
"""

import argparse
from pathlib import Path

import torch
import yaml

import stable_worldmodel as swm
from stable_worldmodel.wm.nanojepa import (
    NanoJEPA,
    TransitionView,
    build_cem_policy,
    compute_collapse_metrics,
    cross_modal_eval,
    linear_probe,
)


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _resolve_data_path(name):
    p = Path(name)
    if p.is_absolute() or p.exists():
        return str(p)
    cache = swm.data.utils.get_cache_dir(sub_folder='datasets')
    return str(Path(cache) / name)


def _load_dataset(cfg, data):
    return swm.data.load_dataset(
        _resolve_data_path(data or cfg['data']['path']),
        num_steps=2,
        keys_to_load=['pixels', 'proprio', 'action'],
    )


@torch.no_grad()
def _embedding_stats(model, dataset, device, num_samples=1000):
    view = TransitionView(dataset)
    n = min(num_samples, len(view))
    imgs, pros = zip(*(view.single_step(i) for i in range(n)))
    z = []
    imgs = torch.stack(imgs)
    pros = torch.stack(pros)
    for i in range(0, n, 256):
        z.append(
            model.encode(
                imgs[i : i + 256].to(device), pros[i : i + 256].to(device)
            ).cpu()
        )
    return compute_collapse_metrics(torch.cat(z))


def evaluate_checkpoint(ckpt, dataset, device) -> dict:
    model, _ = NanoJEPA.from_checkpoint(ckpt, device)
    probe = linear_probe(model, dataset, device)
    cross = cross_modal_eval(model, dataset, device)
    collapse = _embedding_stats(model, dataset, device)
    return {
        'r2': probe['r2'],
        'mse': probe['mse'],
        'cos_vision': cross['cos_vision'],
        'cos_proprio': cross['cos_proprio'],
        'embedding_std': collapse['embedding_std_mean'],
        'effective_rank': collapse['effective_rank'],
    }


def print_summary(name, res):
    print(f'\n=== {name} ===')
    print(f'  linear-probe R^2      : {res["r2"]:.4f}')
    print(f'  cross-modal (vision)  : {res["cos_vision"]:.4f}')
    print(f'  cross-modal (proprio) : {res["cos_proprio"]:.4f}')
    print(f'  embedding std         : {res["embedding_std"]:.4f}')
    print(f'  effective rank        : {res["effective_rank"]:.1f}')


def compare(ckpt_on, ckpt_off, dataset, device):
    on = evaluate_checkpoint(ckpt_on, dataset, device)
    off = evaluate_checkpoint(ckpt_off, dataset, device)
    rows = [
        ('R^2', 'r2'),
        ('cosine_vision', 'cos_vision'),
        ('cosine_proprio', 'cos_proprio'),
        ('embedding_std', 'embedding_std'),
        ('effective_rank', 'effective_rank'),
    ]
    print('\n+--------------------+--------------+--------------+')
    print('| Metric             | VICReg ON    | VICReg OFF   |')
    print('+--------------------+--------------+--------------+')
    for label, key in rows:
        print(f'| {label:<18s} | {on[key]:>12.4f} | {off[key]:>12.4f} |')
    print('+--------------------+--------------+--------------+')


def plan(
    ckpt,
    cfg,
    dataset,
    device,
    num_envs=4,
    horizon=5,
    goal_offset=10,
    eval_budget=50,
    num_samples=300,
):
    model, _ = NanoJEPA.from_checkpoint(ckpt, device)
    policy = build_cem_policy(
        model,
        horizon=horizon,
        receding_horizon=1,
        num_samples=num_samples,
        n_steps=10,
        image_size=cfg['env']['image_size'],
        device=device,
    )
    env = cfg['env']
    world = swm.World(
        env['name'],
        num_envs=num_envs,
        image_shape=(env['image_size'], env['image_size']),
        max_episode_steps=2 * eval_budget,
    )
    world.set_policy(policy)
    results = world.evaluate(
        dataset=dataset,
        episodes_idx=list(range(num_envs)),
        start_steps=[0] * num_envs,
        goal_offset=goal_offset,
        eval_budget=eval_budget,
    )
    world.close()
    print(f'\n=== CEM planning ===\n  success_rate: {results["success_rate"]}')
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--data', default=None)
    p.add_argument('--compare', nargs=2, default=None, metavar=('ON', 'OFF'))
    p.add_argument('--plan', action='store_true')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = pick_device(args.device)
    dataset = _load_dataset(cfg, args.data)

    if args.compare:
        compare(args.compare[0], args.compare[1], dataset, device)
        return

    assert args.checkpoint, 'pass --checkpoint or --compare'
    res = evaluate_checkpoint(args.checkpoint, dataset, device)
    print_summary(args.checkpoint, res)
    if args.plan:
        plan(args.checkpoint, cfg, dataset, device)


if __name__ == '__main__':
    main()
