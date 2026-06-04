"""NanoJEPA trainer — a plain-PyTorch, read-top-to-bottom training loop.

Deliberately *not* the Hydra + Lightning + stable_pretraining harness used by
``scripts/train/prejepa.py``: a learner should be able to read the whole loop.
The framework only supplies the data (``swm.data.load_dataset``) and, if asked,
the collection (``World.collect``); everything that teaches JEPA — the loss,
the EMA update, the collapse metrics — is right here and in
``stable_worldmodel.wm.nanojepa``.

    # quick smoke + collapse experiment
    python scripts/train/nanojepa.py --config scripts/train/config/nanojepa_flat/quick.yaml
    python scripts/train/nanojepa.py --config scripts/train/config/nanojepa_flat/quick.yaml --no-vicreg
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

import stable_worldmodel as swm
from stable_worldmodel.policy import RandomPolicy
from stable_worldmodel.wm.nanojepa import (
    NanoJEPA,
    compute_collapse_metrics,
    count_parameters,
    covariance_loss,
    jepa_loss,
    make_loader,
    variance_loss,
)

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')


# --------------------------------------------------------------------- helpers
def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def cosine_schedule(step, total, start, end):
    t = min(step / max(1, total), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t))


def linear_schedule(step, total, start, end):
    t = min(step / max(1, total), 1.0)
    return start + t * (end - start)


def warmup_cosine(step, total, warmup, lr, lr_min):
    if step < warmup:
        return lr * step / max(1, warmup)
    return cosine_schedule(step - warmup, total - warmup, lr, lr_min)


def resolve_data_path(name: str) -> str:
    """Resolve a dataset name to an absolute path under the datasets cache.

    Keeps ``World.collect`` (writes to the literal path) and ``load_dataset``
    (resolves a bare name against ``$STABLEWM_HOME/datasets``) consistent.
    """
    p = Path(name)
    if p.is_absolute() or p.exists():
        return str(p)
    cache = swm.data.utils.get_cache_dir(sub_folder='datasets')
    return str(Path(cache) / name)


def ensure_dataset(cfg: dict, path: str):
    """Load the dataset, collecting it with a random policy if it's missing."""
    keys = ['pixels', 'proprio', 'action']
    path = resolve_data_path(path)
    try:
        return swm.data.load_dataset(path, num_steps=2, keys_to_load=keys)
    except (FileNotFoundError, ValueError):
        env = cfg['env']
        print(f'Dataset {path!r} not found — collecting it first.')
        world = swm.World(
            env['name'],
            num_envs=8,
            image_shape=(env['image_size'], env['image_size']),
            max_episode_steps=env['max_steps'],
        )
        world.set_policy(RandomPolicy(seed=0))
        world.collect(
            path=path, episodes=env['num_episodes'], seed=0, format='lance'
        )
        world.close()
        return swm.data.load_dataset(path, num_steps=2, keys_to_load=keys)


# ----------------------------------------------------------------------- train
def train(cfg: dict, args) -> dict:
    device = pick_device(args.device)
    tcfg = cfg['training']
    use_vicreg = tcfg.get('vicreg', True) and not args.no_vicreg
    epochs = args.epochs or tcfg['epochs']

    dataset = ensure_dataset(cfg, args.data or cfg['data']['path'])
    masking_cfg = {
        'patch_size': cfg['model']['patch_size'],
        'image_size': cfg['env']['image_size'],
        **cfg.get('masking', {}),
    }
    loader = make_loader(
        dataset,
        batch_size=tcfg['batch_size'],
        masking_cfg=masking_cfg,
        num_workers=cfg['data'].get('num_workers', 0),
        pin_memory=cfg['data'].get('pin_memory', False),
    )

    model = NanoJEPA.from_config(cfg).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg['lr'],
        weight_decay=tcfg['weight_decay_start'],
        betas=(0.9, 0.999),
    )

    history = {
        k: []
        for k in (
            'loss',
            'jepa_loss',
            'var_loss',
            'cov_loss',
            'lr',
            'ema_momentum',
            'weight_decay',
            'embedding_std',
            'effective_rank',
        )
    }
    start_epoch = 0
    if args.resume:
        model, ckpt = NanoJEPA.from_checkpoint(args.resume, device)
        if ckpt.get('optimizer_state_dict'):
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        history = ckpt.get('history') or history
        start_epoch = ckpt.get('epoch', 0)

    counts = count_parameters(model)
    print(
        f'NanoJEPA — {counts["trainable"]:,} trainable / '
        f'{counts["total"]:,} total params'
    )
    print(f'Device: {device} | VICReg: {"ON" if use_vicreg else "OFF"}')
    print(f'Training: {epochs} epochs, {len(loader)} batches/epoch')

    total_steps = epochs * len(loader)
    warmup_steps = tcfg.get('warmup_epochs', 10) * len(loader)
    grad_clip = tcfg.get('grad_clip', 1.0)
    var_w = tcfg.get('variance_weight', 1.0) if use_vicreg else 0.0
    cov_w = tcfg.get('covariance_weight', 0.04) if use_vicreg else 0.0

    run_dir = Path(args.out_dir) / (
        'nanojepa' if use_vicreg else 'nanojepa_no_vicreg'
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    step = start_epoch * len(loader)
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        running = 0.0
        for batch in loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            lr = warmup_cosine(
                step, total_steps, warmup_steps, tcfg['lr'], tcfg['lr_min']
            )
            ema = linear_schedule(
                step,
                total_steps,
                tcfg['ema_momentum_start'],
                tcfg['ema_momentum_end'],
            )
            wd = linear_schedule(
                step,
                total_steps,
                tcfg['weight_decay_start'],
                tcfg['weight_decay_end'],
            )
            for pg in optimizer.param_groups:
                pg['lr'], pg['weight_decay'] = lr, wd

            out = model(batch)
            pred, target = out['predicted'], out['target']
            loss_j = jepa_loss(pred, target)
            loss_v = (
                variance_loss(pred)
                if use_vicreg
                else torch.zeros((), device=device)
            )
            loss_c = (
                covariance_loss(pred)
                if use_vicreg
                else torch.zeros((), device=device)
            )
            loss = loss_j + var_w * loss_v + cov_w * loss_c

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    grad_clip,
                )
            optimizer.step()
            model.update_target_encoder(ema)

            collapse = compute_collapse_metrics(pred)
            running += loss.item()
            step += 1
            history['loss'].append(loss.item())
            history['jepa_loss'].append(loss_j.item())
            history['var_loss'].append(loss_v.item())
            history['cov_loss'].append(loss_c.item())
            history['lr'].append(lr)
            history['ema_momentum'].append(ema)
            history['weight_decay'].append(wd)
            history['embedding_std'].append(collapse['embedding_std_mean'])
            history['effective_rank'].append(collapse['effective_rank'])

            if step % tcfg.get('log_every', 50) == 0:
                print(
                    f'  step {step:5d} | loss {loss.item():.4f} '
                    f'(jepa {loss_j.item():.4f}, var {loss_v.item():.4f}, '
                    f'cov {loss_c.item():.4f}) | '
                    f'std {collapse["embedding_std_mean"]:.3f} | '
                    f'rank {collapse["effective_rank"]:.1f} | lr {lr:.2e}'
                )
            if (
                device.type == 'mps'
                and step % tcfg.get('mps_cache_every', 50) == 0
            ):
                torch.mps.empty_cache()

        avg = running / max(1, len(loader))
        print(
            f'Epoch {epoch + 1}/{epochs} | avg_loss {avg:.4f} | '
            f'{time.time() - t0:.0f}s'
        )
        if (epoch + 1) % tcfg.get('save_every', 10) == 0 or (
            epoch + 1
        ) == epochs:
            path = run_dir / f'epoch_{epoch + 1}.pt'
            model.save_checkpoint(
                str(path),
                epoch=epoch + 1,
                optimizer=optimizer,
                history=history,
                vicreg=use_vicreg,
            )
            print(f'  -> saved {path}')

    print(f'Done: {step} steps in {time.time() - t0:.0f}s')
    return history


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--data', default=None, help='lance dataset name/path')
    p.add_argument('--no-vicreg', action='store_true', help='collapse demo')
    p.add_argument('--resume', default=None, help='checkpoint to resume')
    p.add_argument('--device', default=None)
    p.add_argument('--epochs', type=int, default=None, help='override epochs')
    p.add_argument('--out-dir', default='checkpoints')
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, args)


if __name__ == '__main__':
    main()
