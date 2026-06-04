"""Collect a FetchPush transition dataset for NanoJEPA.

A "transition" dataset for self-supervised JEPA training: just diverse random
experience of how the world changes in response to actions (no expert demos
needed). This delegates all the env wrapping / rendering / writing to
``stable_worldmodel`` — the educational content lives in the model, not here.

    python scripts/data/collect_nanojepa.py --out nanojepa_fetchpush_quick.lance \
        --episodes 500 --num-envs 8

The output lance dataset exposes ``pixels`` (64x64x3), ``proprio`` (25-d) and
``action`` (4-d) columns, consumed by ``swm.data.load_dataset(num_steps=2)``.
"""

import argparse
from pathlib import Path

import numpy as np

import stable_worldmodel as swm
from stable_worldmodel.policy import RandomPolicy


def resolve_data_path(name: str) -> str:
    """Resolve a dataset name to an absolute path under the swm datasets cache.

    ``World.collect`` writes to the literal path while ``load_dataset`` resolves
    a bare name against ``$STABLEWM_HOME/datasets``; resolving both to the same
    absolute path keeps collection and loading consistent regardless of cwd.
    """
    p = Path(name)
    if p.is_absolute() or p.exists():
        return str(p)
    cache = swm.data.utils.get_cache_dir(sub_folder='datasets')
    return str(Path(cache) / name)


def collect(
    out: str,
    episodes: int,
    num_envs: int = 8,
    seed: int = 0,
    image_size: int = 64,
    max_steps: int = 50,
    env_name: str = 'swm/FetchPush-v3',
) -> str:
    """Roll out a random policy and stream transitions to a lance dataset."""
    out = resolve_data_path(out)
    world = swm.World(
        env_name,
        num_envs=num_envs,
        image_shape=(image_size, image_size),
        max_episode_steps=max_steps,
    )
    world.set_policy(RandomPolicy(seed=seed))
    print(f'Collecting {episodes} episodes of {env_name} -> {out}')
    world.collect(path=out, episodes=episodes, seed=seed, format='lance')
    world.close()
    _verify(out)
    return out


def _verify(out: str) -> None:
    """Sanity-check the C0/C9 column contract on a 2-step window."""
    ds = swm.data.load_dataset(
        out, num_steps=2, keys_to_load=['pixels', 'proprio', 'action']
    )
    sample = ds[0]
    pix, pro, act = sample['pixels'], sample['proprio'], sample['action']
    print(f'  pixels  {tuple(pix.shape)} {pix.dtype}')
    print(f'  proprio {tuple(pro.shape)} {pro.dtype}')
    print(f'  action  {tuple(act.shape)} {act.dtype}')
    assert pix.shape[0] == 2 and pix.shape[1] == 3, 'pixels must be [2,3,H,W]'
    assert pro.shape[0] == 2 and pro.shape[1] >= 25, 'proprio must be [2,>=25]'
    assert act.shape[0] == 2 and act.shape[1] == 4, 'action must be [2,4]'
    print(f'  OK — {len(ds)} transition windows available')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='nanojepa_fetchpush_quick.lance')
    p.add_argument('--episodes', type=int, default=500)
    p.add_argument('--num-envs', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--image-size', type=int, default=64)
    p.add_argument('--max-steps', type=int, default=50)
    p.add_argument('--env-name', default='swm/FetchPush-v3')
    args = p.parse_args()

    np.random.seed(args.seed)
    collect(
        out=args.out,
        episodes=args.episodes,
        num_envs=args.num_envs,
        seed=args.seed,
        image_size=args.image_size,
        max_steps=args.max_steps,
        env_name=args.env_name,
    )


if __name__ == '__main__':
    main()
