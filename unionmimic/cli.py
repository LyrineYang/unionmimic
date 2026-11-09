import argparse
from pathlib import Path
from .common import ROOT, RELEASE


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--output', type=Path, required=True)
    c.add_argument('--envs', type=int, default=16)
    c.add_argument('--steps', type=int, default=1600)
    c.add_argument('--gpu', type=int, default=0)
    c.add_argument('--seed', type=int, default=42)
    c.add_argument('--probe-noise', type=float, default=1.0)
    c.add_argument('--probe-mode', choices=['noise','release'], default='noise')
    c.add_argument('--stress-every', type=int, default=4)
    c.add_argument('--gravity-max-degrees', type=float, default=None)
    c.add_argument('--checkpoint', type=Path, default=RELEASE)
    c = sub.add_parser('sm')
    c.add_argument('--cache', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    c.add_argument('--checkpoint', type=Path, required=True)
    c.add_argument('--gpu', type=int, default=1)
    c.add_argument('--seed', type=int, default=47)
    c.add_argument('--updates', type=int, default=500)
    c.add_argument('--batch', type=int, default=128)
    c.add_argument('--lr', type=float, default=1e-5)
    c.add_argument('--gamma', type=float, default=.99)
    c.add_argument('--anchor-coef', type=float, default=1.)
    c.add_argument('--save-every', type=int, default=100)
    c.add_argument('--resume', action='store_true')
    c.add_argument('--fork-seed', action='store_true')
    c = sub.add_parser('evaluate')
    c.add_argument('--checkpoint', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    c.add_argument('--envs', type=int, default=8)
    c.add_argument('--gpu', type=int, default=0)
    c.add_argument('--seed', type=int, default=123)
    c = sub.add_parser('night')
    c.add_argument('--run', type=Path, required=True)
    c.add_argument('--data', type=Path, required=True)
    c.add_argument('--cache', type=Path, required=True)
    c.add_argument('--checkpoint', type=Path, required=True)
    c.add_argument('--baseline', type=Path, required=True)
    c.add_argument('--train-gpu', type=int, default=1)
    c.add_argument('--sim-gpu', type=int, default=0)
    c.add_argument('--cycles', type=int, default=24)
    c.add_argument('--hours', type=float, default=6)
    c.add_argument('--sm-updates', type=int, default=500)
    c.add_argument('--calibration-updates', type=int, default=1000)
    c.add_argument('--collect-steps', type=int, default=600)
    c.add_argument('--lr', type=float, default=1e-5)
    c.add_argument('--seed', type=int, default=1000)
    c.add_argument('--anchor-coef', type=float, default=1.)
    c.add_argument('--keep-cycles', type=int, default=3)
    c.add_argument('--fork-seed', action='store_true')
    c.add_argument('--regression-tolerance', type=float, default=.1)
    c.add_argument('--resume-run', action='store_true')
    c = sub.add_parser('calibrate')
    c.add_argument('--cache', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    c.add_argument('--gpu', type=int, default=1)
    c.add_argument('--seed', type=int, default=44)
    c.add_argument('--updates', type=int, default=2500)
    c.add_argument('--batch', type=int, default=256)
    c.add_argument('--lr', type=float, default=2e-4)
    c.add_argument('--eval-every', type=int, default=250)
    c.add_argument('--checkpoint', type=Path, default=RELEASE)
    c.add_argument('--resume', action='store_true')
    c = sub.add_parser('merge')
    c.add_argument('--data', type=Path, nargs='+', required=True)
    c.add_argument('--output', type=Path, required=True)
    c.add_argument('--seed', type=int, default=45)
    c = sub.add_parser('validate')
    c.add_argument('--cache', type=Path, required=True)
    c.add_argument('--checkpoint', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    c.add_argument('--gpu', type=int, default=1)
    c = sub.add_parser('prepare')
    c.add_argument('--data', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    c.add_argument('--gpu', type=int, default=1)
    c.add_argument('--seed', type=int, default=43)
    c.add_argument('--horizon', type=int, default=4)
    c.add_argument('--warmup', type=int, default=8)
    c.add_argument('--max-starts', type=int, default=20000)
    c.add_argument('--recompute-state', action='store_true',
                   help='Rebuild current-model posterior states; never reuse recorded samples after RSSM updates')
    c.add_argument('--checkpoint', type=Path, default=RELEASE)
    args = p.parse_args()
    if args.command == 'collect':
        from .collect import collect
        from .simulation_worker import run_isolated_simulation
        run_isolated_simulation(collect,args)
    elif args.command == 'prepare':
        from .prepare import prepare
        prepare(args)
    elif args.command == 'calibrate':
        from .calibrate import calibrate
        calibrate(args)
    elif args.command == 'validate':
        from .calibrate import validate_checkpoint
        validate_checkpoint(args)
    elif args.command == 'merge':
        from .merge import merge
        merge(args)
    elif args.command == 'sm':
        from .sm import train_sm
        train_sm(args)
    elif args.command == 'evaluate':
        from .evaluate import evaluate_policy
        from .simulation_worker import run_isolated_simulation
        run_isolated_simulation(evaluate_policy,args)
    elif args.command == 'night':
        from .night import run_night
        run_night(args)


if __name__ == '__main__':
    main()
