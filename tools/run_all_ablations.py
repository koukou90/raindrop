"""Run all A-class ablation experiments for three sites and update xlsx."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITES = ['W2127_Haichaoba', 'W2128_Haichaoyinsi', 'W2129_buligou']

A_MODELS = [
    'AblateV3NoPersistence',
    'AblateV3NoMixGate',
    'AblateV3NoBinAware',
    'AblateV3NoAux',
    'AblateV3NoStreamGate',
]


def run_cmd(cmd: list[str]) -> None:
    print('\n' + '=' * 70)
    print('RUN:', ' '.join(cmd))
    print('=' * 70)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    py = sys.executable
    for site in SITES:
        for model in A_MODELS:
            run_cmd([py, 'run.py', '--model_name', model, '--site_name', site])

    run_cmd([py, 'tools/update_ablation_metrics_xlsx.py', '--site_name', 'all'])


if __name__ == '__main__':
    main()
