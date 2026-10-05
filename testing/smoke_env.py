"""Run with isaaclab.bat -p testing/smoke_env.py --visualizer none."""
import argparse
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from my_env import MyRobotEnv, MyRobotEnvCfg

env = None
try:
    cfg = MyRobotEnvCfg()
    cfg.scene.num_envs = 2
    cfg.seed = 42
    env = MyRobotEnv(cfg=cfg)
    assert env._robot.num_instances == cfg.scene.num_envs
    assert env._cube.num_instances == cfg.scene.num_envs
    observations, _ = env.reset()
    assert observations["policy"].shape == (2, cfg.observation_space)
    for _ in range(5):
        observations, rewards, _, _, _ = env.step(torch.zeros(2, cfg.action_space, device=env.device))
        assert torch.isfinite(observations["policy"]).all()
        assert torch.isfinite(rewards).all()
    print("SMOKE CHECK PASSED: two robots reset and stepped with finite observations and rewards.", flush=True)
except Exception:
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
