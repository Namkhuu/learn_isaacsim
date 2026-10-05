import argparse
import time
from isaaclab.app import AppLauncher
from models import Policy, Value

parser = argparse.ArgumentParser(description="Play trained robot policy")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint, or leave blank for latest")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--real-time", action="store_true", default=True, help="Pace playback to wall-clock time.")
parser.add_argument("--no-real-time", dest="real_time", action="store_false", help="Run as fast as the sim allows.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.memories.torch import RandomMemory

import os
import glob

def get_latest_checkpoint(runs_dir="runs"):
    checkpoints = glob.glob(os.path.join(runs_dir, "*", "checkpoints", "best_agent.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {runs_dir}")
    latest = max(checkpoints, key=os.path.getmtime)
    print(f"Loading checkpoint: {latest}")
    return latest

from my_env import MyRobotEnv, MyRobotEnvCfg


def main():
    env_cfg = MyRobotEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = MyRobotEnv(cfg=env_cfg)
    env = wrap_env(env, wrapper="isaaclab")

    device = env.device

    models = {
        "policy": Policy(env.observation_space, env.action_space, device),
        "value":  Value(env.observation_space, env.action_space, device),
    }

    cfg = PPO_CFG()
    cfg.experiment.write_interval = 0
    cfg.experiment.checkpoint_interval = 0
    cfg.experiment.directory = ""

    agent = PPO(
        models=models,
        memory=RandomMemory(memory_size=16, num_envs=env.num_envs, device=device),
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    checkpoint = args_cli.checkpoint if args_cli.checkpoint else get_latest_checkpoint()
    agent.load(checkpoint)
    agent.set_running_mode("eval")

    dt = env.unwrapped.step_dt  # decimation * physics_dt, typically ~0.02 s

    obs, _ = env.reset()
    with torch.no_grad():
        while simulation_app.is_running():
            start_time = time.time()
            actions, _ = agent.act(obs, env.state(), timestep=0, timesteps=0)
            obs, reward, terminated, truncated, info = env.step(actions)
            if args_cli.real_time:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

