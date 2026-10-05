import argparse

# IsaacLab app launcher MUST come before all other imports
from isaaclab.app import AppLauncher
from models import Policy, Value

parser = argparse.ArgumentParser(description="Train robot with skrl PPO")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint, or use 'latest' to resume from best_agent.pt")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--timesteps", type=int, default=10000, help="Number of training steps")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app       

import torch
from skrl.envs.wrappers.torch import wrap_env
import os
import glob


import skrl
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer

from my_env import MyRobotEnv, MyRobotEnvCfg

# ── Main ──────────────────────────────────────────────────────────────────────

def get_latest_checkpoint(runs_dir="runs"):
    checkpoints = glob.glob(os.path.join(runs_dir, "*", "checkpoints", "best_agent.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {runs_dir}")
    latest = max(checkpoints, key=os.path.getmtime)
    print(f"Loading checkpoint: {latest}")
    return latest

def main():
    # Create environment
    env_cfg = MyRobotEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = MyRobotEnv(cfg=env_cfg)

    # Wrap for skrl
    env = wrap_env(env, wrapper="isaaclab")

    device = env.device

    # Memory
    memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)

    # Models
    models = {
        "policy": Policy(env.observation_space, env.action_space, device),
        "value":  Value(env.observation_space, env.action_space, device),
    }

    # PPO config
    cfg = PPO_CFG()
    cfg.rollouts = 16
    cfg.learning_epochs = 8
    cfg.mini_batches = 1
    cfg.discount_factor = 0.99
    cfg.gae_lambda = 0.95
    cfg.learning_rate = 3e-4
    cfg.grad_norm_clip = 1.0
    cfg.ratio_clip = 0.2
    cfg.value_clip = 0.2
    cfg.entropy_loss_scale = 0.0
    cfg.value_loss_scale = 2.0
    cfg.experiment.write_interval = 100
    cfg.experiment.checkpoint_interval = 1000

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # Trainer
    trainer_cfg = {"timesteps": args_cli.timesteps}
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    if args_cli.checkpoint:
        checkpoint = get_latest_checkpoint() if args_cli.checkpoint == "latest" else args_cli.checkpoint
        agent.load(checkpoint)
        print(f"Resuming training from checkpoint: {checkpoint}")

    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

