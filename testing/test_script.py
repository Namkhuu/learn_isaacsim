import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ============================================
# CONFIGURATION - Edit these values
# ============================================

# Path to your trained policy checkpoint (from Isaac Lab training)
CHECKPOINT_PATH = r"C:\Users\<user>\Documents\ROSE\Projects\project_1\runs\26-01-01_00-00-00-000000_PPO\checkpoints\best_agent.pt"

# Number of actuators/joints (from your hardware config: ${numActuators})
NUM_ACTUATORS = 6 # <--- Change to number of joints

# Observation and action dimensions (adjust based on your environment)
# Check your env_cfg.py for observation_space and action_space sizes
OBS_DIM = NUM_ACTUATORS * 2  # e.g., joint_pos + joint_vel
ACTION_DIM = NUM_ACTUATORS

# Initial joint positions (in radians or your policy's units)
INITIAL_JOINT_POSITIONS = [0,0,0,0,0,0] # <--- n values for n joints

# Initial joint velocities
INITIAL_JOINT_VELOCITIES = [0,0,0,0,0,0] # <--- n values for n joints

# Default joint positions from the environment used during training
DEFAULT_JOINT_POSITIONS = [0,0,0,0,0,0] # <--- n values for n joints

# Match env-style processing: target = default_joint_pos + action_scale * action
ACTION_SCALE = 0.5

# Number of timesteps to simulate
NUM_TIMESTEPS = 10

# Action limits for safety checking${hardwareParams.positionLimits ? ` (from config: ${hardwareParams.positionLimits})` : ''}
POSITION_LIMITS = [(-3.14, 3.14)] * NUM_ACTUATORS  # (min, max) for each joint
VELOCITY_LIMITS = [(-10.0, 10.0)] * NUM_ACTUATORS
TORQUE_LIMITS = [(-25.0, 25.0)] * NUM_ACTUATORS

# ============================================
# SKRL POLICY NETWORK DEFINITION
# ============================================
# This must match the architecture used during training.
# Check your skrl_ppo_cfg.yaml for hidden_units and activation.

class GaussianPolicy(nn.Module):
    """skrl-style Gaussian policy network for PPO."""

    def __init__(self, obs_dim, action_dim, hidden_units=[256, 128], activation=nn.ELU):
        super().__init__()

        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.mean = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        """Forward pass - returns mean action."""
        features = self.features(obs)
        return self.mean(features)

    def act(self, obs, deterministic=True):
        """Get action from observation."""
        mean = self.forward(obs)
        if deterministic:
            return mean
        else:
            std = torch.exp(self.log_std)
            return torch.normal(mean, std)

# ============================================
# POLICY LOADING
# ============================================

def load_skrl_policy(checkpoint_path, obs_dim, action_dim):
    """Load a trained skrl PPO policy from checkpoint."""
    try:
        checkpoint_path = str(Path(checkpoint_path).expanduser())
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # skrl saves checkpoints with 'policy' key
        if 'policy' in checkpoint:
            state_dict = checkpoint['policy']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # Might be raw state dict
            state_dict = checkpoint

        # Create policy network with matching architecture
        # Adjust hidden_units to match your training config
        policy = GaussianPolicy(obs_dim, action_dim, hidden_units=[256, 128])

        # Try to load state dict (may need key remapping for skrl format)
        try:
            policy.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Direct load failed, trying key remapping: {e}")
            # skrl may use different key names
            remapped = {}
            for k, v in state_dict.items():
                new_key = k.replace('net.', 'features.').replace('mean_layer.', 'mean.')
                remapped[new_key] = v
            policy.load_state_dict(remapped, strict=False)

        policy.eval()
        return policy

    except Exception as e:
        print(f"Error loading policy: {e}")
        print("\\nTroubleshooting:")
        print("1. Check CHECKPOINT_PATH points to a valid .pt file")
        print("2. Verify OBS_DIM and ACTION_DIM match your environment")
        print("3. Ensure hidden_units match your skrl_ppo_cfg.yaml")
        return None

# ============================================
# HELPER FUNCTIONS
# ============================================

def build_observation(joint_pos, joint_vel):
    """Build observation tensor for the policy."""
    obs = np.concatenate([
        np.array(joint_pos),
        np.array(joint_vel)
    ])
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

def check_limits(action, limits, name):
    """Check if action is within limits."""
    warnings = []
    for i, (val, (min_val, max_val)) in enumerate(zip(action, limits)):
        if val < min_val or val > max_val:
            warnings.append(f"  Joint {i}: {val:.4f} (limit: [{min_val}, {max_val}])")
    return len(warnings) == 0


def limit_margin(values, limits):
    """Return the smallest margin to a limit (negative means violation)."""
    min_margin = float("inf")
    for val, (min_val, max_val) in zip(values, limits):
        margin = min(val - min_val, max_val - val)
        min_margin = min(min_margin, margin)
    return min_margin

# ============================================
# MAIN TEST
# ============================================

def run_test():
    """Run the policy test."""
    print("=" * 50)
    print("SKRL PPO POLICY TEST")
    print("=" * 50)

    # Load policy
    print(f"\nLoading policy from: {CHECKPOINT_PATH}")
    print(f"Observation dim: {OBS_DIM}, Action dim: {ACTION_DIM}")

    policy = load_skrl_policy(CHECKPOINT_PATH, OBS_DIM, ACTION_DIM)
    if policy is None:
        return

    print("Policy loaded successfully!")

    print(f"\nInitial Conditions:")
    print(f"  Joint Positions:  {INITIAL_JOINT_POSITIONS}")
    print(f"  Joint Velocities: {INITIAL_JOINT_VELOCITIES}")

    print(f"\n{'='*50}")
    print(f"Running {NUM_TIMESTEPS} timesteps...")
    print(f"{'='*50}\n")
    print("Final commanded joint positions per timestep:")

    # Current state
    joint_pos = list(INITIAL_JOINT_POSITIONS)
    joint_vel = list(INITIAL_JOINT_VELOCITIES)
    default_joint_pos = np.array(DEFAULT_JOINT_POSITIONS, dtype=np.float32)

    all_safe = True

    for t in range(NUM_TIMESTEPS):
        # Build observation
        obs = build_observation(joint_pos, joint_vel)

        # Run policy inference
        with torch.no_grad():
            action = policy.act(obs, deterministic=True)

        action = action.squeeze().numpy().astype(np.float32)
        target_pos = default_joint_pos + ACTION_SCALE * action

        # Safety checks on commanded target positions
        safe = check_limits(target_pos, POSITION_LIMITS, "Target Position")
        all_safe = all_safe and safe
        target_str = ", ".join(f"{float(p):.4f}" for p in target_pos)
        print(f"t={t+1:02d}: [{target_str}]")

        # Simple state update (target command as next position approximation)
        joint_pos = target_pos.tolist()

    print("=" * 50)
    if all_safe:
        print("SUCCESS: All commanded joint positions are within limits")
    else:
        print("WARNING: Some actions exceeded limits - check before deploying!")
    print("=" * 50)

if __name__ == "__main__":
    run_test()