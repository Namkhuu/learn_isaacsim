import inspect
from isaaclab import cloner

print("grid_transforms:")
print(inspect.signature(cloner.grid_transforms))

print("\nclone_plan_from_env_0:")
print(inspect.signature(cloner.clone_plan_from_env_0))

print("\nreplicate:")
print(inspect.signature(cloner.replicate))