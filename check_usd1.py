import argparse
from pathlib import Path
from pxr import Usd, UsdPhysics

parser = argparse.ArgumentParser(description="Inspect a robot USD's physics schemas")
parser.add_argument("usd_path", nargs="?", default=str(Path(__file__).resolve().parent / "robot_3.usd"))
usd_path = parser.parse_args().usd_path

print("Opening:", usd_path)

stage = Usd.Stage.Open(usd_path)

if stage is None:
    print("ERROR: Could not open USD file")
    raise SystemExit(1)

print("\n=== USD PRIMS ===")

found_articulation = False
found_rigid = False

for prim in stage.Traverse():
    has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
    has_articulation = prim.HasAPI(UsdPhysics.ArticulationRootAPI)

    if has_rigid:
        found_rigid = True

    if has_articulation:
        found_articulation = True

    print(
        f"{prim.GetPath()} "
        f"| RigidBody={has_rigid} "
        f"| ArticulationRoot={has_articulation}"
    )

print("\n=== RESULT ===")
print("Rigid bodies found:", found_rigid)
print("Articulation root found:", found_articulation)
if not (found_rigid and found_articulation):
    print("ERROR: Robot physics is missing. Check referenced USD files and payloads.")
    raise SystemExit(1)
