"""4D Gaussian Splatting compression for the comma video compression challenge.

Structured decomposition into four physically-motivated populations:
  RoadPlane   - 2D Gaussians on the ground plane (z=0)
  SkyDome     - 2D Gaussians on a large sphere (infinity)
  RoadsideBand- 3D Gaussians in lateral bands along the driven path
  DynamicActor- Cluster of Gaussians with polynomial trajectory

+ EgoPoseINR (t -> SE(3)) + ViewDependentShader.
Rendered via gsplat; trained against a score-shaped loss.
"""
from .populations import (DynamicActor, RoadPlane, RoadsideBand,
                          SceneRepresentation, SkyDome)
from .pose_inr import EgoPoseINR
from .shader import ViewDependentShader

__all__ = [
    "RoadPlane", "SkyDome", "RoadsideBand", "DynamicActor",
    "SceneRepresentation", "EgoPoseINR", "ViewDependentShader",
]
