# from typing import TypedDict
from typing import Dict


class Vector3(Dict):
    x: float
    y: float
    z: float



class AgentPose(Dict):
    position: Vector3
    rotation: Vector3
    horizon: int
    standing: bool