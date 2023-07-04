from dataclasses import dataclass
from cyclonedds.idl import IdlStruct


@dataclass
class orientation_dds(IdlStruct):
    yaw: float
    pitch: float
    roll: float

@dataclass
class acceleration_dds(IdlStruct):
    x: float
    y: float
    z: float

@dataclass
class position_dds(IdlStruct):
    x: float
    y: float
    z: float

@dataclass
class velocity_dds(IdlStruct):
    x: float
    y: float
    z: float

@dataclass
class gnss_dds(IdlStruct):
    position: position_dds
    velocity: velocity_dds
