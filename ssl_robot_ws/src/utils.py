import math

def normalize_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def clamp(v: float, limit: float) -> float:
    return max(-limit, min(limit, v))


def world_to_robot(vx_w: float, vy_w: float, yaw: float):
    """Rotate world-frame velocity into robot-local frame."""
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    vt =  c * vx_w - s * vy_w   # forward  (veltangent)
    vn =  s * vx_w + c * vy_w   # lateral  (velnormal)
    return vt, vn


