import math

def normalize_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a >  math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


def clamp(v: float, limit: float) -> float:
    return max(-limit, min(limit, v))


def world_to_robot(vx_w: float, vy_w: float, yaw: float):
    """Rotate world-frame velocity into robot-local frame."""
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    vt =  c * vx_w - s * vy_w   # forward  (veltangent)
    vn =  s * vx_w + c * vy_w   # lateral  (velnormal)
    return vt, vn