import math

K_ATT     = 4.0
D_STAR    = 1.0
K_REP     = 2.0
RHO_0     = 0.4
N_NEAREST = 7

def compute_apf(rx, ry, tx, ty, obstacles, max_vel=1.8):

    dx, dy = tx - rx, ty - ry
    dist_g = math.hypot(dx, dy)
    if dist_g < 1e-6:
        fax, fay = 0.0, 0.0
    elif dist_g <= D_STAR:
        fax, fay = K_ATT * dx, K_ATT * dy
    else:
        fax = K_ATT * D_STAR * dx / dist_g
        fay = K_ATT * D_STAR * dy / dist_g

    sorted_obs = sorted(obstacles, key=lambda o: math.hypot(rx-o[0], ry-o[1]))[:N_NEAREST]
    frx, fry = 0.0, 0.0
    for ox, oy in sorted_obs:
        rr  = math.hypot(rx - ox, ry - oy)
        rho = max(rr - 0.18, 0.01)
        if rho >= RHO_0:
            continue
        mag = K_REP * (1.0/rho - 1.0/RHO_0) / (rho**2)
        mag = min(mag, 20.0)
        if rr > 1e-6:
            frx += mag * (rx - ox) / rr
            fry += mag * (ry - oy) / rr
        else:
            frx += mag

    fx = fax + frx
    fy = fay + fry

    speed = math.hypot(fx, fy)
    if speed > max_vel:
        fx = fx * max_vel / speed
        fy = fy * max_vel / speed

    return fx, fy