import math


K_ATT     = 4.0
D_STAR    = 1.0
K_REP     = 6.0
RHO_0     = 0.25
N_NEAREST = 7
K_TAN = 3.0 

def _attractive(rx, ry, tx, ty):
    dx, dy  = tx - rx, ty - ry
    dist_g  = math.hypot(dx, dy)
    if dist_g < 1e-6:
        return 0.0, 0.0
    elif dist_g <= D_STAR:
        return K_ATT * dx, K_ATT * dy
    else:
        return K_ATT * D_STAR * dx / dist_g, K_ATT * D_STAR * dy / dist_g


def _repulsive(rx, ry, tx, ty, obstacles):
    sorted_obs = sorted(obstacles, key=lambda o: math.hypot(rx-o[0], ry-o[1]))[:N_NEAREST]
    frx, fry = 0.0, 0.0

    to_goal_x = tx - rx
    to_goal_y = ty - ry
    to_goal_d = math.hypot(to_goal_x, to_goal_y)
    goal_ux   = to_goal_x / to_goal_d if to_goal_d > 1e-6 else 0.0
    goal_uy   = to_goal_y / to_goal_d if to_goal_d > 1e-6 else 0.0

    for ox, oy in sorted_obs:
        rr  = math.hypot(rx - ox, ry - oy)
        rho = max(rr - 0.18, 0.01)
        if rho >= RHO_0:
            continue

        mag = K_REP * (1.0/rho - 1.0/RHO_0) / (rho**2)
        mag = min(mag, 20.0)

        rep_ux = (rx - ox) / rr if rr > 1e-6 else 1.0
        rep_uy = (ry - oy) / rr if rr > 1e-6 else 0.0

        frx += mag * rep_ux
        fry += mag * rep_uy

        dot = (-rep_ux) * goal_ux + (-rep_uy) * goal_uy
        if dot > 0.0:
            tan_ux =  -rep_uy
            tan_uy =   rep_ux
            tan_mag = K_TAN * mag * dot
            frx += tan_mag * tan_ux
            fry += tan_mag * tan_uy

    return frx, fry


def compute_apf(rx, ry, tx, ty, obstacles, max_vel=1.8):
    fax, fay = _attractive(rx, ry, tx, ty)
    frx, fry = _repulsive(rx, ry, tx, ty, obstacles)

    fx = fax + frx
    fy = fay + fry

    speed = math.hypot(fx, fy)
    if speed > max_vel:
        fx = fx * max_vel / speed
        fy = fy * max_vel / speed

    return fx, fy