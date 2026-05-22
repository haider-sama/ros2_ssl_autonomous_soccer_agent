
import math
import socket

import rclpy
from rclpy.node import Node
from ssl_league_msgs.msg import VisionWrapper

from utils  import normalize_angle, clamp, world_to_robot
from packet import build_packet
from apf    import compute_apf


# Configuration

ROBOT_ID   = 0
TEAM       = 'blue'

GRSIM_HOST = '127.0.0.1'
GRSIM_PORT = 20011

KP_ANG    = 4.0
KD_ANG    = 0.05
MAX_OMEGA = 6.0
MAX_VEL   = 2.5   # m/s
STOP_DIST = 0.15  # stop when this close to ball (m)

GOAL_POS      = (6.0, 0.0)    # opponent goal center (12x9m field)
ALIGN_DIST    = 0.22          # orbit radius around ball (m); slightly > STOP_DIST
ANG_ALIGN_TOL = 0.10          # rad aligned when |ang_err_to_goal| < this

HOLD_TICKS    = 25            # control ticks at ball before transitioning (~0.5s)
ALIGN_TICKS   = 10            # ticks of stable heading before declaring READY


KICK_SPEED = 6.0              # m/s - grSim kick speed

# Helpers

def compute_shoot_axis(bx, by):
    """
    Returns shoot unit vector (shoot_ux, shoot_uy) from ball toward goal,
    and the behind-ball APF target (apf_tx, apf_ty).
    """
    gx, gy   = GOAL_POS
    shoot_dx = gx - bx
    shoot_dy = gy - by
    shoot_d  = math.hypot(shoot_dx, shoot_dy)
    shoot_ux = shoot_dx / max(shoot_d, 1e-6)
    shoot_uy = shoot_dy / max(shoot_d, 1e-6)
    # Behind-ball point: directly opposite goal on shoot axis
    apf_tx   = bx - shoot_ux * ALIGN_DIST
    apf_ty   = by - shoot_uy * ALIGN_DIST
    return shoot_ux, shoot_uy, apf_tx, apf_ty


def compute_robot_behind_ball(rx, ry, bx, by, shoot_ux, shoot_uy):
    """
    Compute behind_score: positive means robot is behind ball relative to goal.
    Returns True when robot is on the anti-goal side of the ball.
    """
    behind_score = (rx - bx) * (-shoot_ux) + (ry - by) * (-shoot_uy)
    return behind_score > 0.05


def compute_arc_waypoint(rx, ry, bx, by, apf_tx, apf_ty):
    """
    Tangential waypoint that arcs the robot around the ball without pushing it.
    Used during ALIGN reposition sub-phase.
    Robot-from-ball unit vector is rotated 90° in whichever direction
    reduces angle to the behind-ball target.
    """
    rbx = (rx - bx) / max(math.hypot(rx - bx, ry - by), 1e-6)
    rby = (ry - by) / max(math.hypot(rx - bx, ry - by), 1e-6)

    # Pick tangent direction that goes toward behind-ball point
    # (whichever 90° rotation reduces angle to apf_t)
    t1x, t1y = -rby,  rbx   # CCW tangent
    t2x, t2y =  rby, -rbx   # CW  tangent

    to_target_x = apf_tx - rx
    to_target_y = apf_ty - ry
    dot1 = t1x * to_target_x + t1y * to_target_y
    dot2 = t2x * to_target_x + t2y * to_target_y
    tan_x, tan_y = (t1x, t1y) if dot1 > dot2 else (t2x, t2y)

    # Waypoint: ball + ALIGN_DIST in tangent direction — skirts around ball
    arc_wx = bx + tan_x * ALIGN_DIST * 1.5
    arc_wy = by + tan_y * ALIGN_DIST * 1.5
    return arc_wx, arc_wy


# Control helpers

def apply_angular_pd(ang_err, prev_ang_err, dt):
    """
    PD controller for angular velocity.
    Returns (omega, updated_prev_ang_err).
    """
    d_ang_err = normalize_angle(ang_err - prev_ang_err) / dt
    omega     = clamp(KP_ANG * ang_err + KD_ANG * d_ang_err, MAX_OMEGA)
    return omega, ang_err

def apply_translational_pd(vx_w, vy_w, prev_dist_err, dist_err, dt):
    """
    Apply D term to APF output: scale back velocity when closing in fast.
    d_dist_err is negative when approaching → reduces speed → damps overshoot.
    Returns (vx_w, vy_w, pd_speed).
    """
    KD_VEL     = 0.15  # derivative gain — increase if robot oscillates near ball
    d_dist_err = (dist_err - prev_dist_err) / dt

    # APF gives us the world-frame velocity vector toward the ball (+ obstacle repulsion)
    # We treat the APF output magnitude as our proportional control signal.
    # The D term damps it using the rate of change of distance error.
    speed_apf = math.hypot(vx_w, vy_w)
    if speed_apf < 1e-6:
        return vx_w, vy_w, 0.0

    # Compute PD-adjusted speed: APF gives P term, D term subtracts rate of closure
    pd_speed = speed_apf + KD_VEL * d_dist_err    # d_dist_err < 0 when closing->reduces speed
    pd_speed = max(0.0, min(pd_speed, MAX_VEL))   # clamp, never reverse from D alone

    # Reapply direction from APF, magnitude from PD
    vx_w = (vx_w / speed_apf) * pd_speed
    vy_w = (vy_w / speed_apf) * pd_speed
    return vx_w, vy_w, pd_speed


# State handlers

def handle_approach_ball(node, bx, by, rx, ry, dist_err, prev_dist_err, nearby, dt):
    """
    Drive toward ball with APF + translational PD.
    Face the ball at all times with angular PD.
    Hold at ball for HOLD_TICKS before transitioning->ALIGN.
    """
    if dist_err < STOP_DIST:
        node._align_stable_count += 1
        if node._align_stable_count >= HOLD_TICKS:
            node._align_stable_count = 0
            node._state = 'ALIGN'
            node.get_logger().info('ALIGN')
        else:
            node._send(0.0, 0.0, 0.0)
            node.get_logger().info('AT BALL — holding', throttle_duration_sec=1.0)
        return

    # APF computes direction + proportional magnitude in world frame
    vx_w, vy_w = compute_apf(rx, ry, bx, by, nearby, max_vel=MAX_VEL)
    vx_w, vy_w, pd_speed = apply_translational_pd(
        vx_w, vy_w, prev_dist_err, dist_err, dt
    )
    vt, vn = world_to_robot(vx_w, vy_w, node._yaw)

    # Face the ball at all times during approach
    theta_ball = math.atan2(by - ry, bx - rx)
    ang_err    = normalize_angle(theta_ball - node._yaw)
    omega, node._prev_ang_err = apply_angular_pd(ang_err, node._prev_ang_err, dt)

    node._send(vt, vn, omega)
    node.get_logger().info(
        f'[APPROACH_BALL] ball=({bx:.2f},{by:.2f}) robot=({rx:.2f},{ry:.2f}) '
        f'dist={dist_err:.2f} pd_speed={pd_speed:.2f} ang_err={math.degrees(ang_err):.1f}° '
        f'obs={len(nearby)}',
        throttle_duration_sec=0.5
    )


def handle_align(node, bx, by, rx, ry, dist_err, nearby, dt):
    """
    Two sub-phases:
      reposition — arc around ball to get behind it (anti-goal side), face ball
      rotate     — face goal; declare READY once stable for ALIGN_TICKS

    Ball-moved guard: if dist > 2*STOP_DIST->back to APPROACH_BALL.
    """
    if dist_err > STOP_DIST * 2:
        node._state = 'APPROACH_BALL'
        node._align_stable_count = 0
        node.get_logger().info('Ball moved — back to APPROACH_BALL')
        return

    shoot_ux, shoot_uy, apf_tx, apf_ty = compute_shoot_axis(bx, by)
    robot_behind_check = compute_robot_behind_ball(rx, ry, bx, by, shoot_ux, shoot_uy)

    # ── Translational: arc around or hold behind ball ──────────────────────
    if not robot_behind_check:
        # Reposition: arc around ball using tangential waypoint
        arc_wx, arc_wy = compute_arc_waypoint(rx, ry, bx, by, apf_tx, apf_ty)
        vx_w, vy_w     = compute_apf(rx, ry, arc_wx, arc_wy, nearby, max_vel=MAX_VEL)
    else:
        # Behind ball — drive to hold position behind it
        vx_w, vy_w = compute_apf(rx, ry, apf_tx, apf_ty, nearby, max_vel=MAX_VEL)

    vt, vn = world_to_robot(vx_w, vy_w, node._yaw)

    # ── Angular ────────────────────────────────────────────────────────────
    if not robot_behind_check:
        # Sub-phase 1: reposition to behind-ball point, face the ball
        theta_ball = math.atan2(by - ry, bx - rx)
        ang_err    = normalize_angle(theta_ball - node._yaw)
        omega, node._prev_ang_err = apply_angular_pd(ang_err, node._prev_ang_err, dt)
        node._align_stable_count  = 0
        node._send(vt, vn, omega)
        node.get_logger().info(
            f'[ALIGN:reposition] ball=({bx:.2f},{by:.2f}) robot=({rx:.2f},{ry:.2f}) '
            f'dist={dist_err:.2f} ang_err={math.degrees(ang_err):.1f}°',
            throttle_duration_sec=0.5
        )
    else:
        # Sub-phase 2: behind ball confirmed — rotate to face goal
        gx, gy     = GOAL_POS
        theta_goal = math.atan2(gy - by, gx - bx)
        ang_err    = normalize_angle(theta_goal - node._yaw)
        omega, node._prev_ang_err = apply_angular_pd(ang_err, node._prev_ang_err, dt)

        if abs(ang_err) < ANG_ALIGN_TOL:
            node._align_stable_count += 1
            if node._align_stable_count >= ALIGN_TICKS:
                node._send(0.0, 0.0, 0.0)
                node._state = 'READY'
                node._align_stable_count = 0
                node.get_logger().info('ALIGNED — ready to kick')
                return
        else:
            node._align_stable_count = 0

        node._send(vt, vn, omega)
        node.get_logger().info(
            f'[ALIGN:rotate] ball=({bx:.2f},{by:.2f}) robot=({rx:.2f},{ry:.2f}) '
            f'dist={dist_err:.2f} ang_err={math.degrees(ang_err):.1f}° '
            f'{"ALIGNED" if abs(ang_err) < ANG_ALIGN_TOL else "rotating..."}',
            throttle_duration_sec=0.5
        )

def handle_ready(node, bx, by, rx, ry, dist_err):
    """
    Drive into ball with kicker armed.
    Once ball leaves (dist grows past 3 * STOP_DIST), reset->APPROACH_BALL.
    """
    if dist_err > STOP_DIST * 2:
        node._state = 'APPROACH_BALL'
        node._align_stable_count = 0
        node.get_logger().info('Ball moved — resetting to APPROACH_BALL')
        return

    # Drive into ball with kicker armed
    kick_ux  = (bx - rx) / max(dist_err, 1e-6)
    kick_uy  = (by - ry) / max(dist_err, 1e-6)
    vt_kick, vn_kick = world_to_robot(kick_ux * MAX_VEL, kick_uy * MAX_VEL, node._yaw)
    node._send(vt_kick, vn_kick, 0.0, kickspeedx=KICK_SPEED)
    node.get_logger().info('KICK_BALL — firing!', throttle_duration_sec=0.2)

    # Once ball leaves (dist grows), reset to approach
    if dist_err > STOP_DIST * 3:
        node._state = 'APPROACH_BALL'
        node._align_stable_count = 0
        node.get_logger().info('Ball kicked — resetting to APPROACH_BALL')


class BallTrackerNode(Node):

    def __init__(self):
        super().__init__('ball_tracker')

        self._sock      = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._is_yellow = (TEAM == 'yellow')

        self.create_subscription(
            VisionWrapper,
            '/ssl_vision_bridge/vision_messages',
            self._vision_cb,
            10
        )

        self._ball         = None
        self._robot        = None
        self._yaw          = 0.0
        self._obstacle_map = {}
        self._obstacles    = []

        self._state              = 'APPROACH_BALL'
        self._align_stable_count = 0

        self._prev_ball      = None
        self._prev_ball_time = None
        self._ball_vel       = (0.0, 0.0)

        # PD control — derivative terms need previous error
        self._prev_dist_err = 0.0   # previous distance error for translational D term
        self._prev_ang_err  = 0.0   # previous angular error for angular D term
        self._prev_time     = None  # timestamp of last control loop call

        self.create_timer(0.02, self._control_loop)
        self.get_logger().info(f'ready — team={TEAM} robot_id={ROBOT_ID}')


    # Vision callback
    def _vision_cb(self, msg: VisionWrapper):
        if not msg.detection:
            return

        ball_candidates = []
        our_robots      = {}
        opp_robots      = {}

        # if msg.geometry:
        #     geom = msg.geometry[0].field
        #     field_length = geom.field_length
        #     field_width  = geom.field_width
        #     goal_width   = geom.goal_width

        #     self._goal_x = field_length / 2.0
        #     self._goal_y = 0.0
        #     self.get_logger().info(
        #             f'Field: {field_length}x{field_width}m | Goal center: ({self._goal_x}, 0.0)'
        #     )

        for frame in msg.detection:
            for b in frame.balls:
                ball_candidates.append((b.confidence, b.pos.x, b.pos.y))

            our = frame.robots_blue if TEAM == 'blue' else frame.robots_yellow
            for r in our:
                prev = our_robots.get(r.robot_id, (-1,))
                if r.confidence > prev[0]:
                    q   = r.pose.orientation
                    yaw = math.atan2(
                        2.0*(q.w*q.z + q.x*q.y),
                        1.0 - 2.0*(q.y*q.y + q.z*q.z)
                    )
                    our_robots[r.robot_id] = (r.confidence,
                                              r.pose.position.x,
                                              r.pose.position.y, yaw)

            opp = frame.robots_yellow if TEAM == 'blue' else frame.robots_blue
            for r in opp:
                prev = opp_robots.get(r.robot_id, (-1,))
                if r.confidence > prev[0]:
                    opp_robots[r.robot_id] = (r.confidence,
                                              r.pose.position.x,
                                              r.pose.position.y)

        if ball_candidates:
            best     = max(ball_candidates, key=lambda x: x[0])
            new_ball = (best[1], best[2])
            now      = self.get_clock().now().nanoseconds * 1e-9

            if self._prev_ball is not None and self._prev_ball_time is not None:
                dt = now - self._prev_ball_time
                if dt > 0.005:                          # ignore duplicate frames
                    alpha  = 0.4                        # EMA smoothing factor
                    raw_vx = (new_ball[0] - self._prev_ball[0]) / dt
                    raw_vy = (new_ball[1] - self._prev_ball[1]) / dt
                    self._ball_vel = (
                        alpha * raw_vx + (1 - alpha) * self._ball_vel[0],
                        alpha * raw_vy + (1 - alpha) * self._ball_vel[1],
                    )

            self._prev_ball      = new_ball
            self._prev_ball_time = now
            self._ball           = new_ball

        if ROBOT_ID in our_robots:
            _, x, y, yaw = our_robots[ROBOT_ID]
            self._robot  = (x, y)
            self._yaw    = yaw

        now_t = self.get_clock().now().nanoseconds * 1e-9
        for rid, v in opp_robots.items():
            self._obstacle_map[rid] = (v[1], v[2], now_t)

        # Add teammates as obstacles too (excluding self)
        for rid, v in our_robots.items():
            if rid == ROBOT_ID:
                continue
            self._obstacle_map[rid] = (v[1], v[2], now_t)
            
        # Expire robots not seen for more than 1 second
        self._obstacle_map = {
            rid: val for rid, val in self._obstacle_map.items()
            if now_t - val[2] < 1.0
        }
        self._obstacles = [(val[0], val[1]) for val in self._obstacle_map.values()]


    # Control loop
    def _control_loop(self):
        if self._ball is None or self._robot is None:
            self.get_logger().info('Waiting for vision...', throttle_duration_sec=2.0)
            return

        bx, by = self._ball
        rx, ry = self._robot
        now    = self.get_clock().now().nanoseconds * 1e-9

        # dt for derivative terms — skip first frame
        if self._prev_time is None:
            self._prev_time = now
            return
        dt = now - self._prev_time
        if dt < 1e-6:
            return
        self._prev_time = now

        dist_err            = math.hypot(bx - rx, by - ry)      # how far from ball
        prev_dist_err_snap  = self._prev_dist_err               # updated below
        self._prev_dist_err = dist_err

        nearby = [o for o in self._obstacles
                  if math.hypot(rx - o[0], ry - o[1]) < 2.0]

        if self._state == 'APPROACH_BALL':
            handle_approach_ball(self, bx, by, rx, ry, dist_err, prev_dist_err_snap, nearby, dt)
        elif self._state == 'ALIGN':
            handle_align(self, bx, by, rx, ry, dist_err, nearby, dt)
        elif self._state == 'READY':
            handle_ready(self, bx, by, rx, ry, dist_err)


    # Send command
    def _send(self, vt, vn, omega, kickspeedx=0.0):
        pkt = build_packet(ROBOT_ID, vt, vn, omega,
                           self._is_yellow, kickspeedx, 0.0, False)
        self._sock.sendto(pkt, (GRSIM_HOST, GRSIM_PORT))

    def destroy_node(self):
        self._send(0.0, 0.0, 0.0)
        self._sock.close()
        super().destroy_node()