
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
MAX_OMEGA = 6.0
MAX_VEL   = 1.8    # m/s
STOP_DIST = 0.15   # stop when this close to ball (m)

APPROACH_DIST =  0.25     # how far behind ball to position (meters)

CIRCLE_RADIUS = 0.4   # tune based on robot speed/acceleration

KICK_SPEED = 5.0  # m/s

def compute_shoot_target(rx, ry, bx, by, goal_pos, offset=APPROACH_DIST, r=CIRCLE_RADIUS):
    gx, gy = goal_pos

    # Compute true shooting direction: ball → goal
    dx_g = gx - bx
    dy_g = gy - by
    dist_g = math.hypot(dx_g, dy_g)
    if dist_g < 1e-6:
        return bx, by  # degenerate case: ball is on top of goal

    su = dx_g / dist_g   # was hardcoded 1.0
    sv = dy_g / dist_g   # was hardcoded 0.0

    # Target position: behind ball, opposite to shoot direction
    target_x = bx - su * offset
    target_y = by - sv * offset

    # Robot offset from ball
    dx = rx - bx
    dy = ry - by

    # Project onto shoot axis and perpendicular
    along = dx * su + dy * sv        # positive = behind ball (correct side)
    perp  = dx * (-sv) + dy * su    # side offset

    # Check if robot is already in the approach cone (behind ball, close to axis)
    if along < 0 and abs(perp) < r:
        # Robot is on correct side and close to shoot line — go directly
        return target_x, target_y

    # Two circle centers: offset perpendicular from ball on each side
    c1x = bx + (-sv) * r
    c1y = by + su * r
    c2x = bx - (-sv) * r
    c2y = by - su * r

    # Pick the circle on the same side as robot
    if perp >= 0:
        cx, cy = c1x, c1y
    else:
        cx, cy = c2x, c2y

    # Navigate to tangent point: point on circle closest to target
    # direction from circle center to target
    ctx = target_x - cx
    cty = target_y - cy
    ctd = math.hypot(ctx, cty)
    if ctd < 1e-6:
        return target_x, target_y

    # Arc waypoint: circle center + r * unit(center→target)
    arc_x = cx + (ctx / ctd) * r
    arc_y = cy + (cty / ctd) * r
    return arc_x, arc_y

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

        self._ball      = None
        self._robot     = None
        self._yaw       = 0.0
        self._obstacles = []
        
        self._state = 'APPROACH'  # states: APPROACH, ALIGN, KICK

        self._prev_ball      = None
        self._prev_ball_time = None
        self._ball_vel       = (0.0, 0.0)

        self.create_timer(0.02, self._control_loop)
        self.get_logger().info(
            f'Phase 3 APF ready — team={TEAM} robot_id={ROBOT_ID}'
        )

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
            best = max(ball_candidates, key=lambda x: x[0])
            new_ball = (best[1], best[2])
            now = self.get_clock().now().nanoseconds * 1e-9

            if self._prev_ball is not None and self._prev_ball_time is not None:
                dt = now - self._prev_ball_time
                if dt > 0.005:                          # ignore duplicate frames
                    alpha = 0.4                         # EMA smoothing factor
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
            self._robot = (x, y)
            self._yaw   = yaw

        # only update obstacles when opponents are actually detected
        new_obs = [(v[1], v[2]) for v in opp_robots.values()]
        if new_obs:
            self._obstacles = new_obs

    # Control loop
    def _control_loop(self):
        if self._ball is None or self._robot is None:
            self.get_logger().info(
                'Waiting for vision...', throttle_duration_sec=2.0)
            return

        bx_raw, by_raw = self._ball
        vbx, vby = self._ball_vel
        rx, ry = self._robot

        dist_now = math.hypot(bx_raw - rx, by_raw - ry)
        t_intercept = min(dist_now / max(MAX_VEL, 0.1), 0.6)
        bx = bx_raw + vbx * t_intercept
        by = by_raw + vby * t_intercept

        GOAL_POS = (6.0, 0.0)
        gx, gy = GOAL_POS

        dist_to_real_ball = math.hypot(bx_raw - rx, by_raw - ry)
        theta_shoot = math.atan2(gy - by_raw, gx - bx_raw)
        ang_err = normalize_angle(theta_shoot - self._yaw)

        # Compute behind_score: positive means robot is behind ball relative to goal
        shoot_ux = (gx - bx_raw) / max(math.hypot(gx - bx_raw, gy - by_raw), 1e-6)
        shoot_uy = (gy - by_raw) / max(math.hypot(gx - bx_raw, gy - by_raw), 1e-6)
        behind_score = (rx - bx_raw) * (-shoot_ux) + (ry - by_raw) * (-shoot_uy)

        # State transitions
        if self._state == 'APPROACH':
            if dist_to_real_ball < 0.30 and behind_score > 0.10:
                self._state = 'ALIGN'

        elif self._state == 'ALIGN':
            if dist_to_real_ball > 0.40 or behind_score < -0.05:
                self._state = 'APPROACH'  # lost position, re-approach
            elif abs(ang_err) < 0.12:
                self._state = 'KICK'

        elif self._state == 'KICK':
            if dist_to_real_ball > 0.40:
                self._state = 'APPROACH'  # ball moved away, re-approach

        # State actions
        if self._state == 'APPROACH':
            tx, ty = compute_shoot_target(rx, ry, bx, by, goal_pos=GOAL_POS)
            vx_w, vy_w = compute_apf(rx, ry, tx, ty, self._obstacles, max_vel=MAX_VEL)
            omega = clamp(KP_ANG * ang_err, MAX_OMEGA)
            vt, vn = world_to_robot(vx_w, vy_w, self._yaw)
            self._send(vt, vn, omega)

        elif self._state == 'ALIGN':
            # Rotate in place to face goal, don't move into ball
            self._send(0.0, 0.0, clamp(KP_ANG * ang_err, MAX_OMEGA))

        elif self._state == 'KICK':
            vx_kick = (bx_raw - rx) / max(dist_to_real_ball, 1e-6) * MAX_VEL
            vy_kick = (by_raw - ry) / max(dist_to_real_ball, 1e-6) * MAX_VEL
            vt_kick, vn_kick = world_to_robot(vx_kick, vy_kick, self._yaw)
            self._send(vt_kick, vn_kick, 0.0, kickspeedx=KICK_SPEED)

        self.get_logger().info(
            f'state={self._state} ball=({bx_raw:.2f},{by_raw:.2f}) '
            f'robot=({rx:.2f},{ry:.2f}) dist={dist_to_real_ball:.2f} '
            f'ang_err={ang_err:.3f} behind={behind_score:.2f}',
            throttle_duration_sec=0.5
        )


    def _send(self, vt, vn, omega, kickspeedx=0.0):
        pkt = build_packet(ROBOT_ID, vt, vn, omega,
                           self._is_yellow, kickspeedx, 0.0, False)
        self._sock.sendto(pkt, (GRSIM_HOST, GRSIM_PORT))

    def destroy_node(self):
        self._send(0.0, 0.0, 0.0)
        self._sock.close()
        super().destroy_node()