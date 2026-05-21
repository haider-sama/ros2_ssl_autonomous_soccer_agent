
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
MAX_VEL   = 2.5    # m/s
STOP_DIST = 0.15  # stop when this close to ball (m)

GOAL_POS      = (6.0, 0.0)    # 12x9m field
ALIGN_DIST    = 0.22          # orbit radius around ball (m); slightly > STOP_DIST
ANG_ALIGN_TOL = 0.10          # rad aligned when |ang_err_to_goal| < this

KD_ANG = 0.05

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
        self._obstacle_map = {}
        
        self._state = 'APPROACH_BALL'
        self._align_stable_count = 0 

        self._prev_ball      = None
        self._prev_ball_time = None
        self._ball_vel       = (0.0, 0.0)

        # Add these for PD control (derivative terms need previous error):
        self._prev_dist_err  = 0.0   # previous distance error for translational D term
        self._prev_ang_err   = 0.0   # previous angular error for angular D term
        self._prev_time      = None  # timestamp of last control loop call

        self.create_timer(0.02, self._control_loop)
        self.get_logger().info(
            f'ready — team={TEAM} robot_id={ROBOT_ID}'
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

        now_t = self.get_clock().now().nanoseconds * 1e-9
        for rid, v in opp_robots.items():
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

        bx, by   = self._ball
        rx, ry   = self._robot
        now      = self.get_clock().now().nanoseconds * 1e-9

        # dt for derivative terms — skip first frame
        if self._prev_time is None:
            self._prev_time = now
            return
        dt = now - self._prev_time
        if dt < 1e-6:
            return
        self._prev_time = now

        # ── Translational PD ──────────────────────────────────────────────────
        # APF gives us the world-frame velocity vector toward the ball (+ obstacle repulsion)
        # We treat the APF output magnitude as our proportional control signal.
        # The D term damps it using the rate of change of distance error.

        dist_err         = math.hypot(bx - rx, by - ry)   # how far from ball
        d_dist_err       = (dist_err - self._prev_dist_err) / dt
        self._prev_dist_err = dist_err

        if dist_err < STOP_DIST and self._state == 'APPROACH_BALL':
            self._align_stable_count += 1
            if self._align_stable_count >= 25:   # ~0.5s of stable holding before aligning
                self._state = 'ALIGN'
                self._align_stable_count = 0
                self.get_logger().info('→ ALIGN')
            else:
                self._send(0.0, 0.0, 0.0)
                self.get_logger().info('AT BALL — holding', throttle_duration_sec=1.0)
                return

        # APF computes direction + proportional magnitude in world frame
        nearby = [o for o in self._obstacles if math.hypot(rx - o[0], ry - o[1]) < 2.0]
        
        if self._state == 'ALIGN':
            if dist_err > STOP_DIST * 2:
                self._state = 'APPROACH_BALL'
                self._align_stable_count = 0
                self.get_logger().info('Ball moved — back to APPROACH_BALL')

        gx, gy = GOAL_POS
        shoot_dx = gx - bx
        shoot_dy = gy - by
        shoot_d  = math.hypot(shoot_dx, shoot_dy)
        shoot_ux = shoot_dx / max(shoot_d, 1e-6)
        shoot_uy = shoot_dy / max(shoot_d, 1e-6)
        # Behind-ball point: directly opposite goal on shoot axis
        apf_tx = bx - shoot_ux * ALIGN_DIST
        apf_ty = by - shoot_uy * ALIGN_DIST

        if self._state == 'ALIGN':
            robot_behind_check = (rx - bx) * (-shoot_ux) + (ry - by) * (-shoot_uy) > 0.05
            if not robot_behind_check:
                # Reposition: arc around ball using tangential waypoint
                # Robot-from-ball unit vector
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
                vx_w, vy_w = compute_apf(rx, ry, arc_wx, arc_wy, nearby, max_vel=MAX_VEL)
            else:
                # Behind ball — drive to hold position behind it
                vx_w, vy_w = compute_apf(rx, ry, apf_tx, apf_ty, nearby, max_vel=MAX_VEL)
        else:
            vx_w, vy_w = compute_apf(rx, ry, bx, by, nearby, max_vel=MAX_VEL)

        # Apply D term: scale back velocity when closing in fast (d_dist_err is negative when approaching)
        # KD_VEL * d_dist_err is negative when robot is approaching → reduces speed → damps overshoot
        KD_VEL = 0.15  # derivative gain — increase if robot oscillates near ball

        speed_apf = math.hypot(vx_w, vy_w)
        if speed_apf > 1e-6:
            # Compute PD-adjusted speed: APF gives P term, D term subtracts rate of closure
            pd_speed = speed_apf + KD_VEL * d_dist_err   # d_dist_err < 0 when closing → reduces speed
            pd_speed = max(0.0, min(pd_speed, MAX_VEL))   # clamp, never reverse from D alone
            # Reapply direction from APF, magnitude from PD
            vx_w = (vx_w / speed_apf) * pd_speed
            vy_w = (vy_w / speed_apf) * pd_speed

        vt, vn = world_to_robot(vx_w, vy_w, self._yaw)

        # ── Angular: face ball (APPROACH) or face goal (ALIGN) ───────────────────
        if self._state == 'ALIGN':
            if not robot_behind_check:
                # Sub-phase 1: reposition to behind-ball point, face the ball
                theta_ball = math.atan2(by - ry, bx - rx)
                ang_err    = normalize_angle(theta_ball - self._yaw)
                d_ang_err  = normalize_angle(ang_err - self._prev_ang_err) / dt
                self._prev_ang_err = ang_err
                omega = clamp(KP_ANG * ang_err + KD_ANG * d_ang_err, MAX_OMEGA)
                self._align_stable_count = 0
                self._send(vt, vn, omega)
                self.get_logger().info(
                    f'[ALIGN:reposition] ball=({bx:.2f},{by:.2f}) robot=({rx:.2f},{ry:.2f}) '
                    f'dist={dist_err:.2f} ang_err={math.degrees(ang_err):.1f}°',
                    throttle_duration_sec=0.5
                )
            else:
                # Sub-phase 2: behind ball confirmed — rotate to face goal
                theta_goal = math.atan2(gy - by, gx - bx)
                ang_err    = normalize_angle(theta_goal - self._yaw)
                d_ang_err  = normalize_angle(ang_err - self._prev_ang_err) / dt
                self._prev_ang_err = ang_err
                omega = clamp(KP_ANG * ang_err + KD_ANG * d_ang_err, MAX_OMEGA)

                if abs(ang_err) < ANG_ALIGN_TOL:
                    self._align_stable_count += 1
                    if self._align_stable_count >= 10:
                        self._send(0.0, 0.0, 0.0)
                        self._state = 'APPROACH_BALL'
                        self._align_stable_count = 0
                        self.get_logger().info('ALIGNED ✓ — returning to hold')
                        return
                else:
                    self._align_stable_count = 0

                self._send(vt, vn, omega)
                self.get_logger().info(
                    f'[ALIGN:rotate] ball=({bx:.2f},{by:.2f}) robot=({rx:.2f},{ry:.2f}) '
                    f'dist={dist_err:.2f} ang_err={math.degrees(ang_err):.1f}° '
                    f'{"ALIGNED ✓" if abs(ang_err) < ANG_ALIGN_TOL else "rotating..."}',
                    throttle_duration_sec=0.5
                )
        else:
            # ── Angular PD: face ball during approach ─────────────────────────
            theta_ball = math.atan2(by - ry, bx - rx)
            ang_err    = normalize_angle(theta_ball - self._yaw)
            d_ang_err  = normalize_angle(ang_err - self._prev_ang_err) / dt
            self._prev_ang_err = ang_err
            omega  = clamp(KP_ANG * ang_err + KD_ANG * d_ang_err, MAX_OMEGA)

            self._send(vt, vn, omega)
            self.get_logger().info(
                f'[APPROACH_BALL] ball=({bx:.2f},{by:.2f}) robot=({rx:.2f},{ry:.2f}) '
                f'dist={dist_err:.2f} pd_speed={pd_speed:.2f} ang_err={math.degrees(ang_err):.1f}° '
                f'obs={len(nearby)}',
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