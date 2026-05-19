
import math
import socket

import rclpy
from rclpy.node import Node
from ssl_league_msgs.msg import VisionWrapper

from utils  import normalize_angle, clamp, world_to_robot
from packet import build_packet
from apf    import compute_apf


# ── Configuration ─────────────────────────────────────────────────

ROBOT_ID   = 0
TEAM       = 'blue'

GRSIM_HOST = '127.0.0.1'
GRSIM_PORT = 20011

KP_ANG    = 4.0
MAX_OMEGA = 6.0
MAX_VEL   = 1.8    # m/s
STOP_DIST = 0.15   # stop when this close to ball (m)


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

        self.create_timer(0.02, self._control_loop)
        self.get_logger().info(
            f'Phase 3 APF ready — team={TEAM} robot_id={ROBOT_ID}'
        )

    # ── Vision callback ───────────────────────────────────────────
    def _vision_cb(self, msg: VisionWrapper):
        if not msg.detection:
            return

        ball_candidates = []
        our_robots      = {}
        opp_robots      = {}

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
            self._ball = (best[1], best[2])

        if ROBOT_ID in our_robots:
            _, x, y, yaw = our_robots[ROBOT_ID]
            self._robot = (x, y)
            self._yaw   = yaw

        # only update obstacles when opponents are actually detected
        new_obs = [(v[1], v[2]) for v in opp_robots.values()]
        if new_obs:
            self._obstacles = new_obs

    # ── Control loop ─────────────────────────────────────────────
    def _control_loop(self):
        if self._ball is None or self._robot is None:
            self.get_logger().info(
                'Waiting for vision...', throttle_duration_sec=2.0)
            return

        bx, by = self._ball
        rx, ry = self._robot

        dist = math.hypot(bx - rx, by - ry)

        if dist < STOP_DIST:
            self._send(0.0, 0.0, 0.0)
            return

        vx_w, vy_w = compute_apf(rx, ry, bx, by, self._obstacles, max_vel=MAX_VEL)
        

        angle_to_ball = math.atan2(by - ry, bx - rx)
        ang_err = normalize_angle(angle_to_ball - self._yaw)
        omega = clamp(KP_ANG * ang_err, MAX_OMEGA)

        vt, vn = world_to_robot(vx_w, vy_w, self._yaw)
        self._send(vt, vn, omega)

        self.get_logger().info(
            f'ball=({bx:.2f},{by:.2f}) robot=({rx:.2f},{ry:.2f}) '
            f'dist={dist:.2f} vt={vt:.2f} vn={vn:.2f} '
            f'obs={len(self._obstacles)}',
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