# Autonomous Soccer Agent — ROS 2

RoboCup Small Size League · Striker Agent  


A ROS 2-based autonomous striker for the RoboCup SSL simulation environment (grSim). The robot perceives the field via vision data, navigates to the ball using Artificial Potential Fields, positions itself behind the ball on the goal axis, and executes a kick — fully autonomously.

---

<strong>Demo:</strong> [Watch demo video](https://drive.google.com/file/d/1uI_IxuzODIZjKPOrIM5jjYpK-uJ9CHf0/view?usp=sharing)


## Package Structure

```
.
├── main.py       # Entry point — rclpy init / spin / shutdown
├── control.py    # ROS 2 node, state machine, vision callback, control loop
├── apf.py        # Artificial Potential Field planner
├── utils.py      # Math helpers — angle normalization, clamping, frame transforms
└── packet.py     # grSim UDP packet encoder (no external protobuf dependency)
```

---

## How It Works

### State Machine

The striker runs a three-state FSM:

```
APPROACH_BALL ──► ALIGN ──► READY ──► (kick) ──┐
      ▲               │         │               │
      └───────────────┴─────────┘ (ball moved)  │
      ◄──────────────────────────────────────────┘
```

| State | Behaviour |
|---|---|
| `APPROACH_BALL` | Navigate to ball with APF + PD damping. Hold for 25 ticks (~0.5 s) at ball before transitioning. |
| `ALIGN` | Arc around ball to get behind it on the ball→goal axis. Rotate to face goal. Declare `READY` after 10 stable ticks within 0.10 rad. |
| `READY` | Drive into ball with kicker armed at 6.0 m/s. Reset on ball departure. |

Each state has a **ball-moved guard** — if the ball relocates beyond threshold during `ALIGN` or `READY`, the machine resets to `APPROACH_BALL` immediately.

---

### Motion Model

The robot is **holonomic** (omni-directional). Commands are sent in robot-local frame as `(veltangent, velnormal, velangular)`. World-frame planner output is converted via:

```
vt =  cos(−θ) · vx − sin(−θ) · vy
vn =  sin(−θ) · vx + cos(−θ) · vy
```

**Angular PD controller:**
```
ω = Kp · e_θ + Kd · (Δe_θ / Δt)
```
Angular error is always normalized to `[−π, π]` via `atan2(sin, cos)` before differencing to prevent derivative spikes at the wrap boundary.

**Translational damping:**
```
v_pd = v_apf + Kd_vel · (Δdist / Δt)
```
As the robot closes in, `Δdist/Δt` is negative — automatically reducing speed and preventing overshoot near the ball.

---

### Planning — Artificial Potential Fields

| Force Component | Formula | Effect |
|---|---|---|
| Attractive (near) | `K_att · (target − robot)` | Linear pull within D_star = 1.0 m |
| Attractive (far) | `K_att · D_star · unit(target − robot)` | Capped pull; prevents speed runaway |
| Repulsive | `K_rep · (1/ρ − 1/ρ₀) / ρ² · unit(away)` | Pushes away within ρ₀ = 0.28 m |
| Tangential | `K_tan · mag · dot · tangent` | Sidesteps robot around obstacles toward goal |

- N=7 nearest obstacles considered per tick
- Obstacle radius reduced by 0.18 m (robot body) for surface-to-surface distance
- Repulsive magnitude capped at 20.0 to prevent numerical blowup

**Behind-ball arc maneuver:** During `ALIGN`, if the robot is not yet behind the ball, a tangential arc waypoint is computed by rotating the robot-from-ball unit vector 90° in the direction that reduces angle to the behind-ball target. The robot skirts around the ball at half speed without pushing it.

---

## Parameters

| Parameter | Value | Purpose |
|---|---|---|
| `MAX_VEL` | 2.5 m/s | Maximum translational speed |
| `MAX_OMEGA` | 6.0 rad/s | Maximum angular speed |
| `STOP_DIST` | 0.15 m | Proximity threshold to ball |
| `ALIGN_DIST` | 0.22 m | Hold distance behind ball |
| `KP_ANG / KD_ANG` | 4.0 / 0.05 | Angular PD gains |
| `K_ATT / K_REP` | 4.0 / 6.0 | APF attractive/repulsive gains |
| `RHO_0` | 0.28 m | Obstacle influence radius |
| `K_TAN` | 3.0 | Tangential sidestep gain |
| `KICK_SPEED` | 6.0 m/s | grSim kick speed |
| `HOLD_TICKS / ALIGN_TICKS` | 25 / 10 | Stability dwell counters |

---

## Setup & Usage

### Prerequisites

- ROS 2 (Humble or later)
- [grSim](https://github.com/RoboCup-SSL/grSim)
- [ssl_ros_bridge](https://github.com/SSL-A-Team/ssl_ros_bridge)
- `ssl_league_msgs` ROS 2 package

### Run

**Terminal 1 — start grSim**
```bash
./grSim
```

**Terminal 2 — start the ROS 2 bridge**
```bash
ros2 run ssl_ros_bridge ssl_ros_bridge
```

**Terminal 3 — run the striker**
```bash
ros2 run <your_package_name> main
```

### Configuration

Edit the constants at the top of `control.py`:

```python
ROBOT_ID   = 0          # Robot ID in grSim
TEAM       = 'blue'     # 'blue' or 'yellow'
GRSIM_HOST = '127.0.0.1'
GRSIM_PORT = 20011
```

---

## Topics

| Direction | Topic | Type |
|---|---|---|
| Input | `/ssl_vision_bridge/vision_messages` | `ssl_league_msgs/msg/VisionWrapper` |
| Output | UDP to grSim port 20011 | grSim protobuf packet |

---

## Edge Cases Handled

| Situation | How |
|---|---|
| Ball moves during ALIGN or READY | Reset to `APPROACH_BALL` immediately |
| No vision data | Skip control tick, log warning |
| Stale obstacle positions | Expire after 1 s if not refreshed |
| Duplicate vision frames (dt < 5 ms) | Skip ball velocity update |
| Angle wrap at ±π | `normalize_angle()` on all error terms |
| Near-zero obstacle distance | `rho = max(rr − 0.18, 0.01)` clamp |
| APF local minimum | Tangential force + arc waypoint maneuver |
| Overshoot near ball | Translational D term reduces speed on closure |

---

## References

- Khatib, O. (1986). *Real-time obstacle avoidance for manipulators and mobile robots.* IJRR 5(1).
- van den Berg, J. et al. (2008). *Reciprocal velocity obstacles for real-time multi-agent navigation.* ICRA.
- [grSim](https://github.com/RoboCup-SSL/grSim) · [ssl_ros_bridge](https://github.com/SSL-A-Team/ssl_ros_bridge)
