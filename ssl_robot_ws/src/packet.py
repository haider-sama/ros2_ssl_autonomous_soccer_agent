"""
packet.py - grSim UDP packet encoder (grSim_Packet / grSim_Commands proto2)

No external protobuf library required.
Wire types: 0=varint  1=64-bit  2=len-delimited  5=32-bit
"""

import struct
import time


# Low-level proto2 primitives

def _varint(v: int) -> bytes:
    buf = []
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            buf.append(b | 0x80)
        else:
            buf.append(b)
            break
    return bytes(buf)

def _tag(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)

def _field_varint(field: int, v: int) -> bytes:
    return _tag(field, 0) + _varint(v)

def _field_bool(field: int, v: bool) -> bytes:
    return _tag(field, 0) + _varint(1 if v else 0)

def _field_float(field: int, v: float) -> bytes:
    return _tag(field, 5) + struct.pack('<f', v)

def _field_double(field: int, v: float) -> bytes:
    return _tag(field, 1) + struct.pack('<d', v)

def _field_msg(field: int, data: bytes) -> bytes:
    return _tag(field, 2) + _varint(len(data)) + data


# Public API

def build_packet(robot_id: int,
                 veltangent: float,
                 velnormal: float,
                 velangular: float,
                 is_yellow: bool,
                 kickspeedx: float = 0.0,
                 kickspeedz: float = 0.0,
                 spinner: bool = False) -> bytes:
    """
    Encode a grSim_Packet with one robot command.

    grSim_Robot_Command fields:
      id=1          robot id
      kickspeedx=2  kick speed x  (0 = no kick)
      kickspeedz=3  kick speed z  (0 = no chip)
      veltangent=4  forward vel   (robot frame, m/s)
      velnormal=5   lateral vel   (robot frame, left=+, m/s)
      velangular=6  angular vel   (CCW=+, rad/s)
      spinner=7     dribbler on/off
      wheelsspeed=8 False = use veltangent/velnormal/velangular
    """

    robot_cmd = (
        _field_varint(1, robot_id)     +   # id
        _field_float (2, kickspeedx)   +   # kickspeedx
        _field_float (3, kickspeedz)   +   # kickspeedz
        _field_float (4, veltangent)   +   # veltangent (forward)
        _field_float (5, velnormal)    +   # velnormal  (lateral)
        _field_float (6, velangular)   +   # velangular
        _field_bool  (7, spinner)      +   # spinner
        _field_bool  (8, False)            # wheelsspeed=False, use vel fields
    )

    # grSim_Commands
    commands = (
        _field_double(1, time.time())       +   # timestamp
        _field_bool  (2, is_yellow)         +   # isteamyellow
        _field_msg   (3, robot_cmd)             # robot_commands (repeated)
    )

    # grSim_Packet
    packet = _field_msg(1, commands)

    return packet