#!/bin/bash

WS="/media/haider/Drive1/University Work/semester_08/mobile_robotics/project/ssl_robot_ws"
GRSIM="/media/haider/Drive1/University Work/semester_08/mobile_robotics/project/grSim/bin"

gnome-terminal --tab --title="ROS Bridge" -- bash -c "
    cd \"$WS\" && \
    source install/setup.bash && \
    ros2 launch ssl_ros_bridge ssl_ros_bridge.launch.xml
    exec bash
" &

sleep 1

gnome-terminal --tab --title="grSim" -- bash -c "
    cd \"$GRSIM\" && \
    ./grSim
    exec bash
" &

sleep 2

gnome-terminal --tab --title="Controller" -- bash -c "
    cd \"$WS\" && \
    source install/setup.bash && \
    echo 'Ready — run: python3 src/sample.py' && \
    exec bash
"
