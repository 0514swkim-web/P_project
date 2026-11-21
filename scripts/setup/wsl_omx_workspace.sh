#!/usr/bin/env bash
# ROBOTIS open_manipulator (OMX-AI = 팔로워 OMX-F + 리더 OMX-L) 시뮬레이션용 워크스페이스 빌드.
# 먼저 wsl_ros2_jazzy.sh로 ROS 2 Jazzy를 설치해 둘 것.
#
# 실행 (일반 사용자로, rosdep 설치 단계에서만 sudo 필요):
#   bash /mnt/c/Dev/P_project/scripts/setup/wsl_omx_workspace.sh
# sudo 비밀번호 없이 의존성만 먼저 설치하려면 (PowerShell):
#   wsl -d Ubuntu-24.04 -u root -- bash -lc "source /opt/ros/jazzy/setup.bash && cd /home/<user>/ros2_ws && rosdep install --from-paths src -y --ignore-src --rosdistro jazzy"
set -eo pipefail
source /opt/ros/jazzy/setup.bash

WS=~/ros2_ws
mkdir -p "$WS/src"
cd "$WS/src"

# 공식 jazzy 브랜치. 팀원C(Gazebo 담당)이 쓰는 커밋과 맞출 것.
[ -d open_manipulator ] || git clone -b jazzy https://github.com/ROBOTIS-GIT/open_manipulator.git
vcs import . < open_manipulator/open_manipulator_ci.repos

cd "$WS"
if [[ "${SKIP_ROSDEP:-0}" != "1" ]]; then
  rosdep install --from-paths src -y --ignore-src --rosdistro jazzy
fi
# 병렬 빌드를 2개로 제한: 기본값(코어 수만큼)이면 WSL 메모리(RAM의 절반)가 바닥나 멈춘다 (11/11 16GB 노트북에서 확인)
MAKEFLAGS=-j2 colcon build --symlink-install --parallel-workers 2
echo "빌드 완료. 새 터미널에서: ros2 launch open_manipulator_bringup omx_f_gazebo.launch.py"
