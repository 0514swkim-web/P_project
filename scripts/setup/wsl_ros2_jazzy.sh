#!/usr/bin/env bash
# Ubuntu 24.04 (WSL2 또는 네이티브)에 ROS 2 Jazzy + Gazebo Harmonic + RViz2 + ros2_control 설치.
# ROS 2 공식 문서(Ubuntu deb 설치) 순서를 따른다.
#
# 실행 (Windows PowerShell에서, root로 → sudo 비밀번호 불필요):
#   wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/Dev/P_project/scripts/setup/wsl_ros2_jazzy.sh <리눅스 사용자명>
# 네이티브 Ubuntu에서:
#   sudo bash scripts/setup/wsl_ros2_jazzy.sh $USER
set -euo pipefail

TARGET_USER="${1:-${SUDO_USER:-}}"
if [[ -z "$TARGET_USER" ]]; then
  echo "사용법: bash wsl_ros2_jazzy.sh <리눅스 사용자명>"; exit 1
fi
export DEBIAN_FRONTEND=noninteractive

echo "== 1/6 로케일 (UTF-8) =="
apt-get update
apt-get install -y locales
locale-gen en_US en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

echo "== 2/6 universe 저장소, 기본 도구 =="
apt-get install -y software-properties-common curl git
add-apt-repository -y universe

echo "== 3/6 ROS 2 apt 저장소 (ros2-apt-source) =="
ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
CODENAME=$(. /etc/os-release && echo "${UBUNTU_CODENAME:-${VERSION_CODENAME}}")
curl -L -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${CODENAME}_all.deb"
dpkg -i /tmp/ros2-apt-source.deb

echo "== 4/6 ROS 2 Jazzy Desktop (RViz2 포함) + 개발 도구 =="
apt-get update
apt-get upgrade -y
apt-get install -y ros-jazzy-desktop ros-dev-tools

echo "== 5/6 Gazebo Harmonic 연동 + ros2_control =="
apt-get install -y \
  ros-jazzy-ros-gz \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-xacro \
  ros-jazzy-joint-state-publisher-gui \
  python3-pip python3-venv mesa-utils

if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  rosdep init
fi
sudo -u "$TARGET_USER" rosdep update

echo "== 6/6 사용자 환경 (~/.bashrc) =="
BASHRC="/home/${TARGET_USER}/.bashrc"
grep -q "/opt/ros/jazzy/setup.bash" "$BASHRC" || cat >> "$BASHRC" <<'EOF'

# ---- ROS 2 Jazzy (The-Hand) ----
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=7          # 팀 번호. 같은 네트워크의 다른 팀 ROS와 섞이지 않게
[ -f ~/ros2_ws/install/setup.bash ] && source ~/ros2_ws/install/setup.bash
EOF
chown "$TARGET_USER":"$TARGET_USER" "$BASHRC"

echo "설치 완료. 새 WSL 터미널에서: ros2 --help / gz sim --version / rviz2"
