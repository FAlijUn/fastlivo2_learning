#pragma once

#include "so3_math.h"

#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/core/mat.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <deque>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

// 重力加速度
inline constexpr double kGravity = 9.81;
// 误差状态的维数
// 0-2 旋转分量 3维
// 3-5 位置分量 3维
// 6 逆曝光 1维 图像亮度的相对尺度，用来把不同快门下的像素亮度拉到同一套光度上再比较
// 7-9 速度分量 3维
// 10-12 陀螺零偏 3维
// 13-15 加计零偏 3维
// 16-18 重力 3维
inline constexpr int kDimState = 19; // 状态维数，SO(3) 用 3 维旋转向量表示
// 滤波器刚启动的时候， 19 维状态协方差对角线的初值
inline constexpr double kInitCov = 0.01;
// 视觉稀疏子图预留的容器容量
inline constexpr int kSizeLarge = 500;
inline constexpr int kSizeSmall = 100;

using PointType = pcl::PointXYZINormal;
using PointCloudXYZI = pcl::PointCloud<PointType>;

using V3D = Eigen::Vector3d;
using M3D = Eigen::Matrix3d;

template <int Rows, int Cols>
using MD = Eigen::Matrix<double, Rows, Cols>;

using StateVector = MD<kDimState, 1>;
using StateCovariance = MD<kDimState, kDimState>;

struct Pose6D
{
  double offset_time{}; // 相对本帧第一个雷达点的时间
  double acc[3]{};
  double gyr[3]{};
  double vel[3]{};
  double pos[3]{};
  double rot[9]{}; // 行优先
};

// 从带下标的容器中取出前三个数，做成一个 Eigen 的三维向量
template <typename Container>
[[nodiscard]] auto vec_from_array(const Container& v)
{
  using T = std::decay_t<decltype(v[0])>;
  return Eigen::Matrix<T, 3, 1>(v[0], v[1], v[2]);
}

// 从带下标的容器中取出前九个数，做成一个 Eigen 的三维矩阵
template <typename Container>
[[nodiscard]] auto mat_from_array(const Container& v)
{
  using T = std::decay_t<decltype(v[0])>;
  Eigen::Matrix<T, 3, 3> m;
  m << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8];
  return m;
}

enum class SLAM_MODE : int
{
  ONLY_LIO = 1,
  LIVO = 2
};

// 标记当前包测量应该用哪一种滤波进行更新
enum class EKF_STATE : int
{
  WAIT = 0, // 等待状态，还没有进行过更新
  VIO = 1,  // 通过视觉惯性融合进行更新
  LIO = 2   // 通过激光惯性融合进行更新
};

// 状态向量下标：旋转 3，位置 3，逆曝光 1，速度 3，陀螺零偏 3，加计零偏 3，重力 3。
inline constexpr int kIndexRot = 0;
inline constexpr int kIndexPos = 3;
inline constexpr int kIndexInvExpo = 6;
inline constexpr int kIndexVel = 7;
inline constexpr int kIndexBiasG = 10;
inline constexpr int kIndexBiasA = 13;
inline constexpr int kIndexGravity = 16;

// 一次滤波更新要使用的一小包同步数据
struct MeasureGroup
{
  double vio_time{0.0}; // 视觉更新对应的图像采集时刻，livo 中切分点云的时候，会改成图像曝光的时刻
  double lio_time{0.0}; // 雷达-惯性更新积分到的时刻
  std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu;  // 从上一次更新到本次时刻之间的 IMU 序列，用来传播姿态、速度和位置
  cv::Mat img;  // 本次视觉更新使用的图像
};

// 一次待处理的雷达帧，以及为了和图像对齐而被切开的点云
struct LidarMeasureGroup
{
  // 当前帧雷达扫描的起始时间
  double lidar_frame_beg_time{-0.0};
  // 当前帧雷达扫描的结束时间
  double lidar_frame_end_time{0.0};
  // 滤波器已经传播到的时刻，-1 表示还没有进行过更新
  // 每次 lio 或者 vio 之后修改成本次更新的时间
  double last_lio_update_time{-1.0};
  // 从缓冲区获取的整帧的原始点云
  PointCloudXYZI::Ptr lidar{std::make_shared<PointCloudXYZI>()};
  // 当前这次 lio 要去畸变、配准的点 livo 只保留图像时刻之前的点
  PointCloudXYZI::Ptr pcl_proc_cur{std::make_shared<PointCloudXYZI>()};
  // 图像时刻之后的点，这一次的  vio 做完之后，交换进 pcl_proc_cur 作为下一次 lio 的输入
  PointCloudXYZI::Ptr pcl_proc_next{std::make_shared<PointCloudXYZI>()};
  // 按照时间排好的 MeasureGroup 里面是 imu、图像和对应的更新时刻
  std::deque<MeasureGroup> measures;
  // 标记这一更新包接下来要做哪个估计
  EKF_STATE lio_vio_flg{EKF_STATE::WAIT};
  // 去畸变是处理到点云的哪一个点，开始处理新的点的时候重置为 0 
  int lidar_scan_index_now{0};
};

// 一个带不确定度的雷达点，点本身需要在雷达系下做配准
// 协方差用来给点到平面的残差加权
struct PointWithVar
{
  Eigen::Vector3d point_b{Eigen::Vector3d::Zero()}; // 雷达坐标系下的点
  Eigen::Vector3d point_i{Eigen::Vector3d::Zero()}; // 外参变换后，IMU 坐标系下的点
  Eigen::Vector3d point_w{Eigen::Vector3d::Zero()}; // 世界坐标系下的点 p_w = R * p_i + t
  Eigen::Matrix3d var_nostate{Eigen::Matrix3d::Zero()}; // 去掉状态协方差之后的世界坐标系方差，只保留测量噪声经旋转过去的的一部分
  Eigen::Matrix3d body_var{Eigen::Matrix3d::Zero()}; // 雷达系测量噪声，由测距误差和光束角误差得到
  Eigen::Matrix3d var{Eigen::Matrix3d::Zero()}; // 点在世界坐标系下的总的协方差，测量噪声经过旋转到世界坐标系后，再加上姿态不确定度和位置不确定度
  Eigen::Matrix3d point_crossmat{Eigen::Matrix3d::Zero()};  // IMU 系点的反对称矩阵 
  Eigen::Vector3d normal{Eigen::Vector3d::Zero()};  // 该点所在平面的法向
};

// 滤波器在一次扫描结束时刻的完整状态，以及状态的 19 维误差协方差
struct StatesGroup
{
  // 构造函数先把协方差，设置成 0.01I
  // 逆曝光的协方差设置成 1e-5
  // 陀螺和加计零偏的协方差设置成 1e-5
  StatesGroup()
  {
    cov(kIndexInvExpo, kIndexInvExpo) = 1e-5;
    cov.block<9, 9>(kIndexBiasG, kIndexBiasG) = MD<9, 9>::Identity() * 1e-5;
  }

  StatesGroup(const StatesGroup&) = default;
  StatesGroup& operator=(const StatesGroup&) = default;

  // 流形上的加法，把误差向量加到状态向量上
  // 流形是一组带约束的状态
  // 切空间是某一点旁边，约束被放开之后的一小块平直空间
  [[nodiscard]] StatesGroup operator+(const StateVector& state_add) const
  {
    StatesGroup a = *this;
    // 旋转不能直接相加，需要用 Exp 把误差向量转换成旋转矩阵
    a.rot_end = rot_end * Exp(state_add(0), state_add(1), state_add(2));
    a.pos_end += state_add.block<3, 1>(kIndexPos, 0);
    a.inv_expo_time += state_add(kIndexInvExpo);
    a.vel_end += state_add.block<3, 1>(kIndexVel, 0);
    a.bias_g += state_add.block<3, 1>(kIndexBiasG, 0);
    a.bias_a += state_add.block<3, 1>(kIndexBiasA, 0);
    a.gravity += state_add.block<3, 1>(kIndexGravity, 0);
    return a;
  }

  StatesGroup& operator+=(const StateVector& state_add)
  {
    *this = *this + state_add;
    return *this;
  }

  // 两个状态的差。旋转差是 Log(R_b^T R)，得到 b 到 this 的旋转向量。
  [[nodiscard]] StateVector operator-(const StatesGroup& b) const
  {
    StateVector a;
    a.block<3, 1>(kIndexRot, 0) = Log(b.rot_end.transpose() * rot_end);
    a.block<3, 1>(kIndexPos, 0) = pos_end - b.pos_end;
    a(kIndexInvExpo) = inv_expo_time - b.inv_expo_time;
    a.block<3, 1>(kIndexVel, 0) = vel_end - b.vel_end;
    a.block<3, 1>(kIndexBiasG, 0) = bias_g - b.bias_g;
    a.block<3, 1>(kIndexBiasA, 0) = bias_a - b.bias_a;
    a.block<3, 1>(kIndexGravity, 0) = gravity - b.gravity;
    return a;
  }

  void resetpose()
  {
    rot_end = M3D::Identity();
    pos_end = V3D::Zero();
    vel_end = V3D::Zero();
  }

  // 成员是流形上的值
  // 旋转矩阵
  M3D rot_end{M3D::Identity()};
  // 位置
  V3D pos_end{V3D::Zero()};
  // 速度
  V3D vel_end{V3D::Zero()};
  // 逆曝光时间
  double inv_expo_time{1.0};
  // 陀螺零偏
  V3D bias_g{V3D::Zero()};
  // 角速度零偏
  V3D bias_a{V3D::Zero()};
  // 重力
  V3D gravity{V3D::Zero()};
  // 状态协方差，协方差描述的是切空间里的误差
  StateCovariance cov{StateCovariance::Identity() * kInitCov};
};

// 把某一时刻的 imu 传播结果装进一个 pose 6d，供后面按时间给雷达节点去畸变
template <typename T>
[[nodiscard]] Pose6D set_pose6d(double t, const Eigen::Matrix<T, 3, 1>& a, const Eigen::Matrix<T, 3, 1>& g,
                               const Eigen::Matrix<T, 3, 1>& v, const Eigen::Matrix<T, 3, 1>& p,
                               const Eigen::Matrix<T, 3, 3>& R)
{
  Pose6D pose;
  pose.offset_time = t;
  for (int i = 0; i < 3; ++i)
  {
    pose.acc[i] = a(i);
    pose.gyr[i] = g(i);
    pose.vel[i] = v(i);
    pose.pos[i] = p(i);
    for (int j = 0; j < 3; ++j)
    {
      pose.rot[i * 3 + j] = R(i, j);
    }
  }
  return pose;
}
