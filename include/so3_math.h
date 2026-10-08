#pragma once

#include <Eigen/Core>

#include <cmath>

// [[nodiscard]] 表示返回值不该被丢掉

// skew 
// 是把一个三维向量变成反对称矩阵
// 叉乘的矩阵形式
// 这个矩阵乘任意向量 𝑤，等于 𝑣 × 𝑤
template <typename Derived>
[[nodiscard]] Eigen::Matrix<typename Derived::Scalar, 3, 3> skew(const Eigen::MatrixBase<Derived>& v)
{
  using T = typename Derived::Scalar;
  Eigen::Matrix<T, 3, 3> K;
  K << T(0), -v(2), v(1), v(2), T(0), -v(0), -v(1), v(0), T(0);
  return K;
}

// Exp(ang)
// 把一个旋转向量变成旋转矩阵
// 输入的 ang 的方向就是转轴的方向，ang 的大小就是转角的大小
// 如果模长小于  1e-7, 那么就认为转角为0，直接返回单位矩阵
// 其他的情况认为机器人是有转动的，首先归一化到单位轴
// 然后使用 Rodrigues 公式计算旋转矩阵
// 公式：R = I + sin(theta) * K + (1 - cos(theta)) * K * K
// K 是反对称矩阵
template <typename T>
[[nodiscard]] Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1>& ang)
{
  const T ang_norm = ang.norm();
  if (ang_norm <= T(1e-7))
  {
    return Eigen::Matrix<T, 3, 3>::Identity();
  }
  const Eigen::Matrix<T, 3, 1> axis = ang / ang_norm;
  const Eigen::Matrix<T, 3, 3> K = skew(axis);
  return Eigen::Matrix<T, 3, 3>::Identity() + std::sin(ang_norm) * K + (T(1) - std::cos(ang_norm)) * K * K;
}

// Exp(ang_vel, dt)
// 计算角速度 ang_vel 在时间 dt 内转出来的旋转矩阵，旋转向量是 ang_vel * dt
template <typename T, typename Ts>
[[nodiscard]] Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1>& ang_vel, const Ts& dt)
{
  const T ang_vel_norm = ang_vel.norm();
  if (ang_vel_norm <= T(1e-7))
  {
    return Eigen::Matrix<T, 3, 3>::Identity();
  }
  const T angle = ang_vel_norm * static_cast<T>(dt);  // 转角
  const Eigen::Matrix<T, 3, 1> axis = ang_vel / ang_vel_norm; // 归一化到单位轴
  const Eigen::Matrix<T, 3, 3> K = skew(axis); // 反对称矩阵
  return Eigen::Matrix<T, 3, 3>::Identity() + std::sin(angle) * K + (T(1) - std::cos(angle)) * K * K;
}

// Exp(v1, v2, v3)
// 把一个三维向量变成旋转矩阵
// 输入的 v1, v2, v3 的方向就是转轴的方向，v1, v2, v3 的大小就是转角的大小
// 如果模长小于  1e-5, 那么就认为转角为0，直接返回单位矩阵
// 其他的情况认为机器人是有转动的，首先归一化到单位轴
// 然后使用 Rodrigues 公式计算旋转矩阵
// 公式：R = I + sin(theta) * K + (1 - cos(theta)) * K * K
// K 是反对称矩阵
template <typename T>
[[nodiscard]] Eigen::Matrix<T, 3, 3> Exp(const T& v1, const T& v2, const T& v3)
{
  const T norm = std::sqrt(v1 * v1 + v2 * v2 + v3 * v3); // 模长
  if (norm <= T(1e-5))
  {
    return Eigen::Matrix<T, 3, 3>::Identity();
  }
  const Eigen::Matrix<T, 3, 1> axis(v1 / norm, v2 / norm, v3 / norm); // 归一化到单位轴
  const Eigen::Matrix<T, 3, 3> K = skew(axis); // 反对称矩阵
  return Eigen::Matrix<T, 3, 3>::Identity() + std::sin(norm) * K + (T(1) - std::cos(norm)) * K * K;
}

// Log(R)
// 把旋转矩阵变回旋转向量 theta * u。迹只给出转角：
//   theta = acos(0.5 * (trace(R) - 1))
// R - R^T 的三个分量组成
//   K = (R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1))
// 它等于 2 * sin(theta) * u，所以
//   theta * u = theta / (2 * sin(theta)) * K
// theta < 0.001 时 theta / (2 * sin(theta)) 约等于 1/2，直接返回 K/2，避免除以很小的 sin(theta)。
template <typename T>
[[nodiscard]] Eigen::Matrix<T, 3, 1> Log(const Eigen::Matrix<T, 3, 3>& R)
{
  const T trace = R.trace();
  const T theta = (trace > T(3) - T(1e-6)) ? T(0) : std::acos(T(0.5) * (trace - T(1)));
  const Eigen::Matrix<T, 3, 1> K(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1)); // 2 * sin(theta) * u
  // theta 很小时用 1/2 代替 theta / (2 * sin(theta))
  return (std::abs(theta) < T(0.001)) ? (T(0.5) * K) : (T(0.5) * theta / std::sin(theta) * K);
}

// RotMtoEuler(rot)
// 把旋转矩阵拆成 ZYX 欧拉角(x, y, z)，也就是滚转、俯仰、偏航
// R = Rz(z) * Ry(y) * Rx(x)
// 和代码对应的元素是
// R(0, 0) = cos(y) * cos(z)
// R(1, 0) = cos(y) * sin(z)
// R(2, 0) = -sin(y)
// R(1, 1) = sin(z) * sin(y) * sin(x) + cos(z) * cos(x)
// R(2, 1) = cos(y) * sin(x)
// R(1, 2) = sin(z) * sin(y) * cos(x) - cos(z) * sin(x)
// R(2, 2) = cos(y) * cos(x)
// sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0))
// x = atan2(R(2, 1), R(2, 2))
// y = atan2(-R(2, 0), sy)
// z = atan2(R(1, 0), R(0, 0))
// |cos(y)| < 1e-6 时，认为俯仰角接近 90度，滚转角和偏航角在同一个轴
// 把偏航角度固定为0，滚转角和俯仰角在同一个轴
// x = atan2(-R(1, 2), R(1, 1))
// y = atan2(-R(2, 0), sy)
template <typename T>
[[nodiscard]] Eigen::Matrix<T, 3, 1> RotMtoEuler(const Eigen::Matrix<T, 3, 3>& rot)
{
  const T sy = std::sqrt(rot(0, 0) * rot(0, 0) + rot(1, 0) * rot(1, 0));
  T x = T(0);
  T y = T(0);
  T z = T(0);
  if (sy >= T(1e-6))
  {
    x = std::atan2(rot(2, 1), rot(2, 2));
    y = std::atan2(-rot(2, 0), sy);
    z = std::atan2(rot(1, 0), rot(0, 0));
  }
  else
  {
    x = std::atan2(-rot(1, 2), rot(1, 1));
    y = std::atan2(-rot(2, 0), sy);
  }
  return Eigen::Matrix<T, 3, 1>(x, y, z);
}
