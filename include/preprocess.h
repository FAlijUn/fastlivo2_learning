#pragma once

#include "common.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <array>
#include <cmath>

// 绝对值大于 1e8 时返回 true，与原来的 IS_VALID 相同。
template <typename T>
[[nodiscard]] bool is_valid(const T& a)
{
  return std::abs(a) > static_cast<T>(1e8);
}

enum class LidarFeature
{
    kNormal, // 还没有分类，初始值
    kPossiblePlane, // 一段平面的两个端点，只是可能落在平面上
    kRealPlane, // 一段平面段中间的点，确实落在平面上
    kEdgePlane, // 相邻两段平面的方向夹角大，交界处当作边缘
    kEdgeJump, //前后距离突然变大，属于深度跳变边缘
    kWire, // 前后两侧都发生跳变
    kZeroPoint, // 0 点
};

// 表示扫描线上当前点的两个相邻方向
enum class Surround
{
    kPrevious, // 前一个点
    kNext, // 后一个点
};

// 描述当前点与扫描线上相邻点的连接是否正常
enum class EJump
{
    kNormal, // 当夹角在 8-170 之间的时候表示的是正常连接
    kZero, // 当夹角小于 8 的时候邻居几乎是沿着射线方向，距离发生突变
    k180, // 当夹角大于 170 的时候邻居几乎是沿着射线方向，距离发生突变
    kInf, // 邻居点无效，且当前点距离大于 10 m, 远处没有回波
    kBlind, // 邻居点无效，且当前点距离小于 10 m, 近处是盲区
};

// 扫描线上一个点在特征提取前的几何摘要。
// 点云本身只存坐标，平面和边缘的判断都读取这份摘要。
struct OrgType
{
  double range{0.0};   // 点到雷达的水平距离，用来丢掉盲区的点
  double dista{0.0};   // 当前点到扫描线上下一个点的距离平方，距离突然变大就是深度跳变
  std::array<double, 2> angle{};  // 射线与前后邻居方向的夹角余弦
  double intersect{2.0};          // 前后两段方向的夹角余弦，2 表示尚未计算
  std::array<EJump, 2> edj{EJump::kNormal, EJump::kNormal};  // 前后两个邻居的连接类型
  LidarFeature ftype{LidarFeature::kNormal};    // 特征标签
};

// Isaac 点云直接使用 pcl::PointXYZ。只做盲区剔除和抽点，输出带 curvature 的点，供后续去畸变使用。
class Preprocess
{
public:
  Preprocess();

  void set(bool feat_en, double bld, int pfilt_num);

  void process(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg, PointCloudXYZI::Ptr& pcl_out) const;

private:
  double blind_;             // 盲区距离，单位 m
  double blind_sqr_;         // 盲区距离的平方，比较时少做一次开方
  int point_filter_num_;     // 抽点间隔，为 1 的时候保留全部的点；为 2 的时候每隔一个点留一个点
  bool feature_enabled_;     // 是否提取平面和边缘特征

  int n_scans_;              // 扫描线数
  int scan_rate_;            // 扫描频率，单位 Hz
  bool given_offset_time_;   // 点云是否自带每个点的时间

  int group_size_;           // 判断是不是平面的时候，沿着扫描线一次取 8 个点
  double dis_a_;             // 平面距离阈值：dis_a_ * range + dis_b_, 点允许的距离
  double dis_b_;
  double inf_bound_;         // 超过该距离且邻居无效，记为远处无回波
  double limit_maxmid_;      // 一组点里最大、中间、最小间距的比例上限
  double limit_midmin_;
  double limit_maxmin_;
  double p2l_ratio_;         // 点到直线程度的阈值，越小越像一条直线
  double jump_up_limit_;     // 夹角大于 170° 的余弦
  double jump_down_limit_;   // 夹角小于 8° 的余弦
  double cos160_;            // 前后两端方向的夹角要大于这个余弦对应的角度才接受深度跳变的边缘
  double edgea_;             // 边缘两侧距离比的上限
  double edgeb_;             // 边缘两侧距离差的上限
  double smallp_intersect_;  // 局部夹角很平的时候，把点标成小平面
  double smallp_ratio_;      // 相邻点间距比小于该值时，视为小平面
  double vx_;                // 计算相邻点距离时暂存坐标差
  double vy_;
  double vz_;
};