#include "preprocess.h"

#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <cmath>
#include <memory>

Preprocess::Preprocess() : feature_enabled_(false), blind_(0.01), point_filter_num_(1)
{
  blind_sqr_ = blind_ * blind_;
  inf_bound_ = 10;
  n_scans_ = 6;
  scan_rate_ = 10;
  group_size_ = 8;
  dis_a_ = 0.01;
  dis_b_ = 0.1;
  p2l_ratio_ = 225;
  limit_maxmid_ = 6.25;
  limit_midmin_ = 6.25;
  limit_maxmin_ = 3.24;
  jump_up_limit_ = 170.0;
  jump_down_limit_ = 8.0;
  cos160_ = 160.0;
  edgea_ = 2;
  edgeb_ = 0.1;
  smallp_intersect_ = 172.5;
  smallp_ratio_ = 1.2;
  given_offset_time_ = false;
  vx_ = 0.0;
  vy_ = 0.0;
  vz_ = 0.0;

  jump_up_limit_ = std::cos(jump_up_limit_ / 180.0 * M_PI);
  jump_down_limit_ = std::cos(jump_down_limit_ / 180.0 * M_PI);
  cos160_ = std::cos(cos160_ / 180.0 * M_PI);
  smallp_intersect_ = std::cos(smallp_intersect_ / 180.0 * M_PI);
}

void Preprocess::set(bool feat_en, double bld, int pfilt_num)
{
  feature_enabled_ = feat_en;
  blind_ = bld;
  blind_sqr_ = bld * bld;
  point_filter_num_ = std::max(1, pfilt_num);
}

void Preprocess::process(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg, PointCloudXYZI::Ptr& pcl_out) const
{
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::fromROSMsg(*msg, cloud);

  if (!pcl_out)
  {
    pcl_out = std::make_shared<PointCloudXYZI>();
  }
  pcl_out->clear();
  pcl_out->reserve(cloud.size() / static_cast<std::size_t>(point_filter_num_) + 1);

  for (std::size_t i = 0; i < cloud.size(); ++i)
  {
    if (static_cast<int>(i) % point_filter_num_ != 0)
    {
      continue;
    }

    const pcl::PointXYZ& point = cloud.points[i];
    const double range = static_cast<double>(point.x) * point.x + static_cast<double>(point.y) * point.y +
                         static_cast<double>(point.z) * point.z;
    if (range < blind_sqr_)
    {
      continue;
    }

    PointType output_point;
    output_point.x = point.x;
    output_point.y = point.y;
    output_point.z = point.z;
    output_point.intensity = 0.f;
    output_point.normal_x = 0.f;
    output_point.normal_y = 0.f;
    output_point.normal_z = 0.f;
    output_point.curvature = 0.f;
    pcl_out->push_back(output_point);
  }
}
