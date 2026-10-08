#pragma once

#include <Eigen/Eigen>
#include "common.h"
#include <condition_variable>
#include <nav_msgs/msg/odometry.hpp>
#include <so3_math.h>

class ImuProcess
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuProcess();
    ~ImuProcess();

    void Reset();
    void Reset(double start_timestamp, const sensor_msgs::msg::Imu::ConstSharedPtr &lastimu);

    void set_extrinsic(const V3D &transl, const M3D &rot);
    void set_extrinsic(const V3D &transl);
    void set_extrinsic(const MD(4, 4) & T);

    void set_gyr_cov_scale(const V3D &scaler);
    void set_acc_cov_scale(const V3D &scaler);

    void set_gyr_bias_cov(const V3D &b_g);
    void set_acc_bias_cov(const V3D &b_a);
    void set_inv_expo_cov(const double &inv_expo);
    void set_imu_init_frame_num(const int &num);

    void disable_imu();
    void disable_gravity_est();
    void disable_bias_est();
    void disable_exposure_est();

    void process2(LidarMeasureGroup &lidar_meas, StatesGroup &stat, PointCloudXYZI::Ptr cur_pcl_un_);
    void undistort_pcl(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out);
private:
    void imu_init(const MeasureGroup &meas, StatesGroup &state, int &N);
    void forward_without_imu(LidarMeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &pcl_out);

private:
    double imu_mean_acc_norm_;
    V3D unbiased_gyr_;

    V3D cov_acc_;
    V3D cov_gyr_;
    V3D cov_bias_gyr_;
    V3D cov_bias_acc_;

    double cov_inv_expo_;
    double first_lidar_time_;
    bool imu_time_init_ = false;
    bool imu_need_init_ = true;
    M3D eye3d_;
    V3D zero3d_;

    PointCloudXYZI pcl_wait_proc_;
    sensor_msgs::msg::Imu::ConstSharedPtr last_imu_;
    PointCloudXYZI::Ptr cur_pcl_un_;
    std::vector<Pose6D> imu_pose_;
    M3D lid_rot_to_imu_;
    V3D lid_offset_to_imu_;
    V3D mean_acc_;
    V3D mean_gyr_;
    V3D angvel_last_;
    V3D acc_s_last_;
    double last_prop_end_time_;
    double time_last_scan_;
    int init_iter_num_ = 1;
    static constexpr int kMaxIniCount = 20;
    bool b_first_frame_ = true;
    bool imu_en_ = true;
    bool gravity_est_en_ = true;
    bool ba_bg_est_en_ = true;
    bool exposure_estimate_en_ = true;
};

typedef std::shared_ptr<ImuProcess> ImuProcessPtr;