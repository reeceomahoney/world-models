//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <cstdlib>
#include <set>
#include <cmath>
#include <tuple>
#include <queue>
#include <iostream>
#include "RaisimGymEnv.hpp"

#include "actuation_dynamics_inference/Actuation.hpp"
#include "Utility.hpp"


namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg),
      actuation_(resourceDir_ + "/anymal/parameters/coyote", Eigen::Vector2d{1., 0.1}, 100., 12), 
      visualizable_(visualizable), 
      normDist_(0, 1), uniformDist_(-1,1) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    anymal_ = world_->addArticulatedSystem(resourceDir_+"/anymal/urdf/anymal.urdf");
    anymal_->setName("anymal");
    anymal_->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);
    world_->addGround();

    /// robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// env data
    obDim_ = 36;
    actionDim_ = nJoints_;

    /// initialise containers
    gc_.setZero(gcDim_), gv_.setZero(gvDim_);
    gc_init_.setZero(), gv_init_.setZero();
    gc_rand_.setZero(), gv_rand_.setZero();
    pos_var_.setZero(), vel_var_.setZero();

    jointPositionErrors_.setZero(), jointVelocities_.setZero();
    gf_.setZero(), torque_.setZero();
    actionMean_.setZero(actionDim_), actionStd_.setZero(actionDim_);
    obDouble_.setZero();
    desiredVel_.setZero();

    anymal_->setGeneralizedForce(gf_);
    timeSinceReset_ = 0;
    addVelNoise_ = cfg["add_velocity_noise"].template As<bool>();
    randInitState_ = cfg["random_initial_state"].template As<bool>();


    /// nominal configuration
    gc_init_ << 0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 0.4, -0.8, 0.0, 0.4, -0.8,
        0.0, -0.4, 0.8, 0.0, -0.4, 0.8;
    jointTarget_ = gc_init_.tail(nJoints_);

    /// variances for initial state samples
    pos_var_ << 0.0225, 0.0225, 0.0225, 0.06, 0.06, 0.06, 0.06, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25;
    vel_var_ << 0.012, 0.012, 0.012, 0.4, 0.4, 0.4, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2;

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.5);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);
    terminalRewardCoeff_ = cfg["terminal_reward"].template As<double>();

    /// Command ranges
    maxDesiredVel_[0] = cfg["commands"]["fwdVelMax"].template As<double>();
    maxDesiredVel_[1] = cfg["commands"]["latVelMax"].template As<double>();
    maxDesiredVel_[2] = cfg["commands"]["turnVelMax"].template As<double>();

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer(8086);
      server_->focusOn(anymal_);
    }
  }

  void init() final { }

  void reset() final {
    if (randInitState_) {
      sampleInitialState();
      anymal_->setState(gc_rand_, gv_rand_);
    } else {
      anymal_->setState(gc_init_, gv_init_);
    }
    actuation_.reset();
    timeSinceReset_ = 0.;
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    jointTarget_ = action.cast<double>();
    jointTarget_ = jointTarget_.cwiseProduct(actionStd_);
    jointTarget_ += actionMean_;

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
        if(server_) server_->lockVisualizationServerMutex();

        /// get torques from actuator network
        jointPositionErrors_ = jointTarget_ - gc_.tail(nJoints_);
        torque_ = actuation_.getActuationTorques(jointPositionErrors_, jointVelocities_);

        gf_.tail(nJoints_) = torque_;
        anymal_->setGeneralizedForce(gf_);

        world_->integrate();
        updateObservation();

        if(server_) server_->unlockVisualizationServerMutex();
    }

    /// record rewards
    auto linVelDesired = Eigen::Vector3d{desiredVel_[0], desiredVel_[1], 0.};
    auto angVelDesired = Eigen::Vector3d{0., 0., desiredVel_[2]};

    rewards_.record("linVel", logisticKernel(4*(bodyLinearVel_ - linVelDesired).squaredNorm()));
    rewards_.record("angVel", logisticKernel(4*(bodyAngularVel_[2] - angVelDesired[2])));
    rewards_.record("torque", anymal_->getGeneralizedForce().squaredNorm());
    rewards_.record("jointSpeed", gv_.tail(nJoints_).squaredNorm());

    /// generate commands
    if (int(100*timeSinceReset_) % 200 <= 1) {
        desiredVel_[0] = maxDesiredVel_[0] * uniformDist_(gen_);
        desiredVel_[1] = maxDesiredVel_[1] * uniformDist_(gen_);
        desiredVel_[2] = maxDesiredVel_[2] * uniformDist_(gen_);
    }

    timeSinceReset_ += control_dt_;

    return rewards_.sum();
  }

  void updateObservation() {
    /// update angular and linear velocity
    anymal_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot{};
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    orientation_ = rot.e().row(2).transpose();
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);
    jointAngles_ = gc_.tail(nJoints_);
    jointVelocities_ = gv_.tail(nJoints_);

    /// add nose to velocity observations
    if (addVelNoise_) {
        for (int i = 0; i < nJoints_; i++) {
          if (i < 3) {
            bodyLinearVel_[i] += 0.5 * uniformDist_(gen_);
            bodyAngularVel_[i] += 0.08 * uniformDist_(gen_);
          }
          jointVelocities_[i] += 0.16 * uniformDist_(gen_);
        }
    }

    /// collect observations
    obDouble_.segment(0,3) = orientation_;
    obDouble_.segment(3, nJoints_) = jointAngles_;
    obDouble_.segment(15, 3) = bodyLinearVel_;
    obDouble_.segment(18, 3) = bodyAngularVel_;
    obDouble_.segment(21, nJoints_) = jointVelocities_;
    obDouble_.segment(33, 3) = desiredVel_;
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for(auto& contact: anymal_->getContacts())
      if(std::find(std::begin(footIndices_), std::end(footIndices_), contact.getlocalBodyIndex()) == std::end(footIndices_))
        return true;

    terminalReward = 0.f;
    return false;
  }

  void sampleInitialState() {
      for (int i = 0; i < gcDim_; i++) {
          gc_rand_[i] = gc_init_[i] + pos_var_[i]*normDist_(gen_);
          if (i < gvDim_) gv_rand_[i] = gv_init_[i] + vel_var_[i]*normDist_(gen_);
    }
  }

 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false, addVelNoise_, randInitState_;
  raisim::ArticulatedSystem* anymal_;
  Eigen::VectorXd gc_, gv_;

  Eigen::Matrix<double, 36, 1> obDouble_;
  Eigen::Matrix<double, 19, 1> gc_init_, gc_rand_, pos_var_;
  Eigen::Matrix<double, 18, 1> gv_init_, gv_rand_, vel_var_, gf_;
  Eigen::Matrix<double, 12, 1> actionMean_, actionStd_, torque_;
  Eigen::Matrix<double, 12, 1> jointTarget_, jointPositionErrors_, jointVelocities_, jointAngles_;
  Eigen::Matrix<double, 3, 1> bodyLinearVel_, bodyAngularVel_, desiredVel_, maxDesiredVel_, orientation_;

  int footIndices_[4] = {3, 6, 9, 12};
  double timeSinceReset_, terminalRewardCoeff_;
  Actuation actuation_;

  std::normal_distribution<double> normDist_; std::uniform_real_distribution<double> uniformDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}
