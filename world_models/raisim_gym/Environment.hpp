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
#include "Visualization.hpp"


double logisticKernel(double x) {
    return 1/(exp(x) + 2 + exp(-x));
}

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg),
      actuation_(resourceDir_ + "/anymal/parameters/coyote", Eigen::Vector2d{1., 0.1}, 100., 12), 
      visualizable_(visualizable), 
      normDist_(0, 1), uniformDist_(-1,1),
      visualizationHandler_(visualizable_) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    anymal_ = world_->addArticulatedSystem(resourceDir_+"/anymal/urdf/anymal.urdf");
    anymal_->setName("anymal");
    anymal_->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);
    world_->addGround(0., "ground_material");

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
    pTarget_.setZero();

    anymal_->setGeneralizedForce(gf_);
    timeSinceReset_ = 0;
    addVelNoise_ = cfg["add_velocity_noise"].template As<bool>();
    randInitState_ = cfg["random_init_state"].template As<bool>();
    expertInitState_ = cfg["expert_init_state"].template As<bool>();

    /// nominal configuration
    gc_init_ << 0., 0., 0.55, 1.0, 0., 0., 0.,
            -0.138589, 0.480936, -0.761428, 0.138589, 0.480936, -0.761428,
            -0.138589, -0.480936, 0.761428, 0.138589, -0.480936, 0.761428;
    jointTarget_ = gc_init_.tail(nJoints_);

    /// variances for initial state samples
    pos_var_ << 0.0225, 0.0225, 0.0225, 0.06, 0.06, 0.06, 0.06, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25;
    vel_var_ << 0.012, 0.012, 0.012, 0.4, 0.4, 0.4, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2;

    /// action scaling
    actionMean_.setConstant(0);
    actionStd_.setConstant(1);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);
    terminalRewardCoeff_ = cfg["terminal_reward"].template As<double>();

    /// Commands
    randomCommands_ = cfg["commands"]["random"].template As<bool>();
    maxDesiredVel_[0] = cfg["commands"]["fwdVelMax"].template As<double>();
    maxDesiredVel_[1] = cfg["commands"]["latVelMax"].template As<double>();
    maxDesiredVel_[2] = cfg["commands"]["turnVelMax"].template As<double>();

    /// set pd gains
    /* Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_); */
    /* jointPgain.setZero(); */
    /* jointDgain.setZero(); */
    /* jointPgain.tail(nJoints_).setConstant(85.0); */
    /* jointDgain.tail(nJoints_).setConstant(0.6); */
    /* anymal_->setPdGains(jointPgain, jointDgain); */

    
    /// Set the material property for each of the collision bodies of the robot
    for (auto &collisionBody: anymal_->getCollisionBodies()) {
        if (collisionBody.colObj->name.find("FOOT") != std::string::npos) {
            collisionBody.setMaterial("foot_material");
        } else {
            collisionBody.setMaterial("robot_material");
        }
    }

    /* auto materialPairGroundFootProperties = */
    /*         world_->getMaterialPairProperties("ground_material", "foot_material"); */
    /* world_->setMaterialPairProp( */
    /*         "ground_material", "foot_material", */
    /*         0.6, materialPairGroundFootProperties.c_r, */
    /*         materialPairGroundFootProperties.r_th */
    /* ); */

    /* auto materialPairGroundRobotProperties = */
    /*         world_->getMaterialPairProperties("ground_material", "robot_material"); */
    /* world_->setMaterialPairProp("ground_material", "robot_material", */
    /*                             0.4, materialPairGroundRobotProperties.c_r, */
    /*                             materialPairGroundRobotProperties.r_th */
    /* ); */

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer(8086);
      server_->focusOn(anymal_);
      visualizationHandler_.setServer(server_);
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
    expertDatasetLoaded_ = false;
    actuation_.reset();
    timeSinceReset_ = 0.;
    updateObservation();
  }

  void expertReset(const Eigen::Ref<EigenVec>& init_data) {
    init_data_ = init_data.cast<double>();
    anymal_->setState(init_data_.segment(0, gcDim_), init_data_.segment(gcDim_, gvDim_));
    desiredVel_ = init_data_.segment(gcDim_ + gvDim_, 3);
    expertDatasetLoaded_ = true;
    actuation_.reset();
    timeSinceReset_ = 0.;
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    jointTarget_ = action.cast<double>();

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
        if(server_) server_->lockVisualizationServerMutex();

        /// get torques from actuator network
        jointPositionErrors_ = jointTarget_ - gc_.tail(nJoints_);
        torque_ = actuation_.getActuationTorques(jointPositionErrors_, jointVelocities_);

        gf_.tail(nJoints_) = torque_.cwiseMax(-80.).cwiseMin(80.);
        anymal_->setGeneralizedForce(gf_);

        world_->integrate();
        updateObservation();

        if (visualizable_) {
            server_->focusOn(anymal_);
            visualizationHandler_.updateVelocityVisual(anymal_, desiredVel_, server_);
        }

        if(server_) server_->unlockVisualizationServerMutex();
    }

    /// record rewards
    auto linVelDesired = Eigen::Vector3d{desiredVel_[0], desiredVel_[1], 0.};
    auto angVelDesired = Eigen::Vector3d{0., 0., desiredVel_[2]};
    rewards_.record("linVel", logisticKernel(4*(bodyLinearVel_ - linVelDesired).squaredNorm()));
    rewards_.record("angVel", logisticKernel(4*(bodyAngularVel_[2] - angVelDesired[2])));

    /// generate commands
    if (!expertDatasetLoaded_) {
        if (randomCommands_) {
            if (int(100*timeSinceReset_) % 400 <= 1) {
                if (uniformDist_(gen_) < 0.1) {
                    // generate zero command 10% of the time
                    desiredVel_.setZero();
                } else {
                    desiredVel_[0] = maxDesiredVel_[0] * uniformDist_(gen_);
                    desiredVel_[1] = maxDesiredVel_[1] * uniformDist_(gen_);
                    desiredVel_[2] = maxDesiredVel_[2] * uniformDist_(gen_);
                }
            }
        } else {
            desiredVel_ << 0.5, 0., 0.;
        }
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
    obDouble_.segment(15, 3) = bodyAngularVel_;
    obDouble_.segment(18, nJoints_) = jointVelocities_;
    obDouble_.segment(30, 3) = bodyLinearVel_;
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

  void setTarget(const Eigen::Ref<EigenVec>& decodedObs) {
      raisim::Vec<3> euler;
      raisim::Vec<4> quat;
      Eigen::Matrix<double, 36, 1> obs = decodedObs.cast<double>();

      euler[0] = obs[0];
      euler[1] = obs[1];
      euler[2] = obs[2];
      raisim::eulerVecToQuat(euler, quat);

      pTarget_[2] = 0.61;
      pTarget_.segment(3, 4) = quat.e();
      pTarget_.segment(7, 12) = obs.segment(3, 12);
      vTarget_.segment(0, 3) = obs.segment(30, 3);
      vTarget_.segment(3, 3) = obs.segment(15, 3);
      vTarget_.segment(6, 12) = obs.segment(18, 12);

      for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
          if (server_) server_->lockVisualizationServerMutex();
          anymal_->setState(pTarget_, vTarget_);
          world_->integrate();
          if (server_) server_->unlockVisualizationServerMutex();
      }
  }

  int getInitRow() {
      return initRow_;
  }

 private:
  int gcDim_, gvDim_, nJoints_, datasetSize_, initRow_;
  bool visualizable_ = false, addVelNoise_, randInitState_, expertInitState_, randomCommands_;
  bool expertDatasetLoaded_ = false;
  raisim::ArticulatedSystem* anymal_;
  Eigen::VectorXd gc_, gv_;
  Eigen::MatrixXd expertDataset_;

  Eigen::Matrix<double, 40, 1> init_data_;
  Eigen::Matrix<double, 36, 1> obDouble_;
  Eigen::Matrix<double, 19, 1> gc_init_, gc_rand_, pos_var_;
  Eigen::Matrix<double, 18, 1> gv_init_, gv_rand_, vel_var_, gf_;

  Eigen::Matrix<double, 19, 1> pTarget_;
  Eigen::Matrix<double, 18, 1> vTarget_;

  Eigen::Matrix<double, 12, 1> actionMean_, actionStd_, torque_;
  Eigen::Matrix<double, 12, 1> jointTarget_, jointPositionErrors_, jointVelocities_, jointAngles_;
  Eigen::Matrix<double, 3, 1> bodyLinearVel_, bodyAngularVel_, desiredVel_, maxDesiredVel_, orientation_;

  int footIndices_[4] = {3, 6, 9, 12};
  double timeSinceReset_, terminalRewardCoeff_;
  Actuation actuation_;

  std::normal_distribution<double> normDist_;
  std::uniform_real_distribution<double> uniformDist_;
  thread_local static std::mt19937 gen_;

  VisualizationHandler visualizationHandler_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}
