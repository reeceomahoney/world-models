//
// Created by Reece O'Mahoney on 22/05/23.
//

#ifndef WORLD_MODELS_UTILITY_HPP
#define WORLD_MODELS_UTILITY_HPP

#include "RaisimGymEnv.hpp"


inline Eigen::Vector3d rotationMatrixToRPY(const Eigen::Matrix3d &R) {
    Eigen::Vector3d RPY(3);
    RPY(0) = -atan2(-R(2, 1), R(2, 2));

    // asin(x) returns nan if x > 1 || x < -1, this is potentially possible due
    // to floating point precision issues, so we check if this occurs and bound
    // the result to +/- (pi/2) based on the sign of x, *IFF* the input value
    // was finite.  If it was not, do not make it finite, that would hide an
    // error elsewhere.

    RPY(1) = -asin(R(2, 0));

    if (std::isfinite(R(2, 0)) && !std::isfinite(RPY(1))) {
        RPY(1) = copysign(M_PI / 2.0, R(2, 0));
    }

    RPY(2) = -atan2(-R(1, 0), R(0, 0));

    return RPY;
}

inline Eigen::Matrix3d rpyToRotationMatrix(const Eigen::Vector3d &RPY) {
    double cx = cos(RPY(0));
    double sx = sin(RPY(0));
    double cy = cos(RPY(1));
    double sy = sin(RPY(1));
    double cz = cos(RPY(2));
    double sz = sin(RPY(2));

    Eigen::Matrix3d R(3, 3);
    R(0, 0) = cz * cy;
    R(0, 1) = -sz * cx + cz * sy * sx;
    R(0, 2) = sz * sx + cz * sy * cx;
    R(1, 0) = sz * cy;
    R(1, 1) = cz * cx + sz * sy * sx;
    R(1, 2) = -cz * sx + sz * sy * cx;
    R(2, 0) = -sy;
    R(2, 1) = cy * sx;
    R(2, 2) = cy * cx;

    return R;
}


#endif //WORLD_MODELS_UTILITY_HPP
