//
// Created by reece on 22/11/22.
//

#ifndef AGILE_LOCOMOTION_UTILITY_HPP
#define AGILE_LOCOMOTION_UTILITY_HPP

#include <math.h>
#include <raisim/World.hpp>

double logisticKernel(double x) {
    return 1/(exp(x) + 2 + exp(-x));
}

std::vector<int> getContactIndices(std::vector<raisim::Contact> &contacts) {
    std::vector<int> contactIdx;
    for (auto &contact: contacts) {
        contactIdx.push_back(contact.getlocalBodyIndex());
    }

    return contactIdx;
}

#endif //AGILE_LOCOMOTION_UTILITY_HPP
