#ifndef TEMPLATE_LOSS_CONSTANTS_H
#define TEMPLATE_LOSS_CONSTANTS_H

#include <string>


struct LossConstants {
    std::string loss_type;
    double heaviside_k;
    double heaviside_t;
};

#endif //TEMPLATE_LOSS_CONSTANTS_H
