//===========================================================================
// model.h
//===========================================================================
// @brief: This header file include the parameters for BNN

#ifndef MODEL_CONV
#define MODEL_CONV

#include "typedefs.h"

// Filter Constants
const int F = 3; // filter width
const int F_PAD = F - 1;

// Conv Constants
const int I_WIDTH1 = 16;   // conv1 input width
const int I_CHANNEL1 = 1;  // conv1 input width
const int O_CHANNEL1 = 16; // conv1 output channels
const int I_WIDTH2 = 8;    // conv2 input width
const int O_CHANNEL2 = 32; // conv2 output channels
const int O_WIDTH = 4;     // conv2 output width

// Dense Constants
const int I_UNITS1 = O_WIDTH * O_WIDTH * O_CHANNEL2; // num of fc1 input units
const int I_UNITS2 = 256;

// Other Constants
const int NUM_DIGITS = 10;
const int BUS_WIDTH = 32;

const bit w_conv1[I_CHANNEL1][O_CHANNEL1][F][F] = {
#include "data/weight_conv1"
};

const bit w_conv2[O_CHANNEL1][O_CHANNEL2][F][F] = {
#include "data/weight_conv2"
};

const bit w_fc1[I_UNITS1][I_UNITS2] = {
#include "data/weight_fc1"
};

const bit w_fc2[I_UNITS2][NUM_DIGITS] = {
#include "data/weight_fc2"
};

const bit8_t threshold_conv1[O_CHANNEL1] = {
#include "data/threshold_conv1"
};

const bit8_t threshold_conv2[O_CHANNEL2] = {
#include "data/threshold_conv2"
};

#endif
