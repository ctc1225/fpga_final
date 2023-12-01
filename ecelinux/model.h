//===========================================================================
// model.h
//===========================================================================
// @brief: This header file include the parameters for retnet_recurrent

#ifndef MODEL_RETNET_RECURRENT_H
#define MODEL_RETNET_RECURRENT_H

#include "typedefs.h"

// Dimension constants
const int D_e = 4; // embedding dimension
const int D_h = 4; // head dimension
const int D_k = 4; // key dimension
const int D_v = 4; // value dimension

const int H   = 2; // # of head

const int NUM_OF_THETA = D_e/2; // number of theta for rotary position encoding

const int IN_BIT_WIDTH = 8; // input bit width
const int PROJECTION_OUT_BIT_WIDTH = 16; // projection output bit width
const int BUS_WIDTH = 32;


const bit8_t w_WQ[D_e][D_e] = {
#include "data/weight_WQ"
};

const bit8_t w_WK[D_e][D_e] = {
#include "data/weight_WK"
};

const bit8_t w_WV[D_e][D_v] = {
#include "data/weight_WV"
};

// sin, cos values for rotary position encoding
const dec8_t angle_1_sin[NUM_OF_THETA] = {
#include "data/angle_1_sin"
};

const dec8_t angle_1_cos[NUM_OF_THETA] = {
#include "data/angle_1_cos"
};

#endif // MODEL_RETNET_RECURRENT_H



// //===========================================================================
// // model.h
// //===========================================================================
// // @brief: This header file include the parameters for BNN

// #ifndef MODEL_CONV
// #define MODEL_CONV

// #include "typedefs.h"

// // Filter Constants
// const int F = 3; // filter width
// const int F_PAD = F - 1;

// // Conv Constants
// const int I_WIDTH1 = 16;   // conv1 input width
// const int I_CHANNEL1 = 1;  // conv1 input width
// const int O_CHANNEL1 = 16; // conv1 output channels
// const int I_WIDTH2 = 8;    // conv2 input width
// const int O_CHANNEL2 = 32; // conv2 output channels
// const int O_WIDTH = 4;     // conv2 output width

// // Dense Constants
// const int I_UNITS1 = O_WIDTH * O_WIDTH * O_CHANNEL2; // num of fc1 input units
// const int I_UNITS2 = 256;

// // Other Constants
// const int NUM_DIGITS = 10;
// const int BUS_WIDTH = 32;

// const bit w_conv1[I_CHANNEL1][O_CHANNEL1][F][F] = {
// #include "data/weight_conv1"
// };

// const bit w_conv2[O_CHANNEL1][O_CHANNEL2][F][F] = {
// #include "data/weight_conv2"
// };

// const bit w_fc1[I_UNITS1][I_UNITS2] = {
// #include "data/weight_fc1"
// };

// const bit w_fc2[I_UNITS2][NUM_DIGITS] = {
// #include "data/weight_fc2"
// };

// const bit8_t threshold_conv1[O_CHANNEL1] = {
// #include "data/threshold_conv1"
// };

// const bit8_t threshold_conv2[O_CHANNEL2] = {
// #include "data/threshold_conv2"
// };

// #endif
