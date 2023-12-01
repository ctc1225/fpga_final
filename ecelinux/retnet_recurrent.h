//===========================================================================
// retnet_recurrent.h
//===========================================================================
// @brief: This header file defines the interface for the core functions.

#ifndef RETNET_RECURRENT_H
#define RETNET_RECURRENT_H
#include "model.h"
#include "typedefs.h"
#include <hls_stream.h>

// Top function for synthesis
void dut(hls::stream<bit32_t> &strm_in, hls::stream<bit32_t> &strm_out);

// Top function for retnet_recurrent accelerator
bit16_t retnet_recurrent_xcel(ubit8_t input[D_e]);

// testing projection


#endif // RETNET_RECURRENT_H



// //===========================================================================
// // bnn.h
// //===========================================================================
// // @brief: This header file defines the interface for the core functions.

// #ifndef BNN_H
// #define BNN_H
// #include "model.h"
// #include "typedefs.h"
// #include <hls_stream.h>

// // Top function for synthesis
// void dut(hls::stream<bit32_t> &strm_in, hls::stream<bit32_t> &strm_out);

// // Top function for bnn accelerator
// bit32_t bnn_xcel(bit input[1][I_WIDTH1][I_WIDTH1]);

// #endif
