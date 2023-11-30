//==========================================================================
// bnn.cpp
//==========================================================================
// @brief: A convolution kernel for CNN on digit recognition

#include "bnn.h"
#include "layer.h"
#include "model.h"
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

//----------------------------------------------------------
// Top function
//----------------------------------------------------------

void dut(hls::stream<bit32_t> &strm_in, hls::stream<bit32_t> &strm_out) {
  bit input[1][I_WIDTH1][I_WIDTH1];
  bit32_t input_l;
  bit32_t output;

  // read one test image into digit
  int bitcount = 0;
  for (int i = 0; i < I_WIDTH1 * I_WIDTH1 / BUS_WIDTH; i++) {
    input_l = strm_in.read();
    for (int j = 0; j < BUS_WIDTH; j++) {
      input[0][bitcount / I_WIDTH1][bitcount % I_WIDTH1] = input_l(j, j);
      bitcount++;
    }
  }
  // call bnn
  output = bnn_xcel(input);

  // write out the result
  strm_out.write(output);
}

//----------------------------------------------------------
// BNN Accelerator
//----------------------------------------------------------
// @param[in] : input - the testing instance
// @return : the predicted digit

bit32_t bnn_xcel(bit input[1][I_WIDTH1][I_WIDTH1]) {
  bit input_padded[I_CHANNEL1][I_WIDTH1 + F_PAD][I_WIDTH1 + F_PAD];
  initialize_padded_memory<I_CHANNEL1, I_WIDTH1 + F_PAD, 1>(input_padded);
  bit conv1[O_CHANNEL1][I_WIDTH1][I_WIDTH1];
  bit conv1_pooled[O_CHANNEL1][I_WIDTH2][I_WIDTH2];
  bit conv1_pooled_padded[O_CHANNEL1][I_WIDTH2 + F_PAD][I_WIDTH2 + F_PAD];

  initialize_padded_memory<O_CHANNEL1, I_WIDTH2 + F_PAD, 0>(
      conv1_pooled_padded);
  bit conv2[O_CHANNEL2][I_WIDTH2][I_WIDTH2];
  bit conv2_pooled[O_CHANNEL2][O_WIDTH][O_WIDTH];

  bit reshaped[I_UNITS1];
  bit16_t dense1[I_UNITS2];
  bit signed1[I_UNITS2];
  bit16_t dense2[NUM_DIGITS];
  bit32_t output;

  /* First Conv Layer */
  pad<I_CHANNEL1, I_WIDTH1>(input, input_padded);
  conv<I_CHANNEL1, O_CHANNEL1, I_WIDTH1 + F_PAD>(input_padded, conv1,
                                                 threshold_conv1, w_conv1);
  max_pool<O_CHANNEL1, I_WIDTH1>(conv1, conv1_pooled);

  /* Second Conv Layer */
  pad<O_CHANNEL1, I_WIDTH2>(conv1_pooled, conv1_pooled_padded);
  conv<O_CHANNEL1, O_CHANNEL2, I_WIDTH2 + F_PAD>(conv1_pooled_padded, conv2,
                                                 threshold_conv2, w_conv2);
  max_pool<O_CHANNEL2, I_WIDTH2>(conv2, conv2_pooled);

  flatten(conv2_pooled, reshaped);

  /* Dense Layers */
  dense<I_UNITS1, I_UNITS2>(reshaped, dense1, w_fc1);
  sign<I_UNITS2>(dense1, signed1);
  dense<I_UNITS2, NUM_DIGITS>(signed1, dense2, w_fc2);
  output = argmax(dense2);

  return output;
}
