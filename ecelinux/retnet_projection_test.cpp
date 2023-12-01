//=========================================================================
// retnet_recurrent_test.cpp
//=========================================================================
// @brief: testbench for retnet_recurrent
// application

#include <iostream>
#include <fstream>
#include <bitset>
#include "retnet_projection.h"
#include "timer.h"

using namespace std;

// Number of test instances
const int TEST_SIZE = 1;

//------------------------------------------------------------------------
// Helper function for reading tokens
//------------------------------------------------------------------------

void read_tokens(ubit8_t test_tokens[TEST_SIZE][D_e]) {
  std::ifstream infile("data/projection_token.dat");
  if (infile.is_open()) {
    for (int index = 0; index < TEST_SIZE; index++) {
      for (int embedding = 0; embedding < D_e; embedding++) {
        int i;
        infile >> i;
        test_tokens[index][embedding] = i;
        // std::cout << "Print out read tokens: " << test_tokens[index][embedding] << std::endl;
      }
    }
    infile.close();
  }
}

void read_golden(ubit16_t test_golden[TEST_SIZE][D_e]) {
  std::ifstream infile("data/projection_golden.dat");
  if (infile.is_open()) {
    for (int index = 0; index < TEST_SIZE; index++) {
      for (int embedding = 0; embedding < D_e; embedding++) {
        int i;
        infile >> i;
        test_golden[index][embedding] = i;
        // std::cout << "Print out read goldens: " << test_golden[index][embedding] << std::endl;
      }
    }
    infile.close();
  }
}

//------------------------------------------------------------------------
// Digitrec testbench
//------------------------------------------------------------------------

int main() {
  // HLS streams for communicating with the cordic block
  hls::stream<bit32_t> digitrec_in;
  hls::stream<bit32_t> digitrec_out;

  ubit8_t test_tokens[TEST_SIZE][D_e];
  ubit16_t test_golden[TEST_SIZE][D_e];

  // read test images and labels
  read_tokens(test_tokens);
  read_golden(test_golden);
  ubit32_t test_token;

  // Timer
  Timer timer("digirec retnet recurrent");
  timer.start();

  // pack images to 32-bit and transmit to dut function
  for (int test = 0; test < TEST_SIZE; test++) {
    for (int i = 0; i < D_e * IN_BIT_WIDTH / BUS_WIDTH; i++) {
      for (int j = 0; j < BUS_WIDTH / IN_BIT_WIDTH; j++) {
        // std::cout << "packing data j: " << test_tokens[test][j] << std::endl;
        test_token((j+1)*IN_BIT_WIDTH-1, j*IN_BIT_WIDTH) = test_tokens[test][j];
      }
      // std::bitset<32> test_token_bin(test_token);
      // std::cout << "test token: " << test_token_bin << std::endl;
      digitrec_in.write(test_token);
    }

    // perform prediction
    dut(digitrec_in, digitrec_out);

    // read output line by line into tokens
    ubit32_t output_l;
    ubit16_t output_token[D_e];
    int bitcount = 0;
    for (int i = 0; i < D_e * PROJECTION_OUT_BIT_WIDTH / BUS_WIDTH; i++) {
      output_l = digitrec_out.read();
      for (int j = 0; j < BUS_WIDTH; j++) {
        output_token[bitcount/PROJECTION_OUT_BIT_WIDTH][bitcount % PROJECTION_OUT_BIT_WIDTH] = output_l[j];
        bitcount++;
      }
    }

    // print out output & golden
    std::cout << "output token: [";
    for (int i = 0; i < D_e; i++) {
      std::cout << output_token[i] << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "golden token: [";
    for (int i = 0; i < D_e; i++) {
      std::cout << test_golden[test][i] << " ";
    }
    std::cout << "]" << std::endl;

    // verification, compare with golden
    for (int i = 0; i < D_e; i++) {
      if (output_token[i] != test_golden[test][i]) {
        std::cout << "fail test token " << test << std::endl;
        break;
      }
      else if (i == D_e-1) std::cout << "pass test token " << test << std::endl;
    }
  }
  timer.stop();

  std::cout << "finish testing projection" << std::endl;

  return 0;
}



// //=========================================================================
// // bnn_test.cpp
// //=========================================================================
// // @brief: testbench for Binarized Neural Betwork(BNN) digit recongnition
// // application

// #include <iostream>
// #include <fstream>
// #include "bnn.h"
// #include "timer.h"

// using namespace std;

// // Number of test instances
// const int TEST_SIZE = 100;

// //------------------------------------------------------------------------
// // Helper function for reading images and labels
// //------------------------------------------------------------------------

// void read_test_images(int8_t test_images[TEST_SIZE][256]) {
//   std::ifstream infile("data/test_images.dat");
//   if (infile.is_open()) {
//     for (int index = 0; index < TEST_SIZE; index++) {
//       for (int pixel = 0; pixel < 256; pixel++) {
//         int i;
//         infile >> i;
//         test_images[index][pixel] = i;
//       }
//     }
//     infile.close();
//   }
// }

// void read_test_labels(int test_labels[TEST_SIZE]) {
//   std::ifstream infile("data/test_labels.dat");
//   if (infile.is_open()) {
//     for (int index = 0; index < TEST_SIZE; index++) {
//       infile >> test_labels[index];
//     }
//     infile.close();
//   }
// }

// //------------------------------------------------------------------------
// // Digitrec testbench
// //------------------------------------------------------------------------

// int main() {
//   // HLS streams for communicating with the cordic block
//   hls::stream<bit32_t> digitrec_in;
//   hls::stream<bit32_t> digitrec_out;

//   int8_t test_images[TEST_SIZE][256];
//   int test_labels[TEST_SIZE];

//   // read test images and labels
//   read_test_images(test_images);
//   read_test_labels(test_labels);
//   bit32_t test_image;
//   float correct = 0.0;

//   // Timer
//   Timer timer("digirec BNN");
//   timer.start();

//   // pack images to 32-bit and transmit to dut function
//   for (int test = 0; test < TEST_SIZE; test++) {
//     for (int i = 0; i < I_WIDTH1 * I_WIDTH1 / BUS_WIDTH; i++) {
//       for (int j = 0; j < BUS_WIDTH; j++) {
//         test_image(j, j) = test_images[test][i * BUS_WIDTH + j];
//       }
//       digitrec_in.write(test_image);
//     }

//     // perform prediction
//     dut(digitrec_in, digitrec_out);

//     // check results
//     if (digitrec_out.read() == test_labels[test])
//       correct += 1.0;
//   }
//   timer.stop();

//   // Calculate accuracy
//   std::cout << "Accuracy: " << correct / TEST_SIZE << std::endl;

//   return 0;
// }
