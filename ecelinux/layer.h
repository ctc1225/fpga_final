//===========================================================================
// layer.h
//===========================================================================
// @brief: This header file defines the interface for the core functions.

#ifndef LAYER_H
#define LAYER_H

#include "model.h"
#include "typedefs.h"


//----------------------------------------------------------
// Perform Convolution Layer
//----------------------------------------------------------
// @param[in] : input - input fmaps
//              threshold - threshold for batchnorm operation
//              M - number of input fmaps
//              N - number of output fmaps
//              I - width of input fmaps
//              weight - layer weights
// @param[out] : output - output fmaps
// template <int M, int N, int I>
// void conv(bit input[M][I][I], bit output[N][I - F + 1][I - F + 1],
//           const bit8_t threshold[N], const bit weight[M][N][F][F]) {
// #pragma HLS ARRAY_RESHAPE variable=input complete dim=1
// #pragma HLS ARRAY_RESHAPE variable=output complete dim=1
// #pragma HLS ARRAY_RESHAPE variable=threshold complete dim=1
// #pragma HLS ARRAY_RESHAPE variable=weight complete dim=1
//   int num_accum = F * F * M;
//   for (int n = 0; n < N; n++) {
//     for (int x = 0; x < I - F + 1; x++) {
//       for (int y = 0; y < I - F + 1; y++) {
//         bit16_t accum = 0;
//         for (int c = 0; c < F; c++) {
//           #pragma HLS PIPELINE
//           for (int r = 0; r < F; r++) {
//             #pragma HLS UNROLL
//             for (int m = 0; m < M; m++) {
//               #pragma HLS UNROLL
//               accum += input[m][y + r][x + c] == weight[m][n][r][c];
//             }
//           }
//         }
//         accum = (accum << 1) - num_accum;
//         output[n][y][x] = accum > threshold[n] ? 1 : 0;
//       }
//     }
//   }
// }

//----------------------------------------------------------
// Flatten the Output from Conv Layer
//----------------------------------------------------------
// @param[in] : input - output fmaps from the last conv layer
// @param[out] : output - input famps of the first dense layer

// void flatten(bit input[O_CHANNEL2][O_WIDTH][O_WIDTH], bit output[I_UNITS1]) {
//   for (int c = 0; c < O_CHANNEL2; c++) {
//     #pragma HLS PIPELINE
//     for (int y = 0; y < O_WIDTH; y++) {
//       #pragma HLS UNROLL
//       for (int x = 0; x < O_WIDTH; x++) {
//         #pragma HLS UNROLL
//         int o_index = c + (x + y * O_WIDTH) * O_CHANNEL2;
//         output[o_index] = input[c][y][x];
//       }
//     }
//   }
// }

//----------------------------------------------------------
// Perform Projection to Vector
//----------------------------------------------------------
// @param[in] : input - input fmaps
//              M - input dimension
//              N - output dimension
//              weight - projection matrix weights
// @param[out] : output - output vector

template <int M, int N>
void projection(ubit8_t input[M], ubit16_t output[M], const bit8_t weight[M][N]) {
  for (int n = 0; n < N; n++) {
    ubit16_t accum = 0;
    for (int m = 0; m < M; m++) {
      // std::cout << "input[" << m << "]: " << input[m] << ", weight[" << m << "][" << n << "]: " << weight[m][n] << std::endl;
      accum += input[m] * weight[m][n];
    }
    output[n] = accum;
  }
}


#endif



// //===========================================================================
// // layer.h
// //===========================================================================
// // @brief: This header file defines the interface for the core functions.

// #ifndef LAYER_H
// #define LAYER_H

// #include "model.h"
// #include "typedefs.h"

// //----------------------------------------------------------
// // Padding
// //----------------------------------------------------------
// // @param[in] : input - input fmaps
// //              M - number of input fmaps
// //              I - width of input fmaps
// // @param[out] : output - output fmaps
// template <int M, int I>
// void pad(bit input[M][I][I], bit output[M][I + F_PAD][I + F_PAD]) {
// #pragma HLS ARRAY_RESHAPE variable=input complete dim=1
// #pragma HLS ARRAY_RESHAPE variable=output complete dim=1
//   for (int m = 0; m < M; m++) {
//   #pragma HLS PIPELINE
//     for (int x = 0; x < I; x++) {
//       #pragma HLS UNROLL
//       for (int y = 0; y < I; y++) {
//         #pragma HLS UNROLL
//         output[m][y + F_PAD / 2][x + F_PAD / 2] = input[m][y][x];
//       }
//     }
//   }
// }

// //----------------------------------------------------------
// // Initialize Padded Memory with Constant
// //----------------------------------------------------------
// // @param[in] : input - input fmaps to be initialized
// // @param[out] : output - output fmaps
// template <int M, int I, int C>
// void initialize_padded_memory(bit input[M][I][I]) {
//   for (int m = 0; m < M; m++) {
//     for (int x = 0; x < I; x++) {
//       #pragma HLS PIPELINE
//       for (int y = 0; y < I; y++) {
//         #pragma HLS UNROLL
//         input[m][x][y] = C;
//       }
//     }
//   }
// }

// //----------------------------------------------------------
// // Perform Convolution Layer
// //----------------------------------------------------------
// // @param[in] : input - input fmaps
// //              threshold - threshold for batchnorm operation
// //              M - number of input fmaps
// //              N - number of output fmaps
// //              I - width of input fmaps
// //              weight - layer weights
// // @param[out] : output - output fmaps
// template <int M, int N, int I>
// void conv(bit input[M][I][I], bit output[N][I - F + 1][I - F + 1],
//           const bit8_t threshold[N], const bit weight[M][N][F][F]) {
// #pragma HLS ARRAY_RESHAPE variable=input complete dim=1
// #pragma HLS ARRAY_RESHAPE variable=output complete dim=1
// #pragma HLS ARRAY_RESHAPE variable=threshold complete dim=1
// #pragma HLS ARRAY_RESHAPE variable=weight complete dim=1
//   int num_accum = F * F * M;
//   for (int n = 0; n < N; n++) {
//     for (int x = 0; x < I - F + 1; x++) {
//       for (int y = 0; y < I - F + 1; y++) {
//         bit16_t accum = 0;
//         for (int c = 0; c < F; c++) {
//           #pragma HLS PIPELINE
//           for (int r = 0; r < F; r++) {
//             #pragma HLS UNROLL
//             for (int m = 0; m < M; m++) {
//               #pragma HLS UNROLL
//               accum += input[m][y + r][x + c] == weight[m][n][r][c];
//             }
//           }
//         }
//         accum = (accum << 1) - num_accum;
//         output[n][y][x] = accum > threshold[n] ? 1 : 0;
//       }
//     }
//   }
// }

// //----------------------------------------------------------
// // Max pooling
// //----------------------------------------------------------
// // @param[in] : input - input fmaps
// //              M - number of input fmaps
// //              I - width of input fmaps
// // @param[out] : output - output fmaps
// template <int M, int I>
// void max_pool(bit input[M][I][I], bit output[M][I / 2][I / 2]) {
//   for (int m = 0; m < M; m++) {
//     for (int x = 0; x < I / 2; x++) {
//       for (int y = 0; y < I / 2; y++) {
//         bit max = 0;
//         for (int c = 0; c < 2; c++) {
//           #pragma HLS PIPELINE
//           for (int r = 0; r < 2; r++) {
//             #pragma HLS UNROLL
//             if (input[m][2 * y + r][2 * x + c])
//               max = 1;
//           }
//         }
//         output[m][y][x] = max;
//       }
//     }
//   }
// }

// //----------------------------------------------------------
// // Flatten the Output from Conv Layer
// //----------------------------------------------------------
// // @param[in] : input - output fmaps from the last conv layer
// // @param[out] : output - input famps of the first dense layer

// void flatten(bit input[O_CHANNEL2][O_WIDTH][O_WIDTH], bit output[I_UNITS1]) {
//   for (int c = 0; c < O_CHANNEL2; c++) {
//     #pragma HLS PIPELINE
//     for (int y = 0; y < O_WIDTH; y++) {
//       #pragma HLS UNROLL
//       for (int x = 0; x < O_WIDTH; x++) {
//         #pragma HLS UNROLL
//         int o_index = c + (x + y * O_WIDTH) * O_CHANNEL2;
//         output[o_index] = input[c][y][x];
//       }
//     }
//   }
// }

// //----------------------------------------------------------
// // Perform Sign Layer
// //----------------------------------------------------------
// // @param[in] : input - input fmaps
// //              M - number of input and output channels
// // @param[out] : output - output fmaps

// template <int M> void sign(bit16_t input[M], bit output[M]) {
//   for (int m = 0; m < M; m++) {
//     #pragma HLS PIPELINE
//     output[m] = (input[m] > 0) ? 1 : 0;
//   }
// }

// //----------------------------------------------------------
// // Perform Argmax Layer
// //----------------------------------------------------------
// // @param[in] : input - input channels
// // @param[out] : output - argmax of the inputs

// bit4_t argmax(bit16_t input[NUM_DIGITS]) {
//   bit16_t max = input[0];
//   bit4_t max_id = 0;
//   for (int i = 1; i < NUM_DIGITS; i++) {
//     #pragma HLS PIPELINE
//     if (input[i] > max) {
//       max = input[i];
//       max_id = i;
//     }
//   }
//   return max_id;
// }

// //----------------------------------------------------------
// // Perform Dense Layer
// //----------------------------------------------------------
// // @param[in] : input - input fmaps
// //              M - number of input fmaps
// //              N - number of output fmaps
// //              weight - layer weights
// // @param[out] : output - output fmaps

// template <int M, int N>
// void dense(bit input[M], bit16_t output[N], const bit weight[M][N]) {
//   for (int n = 0; n < N; n++) {
//     bit16_t accum = 0;
//     #pragma HLS PIPELINE
//     for (int m = 0; m < M; m++) {
//       #pragma HLS UNROLL
//       int w_index = m * N + n;
//       accum += input[m] == weight[m][n]; // XNOR
//     }
//     output[n] = (accum << 1) - M;
//   }
// }

// #endif
