//===========================================================================
// typedefs.h
//===========================================================================
// @brief: This header defines the shorthand of several ap_uint data types.

#ifndef TYPEDEFS
#define TYPEDEFS

#include <ap_int.h>

typedef bool bit;
typedef ap_int<8> bit8_t;
typedef ap_int<16> bit16_t;
typedef ap_int<32> bit32_t;
typedef ap_uint<2> ubit2_t;
typedef ap_uint<4> ubit4_t;
typedef ap_uint<8> ubit8_t;
typedef ap_uint<16> ubit16_t;
typedef ap_uint<32> ubit32_t;

typedef ap_ufixed<8, 1, AP_TRN, AP_WRAP> dec8_t;

#endif
