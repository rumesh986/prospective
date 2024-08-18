#ifndef __SAVEFILES_H__
#define __SAVEFILES_H__

#define SAVE_NETWORK		0x01
#define SAVE_TRAIN			0x02
#define SAVE_TEST			0x03
#define SAVE_ARRAY			0x04

#define SAVE_TYPE			0x11
#define SAVE_MNIST_TYPE		0x12
#define SAVE_MNIST_PROC		0x13
#define SAVE_GAMMA			0x14
#define SAVE_ALPHA			0x15
#define SAVE_TAU			0x16
#define SAVE_ACT			0x17
#define SAVE_WEIGHTS_INIT	0x18
#define SAVE_NLAYERS		0x19
#define SAVE_LAYERS			0x1A
#define SAVE_NTARGETS		0x1B
#define SAVE_TARGETS		0x1C
#define SAVE_WEIGHTS		0x1D
#define SAVE_NUM_SAMPLES	0x1E
#define SAVE_DELTAW_MAGS	0x1F
#define SAVE_ITER_COUNTS	0x20
#define SAVE_ENERGIES		0x21
#define SAVE_LENERGIES		0x22
#define SAVE_TRAIN_COSTS	0x23
#define SAVE_COSTS			0x24
#define SAVE_LABELS			0x25
#define SAVE_PREDICTIONS	0x26
#define SAVE_OUTPUTS		0x27
#define SAVE_NUM_CORRECT	0x28
#define SAVE_DELTAX_MAGS	0x29

#define SAVE_SIZET			0x31
#define SAVE_DOUBLET		0x32

// #define SAVE_TYPE			0x01
// #define SAVE_MNIST_TYPE		0x02
// #define SAVE_MNIST_PROC		0x03
// #define SAVE_GAMMA			0x03
// #define SAVE_ALPHA			0x03
// #define SAVE_TAU			0x03
// #define SAVE_ACT			0x03
// #define SAVE_WEIGHTS_INIT	0x03
// #define SAVE_NLAYERS		0x03
// #define SAVE_LAYERS			0x03
// #define SAVE_NTARGETS		0x03
// #define SAVE_TARGETS		0x03
// #define SAVE_WEIGHTS		0x03
// #define SAVE_NUM_SAMPLES	0x03
// #define SAVE_DELTAW_MAGS	0x03
// #define SAVE_ITER_COUNTS	0x03
// #define SAVE_ENERGIES		0x03
// #define SAVE_LENERGIES		0x03
// #define SAVE_TRAIN_COSTS	0x03
// #define SAVE_COSTS			0x03
// #define SAVE_LABELS			0x03
// #define SAVE_PREDICTIONS	0x03
// #define SAVE_OUTPUTS		0x03
// #define SAVE_NUM_CORRECT	0x03

// #define SAVE_NETWORK		0x11
// #define SAVE_TRAIN			0x12
// #define SAVE_TEST			0x13

// #define SAVE_NET_WEIGHTS		0x11
// #define SAVE_NET_TARGETS		0x12

// #define SAVE_TRAIN_DELTA_WMAGS	0x21
// #define SAVE_TRAIN_ITER_COUNTS	0x22
// #define SAVE_TRAIN_ENERGIES		0x23
// #define SAVE_TRAIN_LENERGIES	0x24
// #define SAVE_TRAIN_TRAINCOSTS	0x25

// #define SAVE_TEST_LABELS		0x31
// #define SAVE_TEST_PREDICTIONS	0x32
// #define SAVE_TEST_OUTPUTS		0x33
// #define SAVE_TEST_COSTS			0x34

#endif