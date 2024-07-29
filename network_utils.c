#include <math.h>

#include <gsl/gsl_matrix.h>

#include "include/all.h"

// local header

static inline double _get_rand_double(int n);
static void _xavier_uniform(gsl_matrix *mat);

static inline double _linear(double x);
static inline double _linear_deriv(double x);
static inline double _sigmoid(double x);
static inline double _sigmoid_deriv(double x);
static inline double _relu(double x);
static inline double _relu_deriv(double x);

// main code

void weight_init(struct net_params params, gsl_matrix *mat) {
	switch (params.weights) {
		case weights_zero:				gsl_matrix_set_all(mat, 0);	break;
		case weights_one:				gsl_matrix_set_all(mat, 1);	break;
		// case weights_xavier_normal:		_xavier_normal(mat);		break;
		case weights_xavier_uniform:	_xavier_uniform(mat);		break;
		default:
			printf("Invalid weight initializer provided (%d), setting to -1.0\n", params.weights);
			gsl_matrix_set_all(mat, -1.0);
	}
}

gsl_vector *activation(gsl_vector *inp, enum activation act) {
	switch (act) {
		case act_linear:	return vec_ops(inp, _linear);
		case act_sigmoid:	return vec_ops(inp, _sigmoid);
		case act_relu:		return vec_ops(inp, _relu);
		default:
			printf("ERROR: Invalid activation function specified\n");
			return NULL;
	}
}

gsl_vector *activation_deriv(gsl_vector *inp, enum activation act) {
	switch (act) {
		case act_linear:	return vec_ops(inp, _linear_deriv);
		case act_sigmoid:	return vec_ops(inp, _sigmoid_deriv);
		case act_relu:		return vec_ops(inp, _relu_deriv);
		default:
			printf("ERROR: Invalid activation function specified\n");
			return NULL;
	}
}

// private functions

static inline double _get_rand_double(int n) {
	double val = sqrt(6/(double)n);
	double scale = (double)rand() / (double) RAND_MAX;
	return -val + (scale * val * 2);
}

static void _xavier_uniform(gsl_matrix *mat) {
	size_t n = mat->size1 + mat->size2;
	for (int i = 0; i < mat->size1; i++) {
		for (int j = 0; j < mat->size2; j++) {
			gsl_matrix_set(mat, i, j, _get_rand_double(n));
		}
	}
}

static inline double _linear(double x) {
	return x;
}

static inline double _linear_deriv(double x) {
	return 1;
}

static inline double  _sigmoid(double x) {
	return (double)1/(double)(1 + exp(-x));
}

static inline double _sigmoid_deriv(double x) {
	return _sigmoid(x) * (1 - _sigmoid(x));
}

static inline double _relu(double x) {
	return x > 0.0 ? x : 0.0;
}

static inline double _relu_deriv(double x) {
	return x > 0.0 ? 1.0 : 0.0;
}