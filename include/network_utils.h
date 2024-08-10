#ifndef __NETWORK_UTILS_H__
#define __NETWORK_UTILS_H__

enum weights_init {
	weights_zero,
	weights_one,
	weights_xavier_normal,
	weights_xavier_uniform
};

enum activation {
	act_linear,
	act_sigmoid,
	act_relu
};

// extern struct net_params params;

void weight_init(gsl_matrix *mat, enum weights_init init);

gsl_vector *activation(gsl_vector *inp, enum activation act);
gsl_vector *activation_deriv(gsl_vector *inp, enum activation act);

#endif