#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include "include/all.h"

// local header

struct _network {
	gsl_matrix **weights;
	struct net_params params;
};

double _calc_energy(gsl_vector **layers);
void _calc_lenergy(gsl_vector **layers, gsl_vector **lenergies, size_t iter);

// main code

static struct _network _net;

void build_network(size_t inp_len, size_t out_len, struct net_params *params) {
	params->nlayers += 2;
	size_t *lengths = malloc(sizeof(size_t) * params->nlayers);
	lengths[0] = inp_len;
	lengths[params->nlayers-1] = out_len;

	for (int i = 1; i < params->nlayers-1; i++)
		lengths[i] = params->lengths[i-1];
	
	free(params->lengths);
	params->lengths = lengths;

	_net.params = *params;

	_net.weights = malloc(sizeof(gsl_matrix *) * params->nlayers-1);
	for (int i = 0; i < params->nlayers-1; i++) {
		_net.weights[i] = gsl_matrix_calloc(params->lengths[i+1], params->lengths[i]);
		weight_init(_net.params, _net.weights[i]);
	}
}

struct net_params *load_network(char *filename) {
	FILE *file = fopen(filename, "rb");
	if (!file) {
		printf("Failed to open network file (%s)\n", filename);
		return NULL;
	}

	fread(&_net.params.mnist, sizeof(mnist_db), 1, file);
	fread(&_net.params.mnist_proc, sizeof(mnist_processing), 1, file);

	fread(&_net.params.gamma, sizeof(double), 1, file);
	fread(&_net.params.alpha, sizeof(double), 1, file);
	fread(&_net.params.tau, sizeof(size_t), 1, file);
	fread(&_net.params.act, sizeof(enum activation), 1, file);
	fread(&_net.params.weights, sizeof(enum weights_init), 1, file);

	fread(&_net.params.nlayers, sizeof(size_t), 1, file);
	_net.params.lengths = malloc(sizeof(size_t) * _net.params.nlayers);
	fread(_net.params.lengths, sizeof(size_t), _net.params.nlayers, file);
	printf("nlayers: %ld\n", _net.params.nlayers);
	for (int i = 0; i < _net.params.nlayers; i++)
		printf("%ld\n", _net.params.lengths[i]);
	
	fread(&_net.params.ntargets, sizeof(size_t), 1, file);
	_net.params.targets = malloc(sizeof(size_t) * _net.params.ntargets);
	fread(_net.params.targets, sizeof(size_t), _net.params.ntargets, file);

	_net.weights = malloc(sizeof(gsl_matrix *) * _net.params.nlayers-1);
	for (int i = 0; i < _net.params.nlayers-1; i++)
		_net.weights[i] = file2mat(file);
	
	fclose(file);
	return &_net.params;
}

int save_network(char *filename) {
	FILE *file = fopen(filename, "wb");
	if (!file) {
		printf("Failed to open network file (%s)\n", filename);
		return 1;
	}

	fwrite(&_net.params.mnist, sizeof(mnist_db), 1, file);
	fwrite(&_net.params.mnist_proc, sizeof(mnist_processing), 1, file);

	fwrite(&_net.params.gamma, sizeof(double), 1, file);
	fwrite(&_net.params.alpha, sizeof(double), 1, file);
	fwrite(&_net.params.tau, sizeof(size_t), 1, file);
	fwrite(&_net.params.act, sizeof(enum activation), 1, file);
	fwrite(&_net.params.weights, sizeof(enum weights_init), 1, file);

	fwrite(&_net.params.nlayers, sizeof(size_t), 1, file);
	fwrite(_net.params.lengths, sizeof(size_t), _net.params.nlayers, file);
	
	fwrite(&_net.params.ntargets, sizeof(size_t), 1, file);
	fwrite(_net.params.targets, sizeof(size_t), _net.params.ntargets, file);

	for (int i = 0; i < _net.params.nlayers-1; i++)
		mat2file(_net.weights[i], file);
	
	fclose(file);
}

void free_network() {
	for (int i = 0; i < _net.params.nlayers-1; i++)
		gsl_matrix_free(_net.weights[i]);
	
	free(_net.weights);
	free(_net.params.lengths);
	free(_net.params.targets);
}

struct traindata *train(size_t num_samples) {
	printf("Starting Training \n");

	// prepare training data
	int max_count = mnist_get_count(mnist_train);
	mnist_dataset data[_net.params.ntargets];
	for (int i = 0; i < _net.params.ntargets; i++) {
		data[i] = mnist_get_dataset(mnist_train, _net.params.targets[i], _net.params.mnist_proc);
		if (data[i].count < max_count)
			max_count = data[i].count;
	}
	// size_t num_samples = max_count * _net.params.ntargets;
	// size_t num_samples = 100;
	if (num_samples == 0)
		num_samples = max_count * _net.params.ntargets;

	// prepare logging variables
	#ifdef LOGGING
		struct traindata *ret = malloc(sizeof(struct traindata));
		ret->delta_w_mags = malloc(sizeof(gsl_vector *) * (_net.params.nlayers-1));
		ret->energies = malloc(sizeof(gsl_vector *) * num_samples);
		ret->iter_counts = gsl_vector_calloc(num_samples);
		ret->num_samples = num_samples;
		ret->lenergies = malloc(sizeof(gsl_vector **) * num_samples);

		for (int l = 0; l < _net.params.nlayers-1; l++)
			ret->delta_w_mags[l] = gsl_vector_calloc(num_samples);

	#endif

	gsl_vector *layers[_net.params.nlayers];
	gsl_vector *epsilons[_net.params.nlayers];
	gsl_vector *deltax[_net.params.nlayers];

	gsl_matrix *deltaw[_net.params.nlayers-1];

	for (int l = 0; l < _net.params.nlayers; l++) {
		layers[l] = gsl_vector_calloc(_net.params.lengths[l]);
		epsilons[l] = gsl_vector_calloc(_net.params.lengths[l]);
		deltax[l] = gsl_vector_calloc(_net.params.lengths[l]);
	}

	for (int l = 0; l < _net.params.nlayers-1; l++)
		deltaw[l] = gsl_matrix_calloc(_net.params.lengths[l+1], _net.params.lengths[l]);
	
	int target_counter = -1;
	for (int i = 0; i < num_samples; i++) {
		#ifdef LOGGING
			ret->energies[i] = gsl_vector_calloc(_net.params.tau);
			ret->lenergies[i] = malloc(sizeof(gsl_vector *) * _net.params.nlayers-1);
			for (int l = 0; l < _net.params.nlayers-1; l++)
				ret->lenergies[i][l] = gsl_vector_calloc(_net.params.tau);
		#endif
		int cur_target = i % _net.params.ntargets;
		if (cur_target == 0)
			target_counter++;

		gsl_vector *label_vec = gsl_vector_calloc(_net.params.ntargets);
		gsl_vector_set(label_vec, cur_target, 1.0);
		gsl_vector_memcpy(layers[0], data[cur_target].images[target_counter]);
		gsl_vector_memcpy(layers[_net.params.nlayers-1], label_vec);
		// gsl_vector_memcpy(layers[_net.params.nlayers-1], data[cur_target].label_vec);

		int reduction_count = 2;
		double gamma = _net.params.gamma;

		// initial forprop (match initial energy state as PCLayer code)
		// for (int l = 0; l < _net.params.nlayers-2; l++) {
		// 	gsl_vector *act = activation(layers[l], _net.params.act);
		// 	gsl_blas_dgemv(CblasNoTrans, 1.0, _net.weights[l], act, 0.0, layers[l+1]);
		// 	gsl_vector_free(act);

		// 	// gsl_vector_memcpy(epsilons[l], layers[l]);
		// }

		// relaxation stage
		for (int t = 0; t < _net.params.tau; t++) {
			// calculate epsilons
			for (int l = 0; l < _net.params.nlayers-1; l++) {
				gsl_vector *act = activation(layers[l], _net.params.act);
				gsl_vector_memcpy(epsilons[l+1], layers[l+1]);
				gsl_blas_dgemv(CblasNoTrans, -1.0, _net.weights[l], act, 1.0, epsilons[l+1]);

				gsl_vector_free(act);
			}

			// modify x (layers)
			for (int l = 1; l < _net.params.nlayers-1; l++) {
				gsl_vector *act_deriv = activation_deriv(layers[l], _net.params.act);

				gsl_blas_dgemv(CblasTrans, gamma, _net.weights[l], epsilons[l+1], 0.0, deltax[l]);
				gsl_vector_mul(deltax[l], act_deriv);

				gsl_blas_daxpy(-gamma, epsilons[l], deltax[l]);

				gsl_vector_add(layers[l], deltax[l]);

				gsl_vector_free(act_deriv);
			}
			
			for (int l = 0; l < _net.params.nlayers; l++)
				gsl_vector_set_zero(deltax[l]);
			
			#ifdef LOGGING
				gsl_vector_set(ret->energies[i], t, _calc_energy(layers));

				_calc_lenergy(layers, ret->lenergies[i], t);
				// early stop condition
				if (t == 0)
					continue;

				// printf("Energies: %f %f\n", gsl_vector_get(ret->energies[i], t), gsl_vector_get(ret->energies[i], t-1));
				if (gsl_vector_get(ret->energies[i], t) >= gsl_vector_get(ret->energies[i], t-1)) {
					printf("Energy not reducting\n");
					if (reduction_count > 0) {
						gamma /= (double)2;
						reduction_count--;
					} else {
						gamma = _net.params.gamma;
						printf("Stopping early\n");
						gsl_vector_set(ret->iter_counts, i, t);
						break;
					}
				}
			#endif
		}

		#ifdef LOGGING
			if (gsl_vector_get(ret->iter_counts, i) == 0)
				gsl_vector_set(ret->iter_counts, i, _net.params.tau-1);
		#endif

		// modify weights
		for (int l = 0; l < _net.params.nlayers-1; l++) {
			gsl_vector *act = activation(layers[l], _net.params.act);

			// con combine next three lines into single dger call (replace deltaw with _net.weights[l])
			// doing it this way in order to calculate delta_ws
			gsl_blas_dger(_net.params.alpha, epsilons[l+1], act, deltaw[l]);
			gsl_matrix_add(_net.weights[l], deltaw[l]);

			#ifdef LOGGING
				gsl_vector_set(ret->delta_w_mags[l], i, frobenius_norm(deltaw[l]));
			#endif

			gsl_vector_free(act);
		}

		gsl_vector_free(label_vec);

		for (int l = 0; l < _net.params.nlayers; l++) {
			gsl_vector_set_zero(layers[l]);
			gsl_vector_set_zero(epsilons[l]);
		}
	}

	printf("Completed training\n");

	for (int l = 0; l < _net.params.nlayers; l++) {
		gsl_vector_free(layers[l]);
		gsl_vector_free(epsilons[l]);
		gsl_vector_free(deltax[l]);
	}

	for (int l = 0; l < _net.params.nlayers-1; l++)
		gsl_matrix_free(deltaw[l]);


	for (int i = 0; i < _net.params.ntargets; i++)
		free_mnist_dataset(data[i]);

	#ifdef LOGGING
		return ret;
	#else
		// return (struct traindata){};
		return NULL;
	#endif
}

int save_traindata(struct traindata *data, char *filename) {
	if (!data) {
		printf("Error: Invalid traindata\n");
		return 1;
	}

	FILE *file = fopen(filename, "wb");
	if (!file) {
		printf("Error: Failed to open traindata file (%s) \n", filename);
		return 2;
	}

	fwrite(&data->num_samples, sizeof(size_t), 1, file);
	fwrite(&_net.params.nlayers, sizeof(size_t), 1, file);
	fwrite(&_net.params.ntargets, sizeof(size_t), 1, file);
	fwrite(_net.params.targets, sizeof(size_t), _net.params.ntargets, file);

	for (int l = 0; l < _net.params.nlayers-1; l++)
		vec2file(data->delta_w_mags[l], file);
	
	vec2file(data->iter_counts, file);

	for (int i = 0; i < data->num_samples; i++)
		vec2file(data->energies[i], file);
	
	for (int i = 0; i < data->num_samples; i++) {
		for (int l = 0; l < _net.params.nlayers-1; l++)
			vec2file(data->lenergies[i][l], file);
	}
	
	fclose(file);
	return 0;
}

void free_traindata(struct traindata *data) {
	for (int l = 0; l < _net.params.nlayers-1; l++) 
		gsl_vector_free(data->delta_w_mags[l]);
	
	for (int i = 0; i < data->num_samples; i++)
		gsl_vector_free(data->energies[i]);
	
	free(data->delta_w_mags);
	free(data->energies);
	gsl_vector_free(data->iter_counts);
	free(data);
}

struct testdata *test() {
	printf("Preparing to test\n");

	size_t num_samples = 0;

	mnist_dataset data[_net.params.ntargets];
	for (int i = 0; i < _net.params.ntargets; i++) {
		data[i] = mnist_get_dataset(mnist_test, _net.params.targets[i], _net.params.mnist_proc);
		num_samples += data[i].count;
	}

	#ifdef LOGGING
		struct testdata *ret = malloc(sizeof(struct testdata));
		ret->confusion = gsl_matrix_calloc(_net.params.ntargets, _net.params.ntargets);
		ret->costs = malloc(sizeof(gsl_vector *) * _net.params.ntargets);
		ret->labels = gsl_vector_calloc(num_samples);
		ret->outputs = malloc(sizeof(gsl_vector *) * num_samples);
		ret->num_samples = num_samples;
		size_t counter = 0;
	#endif

	size_t num_correct = 0;
	gsl_vector *layers[_net.params.nlayers];
	for (int l = 0; l < _net.params.nlayers; l++)
		layers[l] = gsl_vector_calloc(_net.params.lengths[l]);
	
	gsl_vector *label_vec = gsl_vector_calloc(_net.params.ntargets);
	gsl_vector *cost_vec = gsl_vector_calloc(_net.params.ntargets);
	
	for (int target_i = 0; target_i < _net.params.ntargets; target_i++) {
		#ifdef LOGGING
			ret->costs[target_i] = gsl_vector_calloc(data[target_i].count);
		#endif

		gsl_vector_set(label_vec, target_i, 1);

		for (int i = 0; i < data[target_i].count; i++) {
			gsl_vector_memcpy(layers[0], data[target_i].images[i]);

			for (int l = 0; l < _net.params.nlayers-1; l++) {
				gsl_vector *act = activation(layers[l], _net.params.act);
				gsl_blas_dgemv(CblasNoTrans, 1.0, _net.weights[l], act, 0.0, layers[l+1]);
				gsl_vector_free(act);
			}

			int prediction_index = gsl_vector_max_index(layers[_net.params.nlayers-1]);
			int prediction = _net.params.targets[prediction_index];

			gsl_vector_memcpy(cost_vec, layers[_net.params.nlayers-1]);
			gsl_vector_sub(cost_vec, label_vec);
			double cost = gsl_blas_dnrm2(cost_vec);
			gsl_vector_set_zero(cost_vec);

			if (prediction == data[target_i].label)
				num_correct++;

			// printf("Test result: target: %d, prediction: %d, cost: %.4f\n", data[target_i].label, prediction, cost);
			
			#ifdef LOGGING
				gsl_vector_set(ret->costs[target_i], i, cost);
				gsl_matrix_set(ret->confusion, prediction_index, target_i, gsl_matrix_get(ret->confusion, prediction_index, target_i)+1);

				gsl_vector_set(ret->labels, counter, data[target_i].label);
				ret->outputs[counter] = gsl_vector_alloc(_net.params.ntargets);
				gsl_vector_memcpy(ret->outputs[counter], layers[_net.params.nlayers-1]);
				counter++;
			#endif
		}

		gsl_vector_set_zero(label_vec);
	}

	double accuracy = (double)num_correct / (double)num_samples;
	printf("Completed testing\n");
	printf("Summary: \n");
	printf("Tested on %ld images of ", num_samples);
	for(int i = 0; i < _net.params.ntargets; i++)
		printf("%ld (%ld),", _net.params.targets[i], data[i].count);
	printf("\b\n");
	printf("Accuracy = %ld/%ld = %.4f\n", num_correct, num_samples, accuracy);

	gsl_vector_free(label_vec);
	for (int l = 0; l < _net.params.nlayers; l++)
		gsl_vector_free(layers[l]);
	
	for (int i = 0; i < _net.params.ntargets; i++)
		free_mnist_dataset(data[i]);

	#ifdef LOGGING
		ret->num_correct = num_correct;
		return ret;
	#else
		return NULL;
	#endif

}

int save_testdata(struct testdata *data, char *filename) {
	if (!data) {
		printf("ERROR: Invalid testdata\n");
		return 1;
	}

	FILE *file = fopen(filename, "w");
	if (!file) {
		printf("ERROR: Failed to open testdata file (%s)\n", filename);
		return 2;
	}

	fwrite(&data->num_samples, sizeof(size_t), 1, file);
	fwrite(&_net.params.ntargets, sizeof(size_t), 1, file);
	fwrite(_net.params.targets, sizeof(size_t), _net.params.ntargets, file);
	fwrite(&data->num_correct, sizeof(size_t), 1, file);
	mat2file(data->confusion, file);
	for (int i = 0; i < _net.params.ntargets; i++)
		vec2file(data->costs[i], file);
	
	vec2file(data->labels, file);
	for (int i = 0; i < data->num_samples; i++)
		vec2file(data->outputs[i], file);
	
	fclose(file);
}

void free_testdata(struct testdata *data) {
	gsl_matrix_free(data->confusion);
	gsl_vector_free(data->labels);

	for (int i = 0; i < _net.params.ntargets; i++)
		gsl_vector_free(data->costs[i]);

	for (int i = 0; i < data->num_samples; i++)
		gsl_vector_free(data->outputs[i]);
	
	free(data->outputs);
	free(data->costs);
	free(data);
}

// private functions

double _calc_energy(gsl_vector **layers) {
	double ret = 0.0;
	for (int l = 1; l < _net.params.nlayers; l++) {
		gsl_vector *epsilon = gsl_vector_calloc(layers[l]->size);
		gsl_vector_memcpy(epsilon, layers[l]);

		gsl_vector *act = activation(layers[l-1], _net.params.act);
		gsl_blas_dgemv(CblasNoTrans, -1.0, _net.weights[l-1], act, 1.0, epsilon);

		double dot;
		gsl_blas_ddot(epsilon, epsilon, &dot);

		ret += dot;

		gsl_vector_free(epsilon);
		gsl_vector_free(act);
	}

	ret *= 0.5;
	return ret;
}

void _calc_lenergy(gsl_vector **layers, gsl_vector **lenergies, size_t iter) {
	for (int l = 1; l < _net.params.nlayers; l++) {
		gsl_vector *epsilon = gsl_vector_calloc(layers[l]->size);
		gsl_vector_memcpy(epsilon, layers[l]);

		gsl_vector *act = activation(layers[l-1], _net.params.act);
		gsl_blas_dgemv(CblasNoTrans, -1.0, _net.weights[l-1], act, 1.0, epsilon);

		double dot;
		gsl_blas_ddot(epsilon, epsilon, &dot);

		dot *= 0.5;

		// gsl_vector
		gsl_vector_set(lenergies[l-1], iter, dot);

		gsl_vector_free(epsilon);
		gsl_vector_free(act);
	}
}