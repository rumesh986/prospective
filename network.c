#include <signal.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include "include/all.h"
#include "include/savefiles.h"

// local header

#define FLOOP(a, b) for (b = a.head; b; b=b->next)	// full loop, all layers
#define HLOOP(a, b) for (b = a.head->next; b->next; b=b->next) // hidden loop, only hidden layers
#define PLOOP(a, b) for (b = a.head->next; b; b=b->next) // partial loop, all layers except input layer
#define PLOOP2(a, b) for (b = a.head; b->next; b=b->next) // partial loop, all layers except output layer

struct traindata *_trainresults = NULL;

static struct network _net;
static struct relaxation_params _relax;
static int _train_i;
static bool _logging = false;

void _relaxation(bool training);
void _adjust_eps();
void _adjust_x(bool training);
void _adjust_w();
bool _check_stop(int iter, bool reset);
void _free_block(struct block *ablock);

// main code

void init_network(struct network *net) {
	printf("Initializing network...\n");

	// update input and output lengths
	net->head->layer->length = db_get_input_length();
	// net->tail->layer->length = ntargets;
	net->nlayers = 0;
	// for (struct block *cur_block = net.head; cur_block; cur_block=cur_block->next) {
	for (struct block *cur_block = net->head; cur_block; cur_block=cur_block->next) {
		if (cur_block->type == block_layer) {
			struct block_layer *layer = cur_block->layer;

			cur_block->layer = gsl_vector_calloc(layer->length);
			cur_block->act = gsl_vector_calloc(layer->length);
			cur_block->epsilon = gsl_vector_calloc(layer->length);
			cur_block->deltax = gsl_vector_calloc(layer->length);
			cur_block->energies = NULL;
			cur_block->epsilon2 = gsl_vector_calloc(layer->length);

			if (cur_block->next && cur_block->next->type == block_layer) {
				cur_block->weights = gsl_matrix_calloc(cur_block->next->layer->length, layer->length);
				cur_block->deltaw = gsl_matrix_calloc(cur_block->next->layer->length, layer->length);
				cur_block->out = gsl_vector_calloc(cur_block->next->layer->length);
				weight_init(cur_block->weights, net->weight_init);
			}
		}

		net->nlayers++;
	}
	net->lenergy_chunks = 0;

	printf("Completed initialization\n");
}

void set_network(struct network *net) {
	_net = *net;
}

void save_network(char *filename) {
	FILE *file = fopen(filename, "w");
	if (!file) {
		printf("[Error] Unable to open file to save network (%s)\n", filename);
		exit(ERR_FILE);
	}

	size_t data[3][3] = {
		{SAVE_TYPE, SAVE_SIZET, SAVE_NETWORK},
		{SAVE_NLAYERS, SAVE_SIZET, _net.nlayers},
		{SAVE_NTARGETS, SAVE_SIZET, _net.ntargets}
	};

	// printf("DOUBLECHECK: alpha = %f %p\n", data[3][2], &data[3][2]);
	save_data(SAVE_ARRAY, 0, &data, 0, 3, NULL, file);

	gsl_vector_ulong_view targets = gsl_vector_ulong_view_array(_net.targets, _net.ntargets);
	save_data(SAVE_TARGETS, SAVE_SIZET, &targets.vector, 1, 1, PS(1), file);

	fclose(file);
	// save_data(SAVE_WEIGHTS, SAVE_DOUBLET, _net.weights, 2, 1, PS(_net.params.nlayers-1), file);
}

struct traindata *train(struct training train, bool logging) {
	printf("Preparing to train...\n");

	_relax = train.relax;
	_logging = logging;

	size_t max_count = db_get_count(db_train);
	db_dataset data_train[_net.ntargets];
	db_dataset data_test[_net.ntargets];

	for (int i = 0; i < _net.ntargets; i++) {
		data_train[i] = db_get_dataset(db_train, _net.targets[i], train.proc);
		data_test[i] = db_get_dataset(db_test, _net.targets[i], train.proc);

		if (data_train[i].count < max_count)
			max_count = data_train[i].count;
	}

	size_t num_samples = train.num_samples;
	if (num_samples == 0)
		num_samples = max_count * _net.ntargets;
	
	struct traindata *ret;
	int target_counter = -1;
	if (_logging) {
		ret = malloc(sizeof(struct traindata));
		ret->iter_counts = malloc(sizeof(size_t) * num_samples);
		ret->train_costs = malloc(sizeof(double) * num_samples);
		ret->num_samples = num_samples;
		_trainresults = ret;

		// prepare lenergy and deltaw_mags storage location
		struct block *cblock;
		FLOOP(_net, cblock) {
			if (cblock->type == block_layer) {
				cblock->deltaw_mags = malloc(sizeof(double *) * num_samples);
				cblock->energies = malloc(sizeof(double *) * num_samples);
				for (int i = 0; i < num_samples; i++)
					cblock->energies[i] = NULL;
				
			}
		}
	}

	printf("Starting to train on %ld images\n", num_samples);
	for (_train_i = 0; _train_i < num_samples; _train_i++) {
		int cur_target = _train_i % _net.ntargets;

		if (cur_target == 0)
			target_counter++;
		
		gsl_vector_memcpy(_net.head->layer, data_train[cur_target].images[target_counter]);
		gsl_vector_set_basis(_net.tail->layer, cur_target);
	
		_relaxation(true);
		_adjust_w();

		_net.lenergy_chunks = 0;

	}

	printf("Completed training, freeing datasets\n");
	for (int i = 0; i < _net.ntargets; i++) {
		db_free_dataset(data_train[i]);
		db_free_dataset(data_test[i]);
	}

	_trainresults = NULL;
	_logging = false;
	return ret;
}

void save_traindata(struct traindata *data, char *filename) {
	gsl_vector ***lenergies = malloc(sizeof(gsl_vector **) * data->num_samples);
	// gsl_vector_view **delta_w_mags = malloc(sizeof(gsl_vector_view) * _net.nlayers-1);
	// gsl_vector *lenergies[data->num_samples][_net.nlayers-1];
	gsl_vector *delta_w_mags[_net.nlayers-1];
	// gsl_vector *iter_counts = gsl_vector_calloc(data->num_samples);
	// gsl_vector *train_costs = gsl_vector_calloc(data->num_samples);

	gsl_vector_view iter_counts = gsl_vector_view_array(data->iter_counts, data->num_samples);
	gsl_vector_view train_costs = gsl_vector_view_array(data->train_costs, data->num_samples);

	struct block *cblock;
	int l = 0;
	PLOOP2(_net, cblock) {
		gsl_vector_view view = gsl_vector_view_array(cblock->deltaw_mags, data->num_samples);
		delta_w_mags[l] = &view.vector;
		
		print_vec(delta_w_mags[l], "deltawmags", false);
		l++;
	}

	for (int i = 0; i < data->num_samples; i++) {
		lenergies[i] = malloc(sizeof(gsl_vector *) * _net.nlayers-1);
		l = 0;
		PLOOP(_net, cblock) {
			gsl_vector_view view = gsl_vector_view_array(cblock->energies[i], data->iter_counts[i]);
			lenergies[i][l] = &view.vector;
			l++;
		}
	}

	FILE *file = fopen(filename, "w");
	if (!file) {
		printf("[Error] Failed to open traindata file (%s)\n", filename);
		exit(ERR_FILE);
	}

	save_data(SAVE_TYPE, size_dt, PS(SAVE_TRAIN), 0, 1, NULL, file);
	save_data(SAVE_DELTAW_MAGS, double_dt, &delta_w_mags, 1, 1, PS(_net.nlayers-1), file);
	save_data(SAVE_ITER_COUNTS, double_dt, &iter_counts, 1, 1, PS(1), file);
	save_data(SAVE_LENERGIES, double_dt, lenergies, 1, 2, (size_t[]){data->num_samples, _net.nlayers-1}, file);
	save_data(SAVE_COSTS, double_dt, &train_costs, 1, 1, PS(1), file);

	fclose(file);
}

void free_network(struct network *net) {
	free(net->targets);
	struct block *cblock = net->head;
	while (cblock->next) {
		cblock = cblock->next;
		_free_block(cblock->prev);
	}
	_free_block(cblock);

	free(net);
}

void _relaxation(bool training) {
	printf("Starting relaxation\n");

	// reset layers, epsilons and deltax hidden layers
	struct block *cur_block;
	HLOOP(_net, cur_block) {
		if (cur_block->type == block_layer) {
			gsl_vector_set_zero(cur_block->layer);
			gsl_vector_set_zero(cur_block->epsilon);
			gsl_vector_set_zero(cur_block->deltax);
		}
	}

	// check stop condition every iteration
	bool stop = false;
	int iter = 0;
	do {
		_adjust_eps();
		_adjust_x(true);
		stop = _check_stop(iter, iter == 0);
		iter++;
	} while(!stop);

	_trainresults->iter_counts[_train_i] = iter;
	printf("Relaxation complete after %d iterations\n", iter);
}

void _prop_layer(struct block *ablock) {
	if (ablock->type == block_layer) {
		struct block_layer *layer = ablock->layer;

		activation_inplace(ablock->layer, ablock->act, _net.act);
		gsl_blas_dgemv(CblasNoTrans, 1.0, ablock->weights, ablock->act, 0.0, ablock->out);
	} else if (ablock->type == block_cnn) {
		for (int i = 0; i < ablock->out->size; i++) {
			struct block_cnn *cnn = ablock->cnn;
			struct db_image_info img_shape = db_get_image_info();
			cnn->layer_mat = gsl_matrix_view_vector(ablock->layer, img_shape.size1, img_shape.size2);
			gsl_matrix_view mat_view = gsl_matrix_submatrix(&cnn->layer_mat.matrix, i / img_shape.size2, i % img_shape.size2 , cnn->kernel_size, cnn->kernel_size);

			gsl_vector_set(ablock->out, i, mat_dot(ablock->weights, &mat_view.matrix));
		}
	}
}

// void _adjust_eps_layer(struct block *ablock) {
// 	gsl_vector_memcpy(ablock->epsilon, ablock->layer);
// 	_prop_layer(ablock->prev);
// 	gsl_vector_sub(ablock->epsilon, ablock->prev->out);
// }

// void _adjust_eps_cnn(struct block *ablock) {
// 	struct block_cnn *cnn = ablock->cnn;
// 	gsl_vector_memcpy(cnn->epsilon, cnn->layer);
// 	_prop_layer(ablock->prev);
// 	gsl_vector_sub(cnn->epsilon, ablock->prev);
// }

void _adjust_eps() {
	struct block *cblock;
	PLOOP(_net, cblock) {
		gsl_vector_memcpy(cblock->epsilon, cblock->layer);
		_prop_layer(cblock->prev);
		gsl_vector_sub(cblock->epsilon, cblock->prev->out);

		// switch (cblock->type) {
		// 	case block_layer:	_adjust_eps_layer(cblock);	break;
		// 	case block_cnn: 	_adjust_eps_cnn(cblock);	break;
		// }
	}
}

void _adjust_x_layer(struct block *ablock) {
	struct block_layer *layer = ablock->layer;
	activation_deriv_inplace(ablock->layer, ablock->act, _net.act);
	gsl_blas_dgemv(CblasTrans, _relax.gamma, ablock->weights, ablock->next->epsilon, 0.0, ablock->deltax);
	gsl_vector_mul(ablock->deltax, ablock->act);

	gsl_blas_daxpy(-_relax.gamma, ablock->epsilon, ablock->deltax);
	gsl_vector_add(ablock->layer, ablock->deltax);
}

void _adjust_x(bool training) {
	struct block *cblock;
	// for (struct block *cblock = _net.head->next; training ? cblock->next : cblock; cblock=cblock->next) {
	HLOOP(_net, cblock) {
		switch (cblock->type) {
			case block_layer:	_adjust_x_layer(cblock);	break;
			case block_cnn:		_adjust_x_cnn(cblock);		break;
		}
	}

	if (!training)
		gsl_blas_daxpy(-_relax.gamma, _net.tail->epsilon, _net.tail->layer);
}

void _adjust_w_layer(struct block *ablock) {
	activation_inplace(ablock->layer, ablock->act, _net.act);

	gsl_blas_dger(_net.alpha, ablock->next->epsilon, ablock->act, ablock->deltaw);
	gsl_matrix_add(ablock->weights, ablock->deltaw);

	if (_logging)
		ablock->deltaw_mags[_train_i] = frobenius_norm(ablock->deltaw);

	gsl_matrix_set_zero(ablock->deltaw);
}

void _adjust_w_cnn(struct block *ablock) {

}

void _adjust_w() {
	struct block *cblock;
	PLOOP2(_net, cblock) {
		switch (cblock->type) {
			case block_layer:	_adjust_w_layer(cblock);	break;
			case block_cnn:		_adjust_w_cnn(cblock);		break;
		}
	}
}

double _calc_energies(int iter, bool resize) {
	double ret = 0.0;

	struct block *cblock;
	PLOOP(_net, cblock) {
		if (cblock->type == block_layer) {
			struct block_layer *layer = cblock->layer;

			if (resize) {
				double *new_ptr = realloc(cblock->energies[_train_i], sizeof(double) * _net.lenergy_chunks * CHUNK_SIZE);
				if (new_ptr) {
					cblock->energies[_train_i] = new_ptr;
				} else {
					printf("[Error] Failed to realloc lenergy data, expect errors\n");
					return -1.0;
				}
			}

			gsl_vector_memcpy(cblock->epsilon2, cblock->layer);
			_prop_layer(cblock->prev);
			gsl_vector_sub(cblock->epsilon2, cblock->prev->out);

			double dot;
			gsl_blas_ddot(cblock->epsilon2, cblock->epsilon2, &dot);
			dot *= 0.5;

			if (_logging)
				cblock->energies[_train_i][iter] = dot;
			ret += dot;
		}
	}

	return ret;
}

void _print_lenergies(struct network net, int iter) {
	printf("lenergies at iter %d\n", iter);
	struct block *cur_block = net.head->next;
	while (cur_block) {
	if (cur_block->type == block_layer) {
			printf("\t%.40f\n", cur_block->energies[_train_i][iter]);
		}
		cur_block = cur_block->next;
	}
}

bool _check_stop(int iter, bool reset) {
	if (_relax.max_iters && iter > _relax.max_iters)
		return true;
	
	bool resize = false;
	if (_logging && (iter + 1) >= (_net.lenergy_chunks * CHUNK_SIZE)) {
		_net.lenergy_chunks++;
		resize = true;
	}
	
	static double prev_energy = 100000000.0;
	if (reset)
		prev_energy = 10000000000.0;

	double energy = _calc_energies(iter, resize);
	double res = prev_energy - energy;
	printf("[%3d] energy: %e, res = %e\n", iter, energy, res);
	if (_relax.energy_res &&  res < _relax.energy_res) {
		if (res < 0) {
			printf("prev: %.60f\n", prev_energy);
			printf("curr: %.60f\n", energy);
			_print_lenergies(_net, iter-1);
			_print_lenergies(_net, iter);		
		}
		return true;

	}
	
	prev_energy = energy;
}

void _free_block(struct block *ablock) {
	gsl_vector_free(ablock->layer);
	gsl_vector_free(ablock->act);
	gsl_vector_free(ablock->epsilon);
	gsl_vector_free(ablock->deltax);
	gsl_vector_free(ablock->epsilon2);

	if (ablock->next) {
		gsl_vector_free(ablock->out);
		gsl_matrix_free(ablock->weights);
		gsl_matrix_free(ablock->deltaw);
	}

	for (int i = 0; i < _train_i; i++)
		free(ablock->energies[i]);
	
	free(ablock->energies);

	if (ablock->type == block_layer) {
		free(ablock->layer);
	}

	free(ablock);
}