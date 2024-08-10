#include <signal.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include "include/all.h"
#include "include/savefiles.h"

// local header

// struct _network {
// 	gsl_matrix **weights;
// 	// struct net_params params;
// };

// struct relaxation {
// 	gsl_vector *energies;
// 	gsl_vector **lenergies;

// 	size_t iters;
// };

struct traindata *trainresults = NULL;



// double _calc_energy(gsl_vector **layers);
// void _calc_lenergy(gsl_vector **layers, gsl_vector **lenergies, size_t iter);
// struct relaxation _relaxation(gsl_vector **layers, gsl_vector **epsilons, bool prediction, struct relaxation_params relax_params, bool logging);
void _relaxation(struct network *net, int index, bool training);
void _adjust_eps(struct network *net);
void _adjust_x(struct network *net, bool training);
void _adjust_w(struct network *net);
bool _check_stop(struct network *net, int iter);

// main code

void init_network(struct network *network) {
	printf("Initializing network\n");

	// add new block to hold input layer and initialize it
	size_t * inp_lengths = malloc(sizeof(size_t));
	inp_lengths[0] = db_get_input_length();
	struct block *inp_block = new_dense_block(1, inp_lengths);
	inp_block->next = network->head;
	network->head->prev = inp_block;
	network->head = inp_block;

	inp_block->dense->weights = malloc(sizeof(gsl_matrix *) * 3);
	for (int i = 0; i < 3; i++)
		inp_block->dense->weights[i] = NULL;
	inp_block->dense->deltaw = NULL;

	inp_block->dense->layers = malloc(sizeof(gsl_vector *) * 3);
	inp_block->dense->layers[0] = NULL;
	inp_block->dense->layers[1] = gsl_vector_calloc(inp_block->dense->lengths[0]);
	inp_block->dense->epsilons = malloc(sizeof(gsl_vector *) * 3);
	inp_block->dense->epsilons[0] = NULL;
	inp_block->dense->epsilons[1] = gsl_vector_calloc(inp_block->dense->lengths[0]);
	inp_block->dense->deltax= NULL;


	// add new block to hold output layer
	size_t *out_lengths = malloc(sizeof(size_t));
	out_lengths[0] = network->training->ntargets;
	struct block *out_block = new_dense_block(1, out_lengths);
	// set weight init to xavier uniform, change in future to config option
	out_block->dense->weight_init = weights_xavier_uniform;
	struct block *cur_final = network->head;
	while (cur_final->next)
		cur_final = cur_final->next;

	out_block->prev = cur_final;
	cur_final->next = out_block;

	out_block->dense->act = cur_final->dense->act;


	// loop through hidden layers
	// for dense blocks, hidden layers run from index 1 -> nlayers
	// index 0 refers to last element of previous block
	// index nlayers+1 refers to first element of next block
	// this is to simplify later loops to cover the edge (literal) cases
	struct block *cur_block = network->head->next;
	size_t prev_len = network->head->dense->lengths[0];
	int block_cnt = 0;
	int layer = 1;
	while (cur_block) {
		// printf("initing block %d\n", block_cnt);
		if (cur_block->type == block_dense) {

			struct dense_block *dblock = cur_block->dense;
			dblock->weights = malloc(sizeof(gsl_matrix *) * (dblock->nlayers+2));
			dblock->deltaw = malloc(sizeof(gsl_matrix *) * dblock->nlayers);

			dblock->layers = malloc(sizeof(gsl_vector *) * (dblock->nlayers+2));
			dblock->epsilons = malloc(sizeof(gsl_vector *) * (dblock->nlayers+2));
			dblock->deltax = malloc(sizeof(gsl_vector *) * dblock->nlayers);

			dblock->global_pos = layer;

			for (int l = 1; l < dblock->nlayers+1; l++) {
				dblock->weights[l] = gsl_matrix_calloc(dblock->lengths[l-1], prev_len);

				dblock->layers[l] = gsl_vector_calloc(dblock->lengths[l-1]);
				dblock->epsilons[l] = gsl_vector_calloc(dblock->lengths[l-1]);

				// these do not connect to other blocks
				dblock->deltaw[l-1] = gsl_matrix_calloc(dblock->lengths[l-1], prev_len);
				dblock->deltax[l-1] = gsl_vector_calloc(dblock->lengths[l-1]);

				weight_init(dblock->weights[l], dblock->weight_init);
				prev_len = dblock->lengths[l-1];
				layer++;
			}

			if (cur_block->prev->type == block_dense) {
				struct dense_block *prev = cur_block->prev->dense;
				dblock->layers[0] = prev->layers[prev->nlayers];
				dblock->epsilons[0] = prev->epsilons[prev->nlayers];
				dblock->weights[0] = prev->weights[prev->nlayers];

				prev->layers[prev->nlayers+1] = dblock->layers[1];
				prev->epsilons[prev->nlayers+1] = dblock->epsilons[1];

				prev->weights[prev->nlayers+1] = dblock->weights[1];
			}

			dblock->lenergy_nchunks = 1;
			dblock->lenergies = malloc(sizeof(double *) * CHUNK_SIZE * dblock->nlayers * dblock->lenergy_nchunks);
		}

		cur_block = cur_block->next;
		block_cnt++;
	}
}

struct traindata *train(struct network *net, bool logging) {
	printf("Preparing to train...\n");
	struct training train = *net->training;

	size_t max_count = db_get_count(db_train);
	db_dataset data_train[train.ntargets];
	db_dataset data_test[train.ntargets];

	for (int i = 0; i < net->training->ntargets; i++) {
		data_train[i] = db_get_dataset(db_train, train.targets[i], train.params.proc);
		data_test[i] = db_get_dataset(db_test, train.targets[i], train.params.proc);

		if (data_train[i].count < max_count)
			max_count = data_train[i].count;
	}

	size_t num_samples = train.params.num_samples;
	if (num_samples == 0)
		num_samples = max_count * train.ntargets;
	
	// gsl_vector *train_sample = gsl_vector_calloc(db_get_input_length());
	// gsl_vector *input_cpy = input;

	// combine dense layers into contiguous block for easy working?
	// or propagate per dense block
	//		this is probably better in future when cnn blocks are added
	//		add layers, epsilons etc. to reduce number of alloc's
	// or change network to linked list
	// start function at start of net, it can propagate itself through the list without a for loop

	struct traindata *ret;
	int target_counter = -1;
	if (logging) {
		ret = malloc(sizeof(struct traindata));
		ret->lenergies = malloc(sizeof(gsl_vector **) * num_samples);
		// ret->lenergiesd = malloc(sizeof (double **) * num_samples);
		// ret->delta_w_mags = malloc(sizeof(gsl_vector *) * )
		ret->num_samples = num_samples;
		trainresults = ret;
	}

	for (int sample_i = 0; sample_i < num_samples; sample_i++) {
		int cur_target = sample_i % net->training->ntargets;

		if (cur_target == 0)
			target_counter++;
		
		gsl_vector_memcpy(net->head->dense->layers[1], data_train[cur_target].images[target_counter]);
		// gsl_vector *input = train_sample;
		// remember to clamp output, below line is too unreadable to safely use
		// gsl_vector_memcpy(net->blocks[net->nblocks-1]->block->layers[net->blocks[net->nblocks-1]->block->nlayers-1], data_train[cur_target].label_vec);
		
		_relaxation(net, sample_i, true);
		printf("Finished relaxing, starting w adjusts\n");
		_adjust_w(net);

		// _propagate(net, input);


	}
	
	printf("Freeing datasets\n");
	for (int i = 0; i < train.ntargets; i++) {
		db_free_dataset(data_train[i]);
		db_free_dataset(data_test[i]);
	}

	trainresults = NULL;
	return ret;
}

void save_traindata(struct network *net, struct traindata *data, char *filename) {
	gsl_vector ***lenergies = malloc(sizeof(gsl_vector **) * data->num_samples);

}

void _relaxation(struct network *net, int index, bool training) {
	printf("Starting relaxation\n");
	// reset layers, epsilons and deltax (layers 1 - n-1)
	struct block *cur_block = net->head->next;
	size_t nlayers = 0;
	while (cur_block->next) {
		if (cur_block->type == block_dense) {
			struct dense_block *dblock = cur_block->dense;
			for (int l = 1; l < dblock->nlayers+1; l++) {
				gsl_vector_set_zero(dblock->layers[l]);
				gsl_vector_set_zero(dblock->epsilons[l]);
				gsl_vector_set_zero(dblock->deltax[l-1]);
			}
			
			nlayers += dblock->nlayers;
			gsl_vector_set_zero(dblock->epsilons[dblock->nlayers]);
		}
		cur_block = cur_block->next;
	}

	// check stop condition every iteration
	bool stop = false;
	int iter = 0;
	do {
		// printf("Relaxation iteration %d\n", iter);
		_adjust_eps(net);
		_adjust_x(net, true);
		stop = _check_stop(net, iter);
		iter++;
	} while(!stop);

	printf("Relaxation complete after %d iterations\n", iter);
}

void _adjust_eps_dense(struct block *ablock) {
	struct dense_block *block = ablock->dense;

	// prepare epsilons
	// for (int l = 0; l < block->nlayers-1; l++)

	// if (ablock->next && ablock->next->type == block_dense)
	// 	gsl_vector_memcpy(block->epsilons[block->nlayers], ablock->next->dense->layers[0]);


	// gsl_vector *act = activation(input, block->act);
	// gsl_blas_dgemv(CblasNoTrans, -1.0, block->weights[0], act, 1.0, block->epsilons[0]);
	// gsl_vector_free(act);
	for (int l = 1; l < block->nlayers+1; l++) {
		gsl_vector *act = activation(block->layers[l-1], block->act);
		gsl_vector_memcpy(block->epsilons[l], block->layers[l]);
		gsl_blas_dgemv(CblasNoTrans, -1.0, block->weights[l], act, 1.0, block->epsilons[l]);
		gsl_vector_free(act);
	}

	// return block->layers[block->nlayers-1];
}

void _adjust_eps(struct network *net) {
	struct block *cur_block = net->head->next;
	// gsl_vector *prev_layer = net->head->dense->layers[0];
	// print_img(prev_layer, "input layer");
	int current_block = 0;
	while (cur_block->next) {
		// printf("propagating through block %d\n", current_block);
		switch (cur_block->type) {
			case block_dense:	_adjust_eps_dense(cur_block);	break;
		}
		cur_block = cur_block->next;
		current_block++;
	}
}

void _adjust_x_dense(struct block *ablock, double gamma) {
	struct dense_block *block = ablock->dense;

	// printf("%ld layers in block\n", block->nlayers);
	for (int l = 1; l < block->nlayers+1; l++) {
		gsl_vector *act_deriv = activation_deriv(block->layers[l], block->act);

		gsl_blas_dgemv(CblasTrans, gamma, block->weights[l+1], block->epsilons[l+1], 0.0, block->deltax[l-1]);
		gsl_vector_mul(block->deltax[l-1], act_deriv);
		gsl_blas_daxpy(-gamma, block->epsilons[l], block->deltax[l-1]);

		gsl_vector_add(block->layers[l], block->deltax[l-1]);
		gsl_vector_free(act_deriv);
	}
}

void _adjust_x(struct network *net, bool training) {
	struct block *cur_block = net->head->next;
	int current_block = 0;
	while (training ? cur_block->next : cur_block) {
		// printf("adjust layers in block %d\n", current_block);
		switch (cur_block->type) {
			case block_dense:	_adjust_x_dense(cur_block, net->training->params.relax->gamma);	break;
		}
		cur_block = cur_block->next;
		current_block++;
	}
}

void _adjust_w_dense(struct block *ablock) {
	struct dense_block *block = ablock->dense;

	for (int l = 1; l < block->nlayers+1; l++) {
		gsl_vector *act = activation(block->layers[l-1], block->act);

		gsl_blas_dger(block->alpha, block->epsilons[l], act, block->deltaw[l-1]);
		gsl_matrix_add(block->weights[l], block->deltaw[l-1]);

		gsl_matrix_set_zero(block->deltaw[l-1]);
		gsl_vector_free(act);
	}
}

void _adjust_w(struct network *net) {
	struct block *cur_block = net->head->next;
	while (cur_block->next) {
		switch (cur_block->type) {
			case block_dense:	_adjust_w_dense(cur_block);	break;
		}
		cur_block = cur_block->next;
	}
}

double _calc_energies(struct network *net, int iter) {
	struct block *cur_block = net->head->next;
	gsl_vector *prev_layer = net->head->dense->layers[1];

	double ret = 0.0;
	while (cur_block) {
		if (cur_block->type == block_dense) {
			struct dense_block *dblock = cur_block->dense;

			if (iter >= (dblock->lenergy_nchunks * CHUNK_SIZE)-1) {
				dblock->lenergy_nchunks++;
				double **new_ptr = realloc(dblock->lenergies, sizeof(double *) * dblock->lenergy_nchunks * CHUNK_SIZE * dblock->nlayers);
				if (new_ptr) {
					dblock->lenergies = new_ptr;
				} else {
					printf("[ERROR] Failed to realloc lenergy data, expect errors\n");
					return -1.0;
				}
			}

			// dblock->lenergies[iter] = malloc(sizeof(double) * dblock->nlayers);
			for (int l = 1; l < dblock->nlayers+1; l++) {
				gsl_vector *epsilon = gsl_vector_calloc(dblock->layers[l]->size);
				gsl_vector_memcpy(epsilon, dblock->layers[l]);

				gsl_vector *act = activation(prev_layer, dblock->act);
				gsl_blas_dgemv(CblasNoTrans, -1.0, dblock->weights[l], act, 1.0, epsilon);

				double dot;
				gsl_blas_ddot(epsilon, epsilon, &dot);
				dot *= 0.5;
				dblock->lenergies[(iter * dblock->nlayers) + (l-1)] = dot;

				ret += dot;

				prev_layer = dblock->layers[l];
				gsl_vector_free(epsilon);
				gsl_vector_free(act);
			}
		}

		cur_block = cur_block->next;
	}

	return ret;
}

void _print_lenergies(struct network *net, int iter) {
	printf("lenergies at iter %d\n", iter);
	struct block *cur_block = net->head->next;
	while (cur_block) {
		if (cur_block->type == block_dense) {
			struct dense_block *dblock = cur_block->dense;

			double sum = 0;
			printf("\t");
			for (int l = 0; l < dblock->nlayers; l++) {
				printf("%.50f ", dblock->lenergies[(iter *dblock->nlayers) + l]);
				sum += dblock->lenergies[iter * dblock->nlayers + l];
			}
			printf("\t sum = %.50f\n", sum);
			printf("\n");
		}
		cur_block = cur_block->next;
	}
}

bool _check_stop(struct network *net, int iter) {
	if (net->training->params.relax->max_iters && iter > net->training->params.relax->max_iters)
		return true;
	
	static double prev_energy = 100000000.0;
	double energy = _calc_energies(net, iter);
	double res = prev_energy - energy;
	printf("[%3d] energy: %e, res = %e\n", iter, energy, res);
	if (net->training->params.relax->energy_res &&  res < net->training->params.relax->energy_res) {
		if (res < 0) {
			printf("prev: %.60f\n", prev_energy);
			printf("curr: %.60f\n", energy);
			_print_lenergies(net, iter-1);
			_print_lenergies(net, iter);		
		}
		return true;

	}
	
	prev_energy = energy;
}