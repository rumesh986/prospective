#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include "include/all.h"
#include "include/savefiles.h"

// local header

// struct _network {
// 	gsl_matrix **weights;
// 	// struct net_params params;
// };

struct relaxation {
	gsl_vector *energies;
	gsl_vector **lenergies;

	size_t iters;
};

// double _calc_energy(gsl_vector **layers);
// void _calc_lenergy(gsl_vector **layers, gsl_vector **lenergies, size_t iter);
// struct relaxation _relaxation(gsl_vector **layers, gsl_vector **epsilons, bool prediction, struct relaxation_params relax_params, bool logging);

// main code

void init_network(struct network *network) {
	size_t prev_len = db_get_input_length();
	printf("Initializing network\n");

	// add output layer to final dense block
	if (network->blocks[network->nblocks-1]->type == block_dense) {
		struct dense_block *block = network->blocks[network->nblocks-1]->block;
		// size_t *lengths = malloc(sizeof(size_t) * block->nlayers+1);
		block->nlayers++;
		block->lengths = realloc(block->lengths, sizeof(size_t) * block->nlayers);
		block->lengths[block->nlayers-1] = network->training->ntargets;
	}

	printf("%ld\n", prev_len);
	for (int block_i = 0; block_i < network->nblocks; block_i++) {
		if (network->blocks[block_i]->type == block_dense) {
			struct dense_block *block = network->blocks[block_i]->block;
			block->weights = malloc(sizeof(gsl_matrix *) * block->nlayers);
			block->deltaw = malloc(sizeof(gsl_matrix *) * block->nlayers);
			block->layers = malloc(sizeof(gsl_vector *) * block->nlayers);
			block->epsilons = malloc(sizeof(gsl_vector *) * block->nlayers);
			block->deltax = malloc(sizeof(gsl_vector *) * block->nlayers);
			for (int i = 0; i < block->nlayers; i++) {
				block->weights[i] = gsl_matrix_calloc(prev_len, block->lengths[i]);
				block->deltaw[i] = gsl_matrix_calloc(prev_len, block->lengths[i]);
				weight_init(block->weight_init, block->weights[i]);
				prev_len = block->lengths[i];
				printf("%ld\n", prev_len);

				block->layers[i] = gsl_vector_calloc(block->lengths[i]);
				block->epsilons[i] = gsl_vector_calloc(block->lengths[i]);
				block->deltax[i] = gsl_vector_calloc(block->lengths[i]);
			}
		}
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
	
	gsl_vector *train_sample = gsl_vector_calloc(db_get_input_length());
	// gsl_vector *input_cpy = input;

	// combine dense layers into contiguous block for easy working?
	// or propagate per dense block
	//		this is probably better in future when cnn blocks are added
	//		add layers, epsilons etc. to reduce number of alloc's

	struct traindata *ret;
	int target_counter = -1;
	if (logging) {
		ret = malloc(sizeof(struct traindata));
		// ret->delta_w_mags = malloc(sizeof(gsl_vector *) * )
	}

	for (int sample_i = 0; sample_i < num_samples; sample_i++) {
		int cur_target = sample_i % net->training->ntargets;

		if (cur_target == 0)
			target_counter++;
		
		gsl_vector_memcpy(train_sample, data_train[cur_target].images[target_counter]);
		gsl_vector *input = train_sample;
		// remember to clamp output, below line is too unreadable to safely use
		// gsl_vector_memcpy(net->blocks[net->nblocks-1]->block->layers[net->blocks[net->nblocks-1]->block->nlayers-1], data_train[cur_target].label_vec);
		
		_relaxation(net, input);

		// _propagate(net, input);


	}
	
	printf("Freeing datasets\n");
	for (int i = 0; i < train.ntargets; i++) {
		db_free_dataset(data_train[i]);
		db_free_dataset(data_test[i]);
	}
}

void _relaxation(struct network *net, gsl_vector *input) {
	// check stop condition every iteration
	bool stop = true;
	do {
		_propagate(net, input);
		_adjust_x(net);
		_adjust_w(net);
	} while(!stop);
}

void _propagate(struct network *net, gsl_vector *input) {
	for (int i = 0; i < net->nblocks; i++) {
		switch (net->blocks[i]->type) {
			case block_dense:	input = _propagate_dense(net, (struct dense_block *)net->blocks[i]->block, input);	break;
		}
	}
}

gsl_vector *_propagate_dense(struct network *net, struct dense_block *block, gsl_vector *input) {
	gsl_vector *act = activation(input, block->act);
	for (int l = 0; l < block->nlayers-1; l++) {
		gsl_vector_memcpy(block->epsilons[l+1], block->layers[l+1]);
		gsl_blas_dgemv(CblasNoTrans, -1.0, block->weights[l], act, 1.0, block->epsilons[l+1]);
		
	}
}