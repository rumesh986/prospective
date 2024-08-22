#include <signal.h>
#include <math.h>

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
static int _sample_i;
static bool _logging = false;

void _prop_layer(struct block *ablock, bool temp_input, bool temp_output);
int _relaxation(bool training);
void _adjust_eps();
void _adjust_w_layer(struct block *ablock);
void _adjust_w_cnn(struct block *ablock);
void _adjust_x_layer(struct block *ablock, int iter);
void _adjust_x_cnn(struct block *ablock, int iter);
void _free_block(struct block *ablock);
double _calc_energies(int iter);

// main code

void init_network(struct network *net) {
	printf("Initializing network...\n");

	// update input length
	net->head->length = db_get_input_length();

	net->nlayers = 0;
	// for (struct block *cur_block = net->head; cur_block; cur_block=cur_block->next) {
	struct block *cur_block;
	FLOOP((*net), cur_block) {
		if (cur_block->type == block_layer) {
			if (cur_block->prev) {
				cur_block->weights = gsl_matrix_calloc(cur_block->length, cur_block->prev->length);
				cur_block->deltaw = gsl_matrix_calloc(cur_block->length, cur_block->prev->length);
				weight_init(cur_block->weights, net->weight_init);
			} else {
				cur_block->weights = NULL;
				cur_block->deltaw = NULL;
			}
		} else if (cur_block->type == block_cnn) {
			size_t prev_img_size = sqrt(cur_block->prev->length);
			size_t prev_channels = cur_block->prev->type == block_cnn ? cur_block->prev->cnn->nchannels : 1;

			cur_block->cnn->padded_input = malloc(sizeof(gsl_matrix *) * prev_channels);
			for (int i = 0; i < prev_channels; i++)
				cur_block->cnn->padded_input[i] = gsl_matrix_calloc(prev_img_size + 2*cur_block->cnn->padding, prev_img_size + 2*cur_block->cnn->padding);
			
			cur_block->cnn->conv_size = ((double)(prev_img_size + (2*cur_block->cnn->padding) - (cur_block->cnn->kernel_size))/((double)cur_block->cnn->stride)) + 1;
			cur_block->cnn->conv_length = cur_block->cnn->conv_size * cur_block->cnn->conv_size;
			cur_block->cnn->conv_layer = gsl_vector_calloc(cur_block->cnn->conv_length);

			cur_block->cnn->image_size = ((double)cur_block->cnn->conv_size/(double)cur_block->cnn->pool_size);
			cur_block->cnn->image_length = cur_block->cnn->image_size * cur_block->cnn->image_size;
			cur_block->length = cur_block->cnn->image_length * cur_block->cnn->nchannels;

			cur_block->cnn->pool_indices = gsl_vector_calloc(cur_block->length);

			// cur_block->cnn->dAdx = malloc(sizeof(gsl_matrix *) * cur_block->cnn->nchannels);
			cur_block->cnn->dAdx = gsl_matrix_alloc(cur_block->length, cur_block->prev->length);
			cur_block->cnn->nmats = cur_block->cnn->nchannels * (cur_block->prev->type == block_cnn ? cur_block->prev->cnn->nchannels : 1);

			cur_block->deltaw = gsl_matrix_calloc(cur_block->cnn->kernel_size, cur_block->cnn->kernel_size * cur_block->cnn->nmats);
			cur_block->weights = gsl_matrix_calloc(cur_block->cnn->kernel_size, cur_block->cnn->kernel_size * cur_block->cnn->nmats);
			weight_init(cur_block->weights, net->weight_init);

			cur_block->cnn->dAdw = malloc(sizeof(gsl_matrix *) * cur_block->cnn->nmats);
			for (int c = 0; c < cur_block->cnn->nmats; c++)
				cur_block->cnn->dAdw[c] = gsl_matrix_calloc(cur_block->cnn->kernel_size, cur_block->cnn->kernel_size * cur_block->cnn->conv_length);

			cur_block->cnn->dAdxP = malloc(sizeof(gsl_matrix *) * cur_block->cnn->nchannels);
			for (int c = 0; c < cur_block->cnn->nchannels; c++) {
				cur_block->cnn->dAdxP[c] = gsl_matrix_calloc(cur_block->cnn->conv_length, cur_block->prev->length);
				// cur_block->cnn->dAdx[c] = gsl_matrix_calloc(cur_block->length, cur_block->prev->length);
			}
		}

		cur_block->layer = gsl_vector_calloc(cur_block->length);
		cur_block->epsilon = gsl_vector_calloc(cur_block->length);
		cur_block->deltax = gsl_vector_calloc(cur_block->length);
		cur_block->tlayer = gsl_vector_calloc(cur_block->length);
		cur_block->tepsilon = gsl_vector_calloc(cur_block->length);

		cur_block->energies = NULL;
		cur_block->deltaw_mags = NULL;
		cur_block->deltax_mags = NULL;

		net->nlayers++;
	}
	net->lenergy_chunks = 0;

	printf("Completed initialization\n");
}

void set_network(struct network *net) {
	printf("\n\nChanging net to %p\n\n", net);
	_net = *net;
}

void save_network(char *filename) {
	FILE *file = fopen(filename, "w");
	if (!file) {
		printf("[Error] Unable to open file to save network (%s)\n", filename);
		exit(ERR_FILE);
	}

	size_t data[3][3] = {
		{SAVE_TYPE, size_dt, SAVE_NETWORK},
		{SAVE_NLAYERS, size_dt, _net.nlayers},
		{SAVE_NTARGETS, size_dt, _net.ntargets}
	};

	save_data(SAVE_ARRAY, 0, &data, 0, 3, NULL, file);

	gsl_vector_ulong_view targets = gsl_vector_ulong_view_array(_net.targets, _net.ntargets);
	save_data(SAVE_TARGETS, size_dt, &targets.vector, 1, 1, PS(1), file);

	fclose(file);
	// save_data(SAVE_WEIGHTS, SAVE_DOUBLET, _net.weights, 2, 1, PS(_net.params.nlayers-1), file);
}

struct traindata *train(struct training train, bool logging) {
	printf("Preparing to train...\n");

	clear_block_data();

	_relax = train.relax;
	_logging = logging;

	size_t max_count = db_get_count(db_train);
	db_dataset data_train[_net.ntargets];
	db_dataset data_test[_net.ntargets];

	for (int i = 0; i < _net.ntargets; i++) {
		data_train[i] = db_get_dataset(db_train, _net.targets[i], _net.proc);
		data_test[i] = db_get_dataset(db_test, _net.targets[i], _net.proc);

		if (data_train[i].count < max_count)
			max_count = data_train[i].count;
	}

	size_t num_samples = train.num_samples;
	if (num_samples == 0)
		num_samples = max_count * _net.ntargets;
	
	struct traindata *ret = NULL;
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
			cblock->deltaw_mags = malloc(sizeof(double) * num_samples);
			cblock->energies = malloc(sizeof(double *) * num_samples);
			cblock->deltax_mags = malloc(sizeof(double *) * num_samples);
			cblock->nenergies = num_samples;
			for (int i = 0; i < num_samples; i++) {
				cblock->energies[i] = NULL;
				cblock->deltax_mags[i] = NULL;
			}
		}
	}
	gsl_vector *test_label_vec;
	gsl_vector *test_cost_vec;

	if (train.test_samples_per_iters != 0) {
		test_label_vec = gsl_vector_calloc(_net.ntargets);
		test_cost_vec = gsl_vector_calloc(_net.ntargets);
	}

	printf("Starting to train on %ld images\n", num_samples);
	for (_sample_i = 0; _sample_i < num_samples; _sample_i++) {
		int cur_target = _sample_i % _net.ntargets;

		if (cur_target == 0)
			target_counter++;
		
		gsl_vector_memcpy(_net.head->layer, data_train[cur_target].images[target_counter]);
		gsl_vector_set_basis(_net.tail->layer, cur_target);

		// clear dAdw data
		struct block *cblock;
		HLOOP(_net, cblock) {
			if (cblock->type == block_cnn) {
				for (int c = 0; c < cblock->cnn->nmats; c++) {
					for (int i = 0; i < cblock->length; i++)
						gsl_matrix_set_zero(cblock->cnn->dAdw[c]);
				}
			}
		}
	
		int iters = _relaxation(true);
		
		// update weights
		PLOOP(_net, cblock) {
			switch (cblock->type) {
				case block_layer:	_adjust_w_layer(cblock);	break;
				case block_cnn:		_adjust_w_cnn(cblock);		break;
			}
		}

		// printf("[%d] deltaw_mags: ", _sample_i);
		// PLOOP(_net, cblock) {
		// 	printf("%.15f ", cblock->deltaw_mags[_sample_i]);
		// }
		// printf("\n");
		
		if (train.test_samples_per_iters != 0) {
			double train_cost = 0.0;
			struct block *cblock;
			
			for (int test_i = 0; test_i < train.test_samples_per_iters; test_i++) {
				gsl_vector_memcpy(_net.head->tlayer, data_test[test_i % _net.ntargets].images[test_i]);
				gsl_vector_set_zero(_net.tail->tlayer);
				gsl_vector_set_basis(test_label_vec, test_i % _net.ntargets);

				PLOOP(_net, cblock) 
					_prop_layer(cblock, true, true);

				gsl_vector_memcpy(test_cost_vec, _net.tail->tlayer);
				gsl_vector_sub(test_cost_vec, test_label_vec);

				train_cost += gsl_blas_dnrm2(test_cost_vec);
				gsl_vector_set_zero(test_cost_vec);
			}

			train_cost /= (double)train.test_samples_per_iters;
			// printf("[%5d] Normalized cost during training: %f\n", _sample_i, train_cost);
			ret->train_costs[_sample_i] = train_cost;
		}

		if (logging)
			ret->iter_counts[_sample_i] = iters;

		_net.lenergy_chunks = 0;

	}

	printf("Completed training, freeing datasets\n");
	for (int i = 0; i < _net.ntargets; i++) {
		db_free_dataset(data_train[i]);
		db_free_dataset(data_test[i]);
	}

	if (train.test_samples_per_iters) {
		gsl_vector_free(test_label_vec);
		gsl_vector_free(test_cost_vec);
	}

	_trainresults = NULL;
	_logging = false;
	return ret;
}

void save_traindata(struct traindata *data, char *filename) {
	if (!data) {
		printf("[Error] Invalid traindata\n");
		return;
	}

	gsl_vector ***lenergies = malloc(sizeof(gsl_vector **) * data->num_samples);
	gsl_vector ***deltax_mags = malloc(sizeof(gsl_vector **) * data->num_samples);
	gsl_vector *delta_w_mags[_net.nlayers-1];

	gsl_vector_view iter_counts = gsl_vector_view_array(data->iter_counts, data->num_samples);
	gsl_vector_view train_costs = gsl_vector_view_array(data->train_costs, data->num_samples);

	gsl_vector_view deltaw_views[_net.nlayers-1];
	gsl_vector_view lenergy_views[data->num_samples * (_net.nlayers-1)];
	gsl_vector_view deltax_views[data->num_samples * (_net.nlayers-2)];
	struct block *cblock;

	int l = 0;
	PLOOP(_net, cblock) {
		deltaw_views[l] = gsl_vector_view_array(cblock->deltaw_mags, data->num_samples);
		delta_w_mags[l] = &deltaw_views[l].vector;
		
		l++;
	}

	int lenergy_view_counter = 0;
	int deltax_view_counter = 0;
	for (int i = 0; i < data->num_samples; i++) {
		lenergies[i] = malloc(sizeof(gsl_vector *) * (_net.nlayers-1));
		deltax_mags[i] = malloc(sizeof(gsl_vector *) * (_net.nlayers-2));
		l = 0;
		PLOOP(_net, cblock) {
			lenergy_views[lenergy_view_counter] = gsl_vector_view_array(cblock->energies[i], data->iter_counts[i]);
			lenergies[i][l] = &lenergy_views[lenergy_view_counter].vector;
			lenergy_view_counter++;
			l++;
		}

		l = 0;
		HLOOP(_net, cblock) {
			deltax_views[deltax_view_counter] = gsl_vector_view_array(cblock->deltax_mags[i], data->iter_counts[i]);
			deltax_mags[i][l] = &deltax_views[deltax_view_counter].vector;
			deltax_view_counter++;
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
	save_data(SAVE_ITER_COUNTS, double_dt, &iter_counts.vector, 1, 1, PS(1), file);
	save_data(SAVE_LENERGIES, double_dt, lenergies, 1, 2, (size_t[]){data->num_samples, _net.nlayers-1}, file);
	save_data(SAVE_COSTS, double_dt, &train_costs.vector, 1, 1, PS(1), file);
	save_data(SAVE_DELTAX_MAGS, double_dt, deltax_mags, 1, 2, (size_t[]){data->num_samples, _net.nlayers-2}, file);

	fclose(file);

	for (int i = 0; i < data->num_samples; i++) {
		free(lenergies[i]);
		free(deltax_mags[i]);
	}
	free(lenergies);
	free(deltax_mags);
}

void free_traindata(struct traindata *data) {
	if (data) {
		free(data->iter_counts);
		free(data->train_costs);
		free(data);
	}
}

struct traindata **train_amg(struct training train, bool logging) {
	printf("Preparing to train...\n");
	_relax = train.relax;
	_logging = logging;

	size_t max_count = db_get_count(db_train);
	db_dataset data_train[_net.ntargets];
	db_dataset data_test[_net.ntargets];

	for (int i = 0; i < _net.ntargets; i++) {
		data_train[i] = db_get_dataset(db_train, _net.targets[i], _net.proc);
		
		if (train.test_samples_per_iters > 0)
			data_test[i] = db_get_dataset(db_test, _net.targets[i], _net.proc);

		if (data_train[i].count < max_count)
			max_count = data_train[i].count;
	}

	size_t num_samples = train.num_samples;
	if (num_samples == 0)
		num_samples = max_count * _net.ntargets;
	
	struct traindata **ret = malloc(sizeof(struct traindata *) * train.amg.depth);

	for (int amg_depth = 0; amg_depth < train.amg.depth; amg_depth++) {
		// do multigrid
	}
}

void clear_block_data() {
	struct block *cblock;
	FLOOP(_net, cblock) {
		if (cblock->energies) {
			for (int i = 0; i < cblock->nenergies; i++) {
				if (cblock->energies[i]) {
					free(cblock->energies[i]);
					cblock->energies[i] = NULL;
				}
			}

			free(cblock->energies);
			cblock->energies = NULL;
		}

		if (cblock->deltax_mags) {
			for (int i = 0; i < cblock->nenergies; i++) {
				if (cblock->deltax_mags[i]) {
					free(cblock->deltax_mags[i]);
					cblock->deltax_mags[i] = NULL;
				}
			}

			free(cblock->deltax_mags);
			cblock->deltax_mags = NULL;
		}

		if (cblock->deltaw_mags) {
			free(cblock->deltaw_mags);
			cblock->deltaw_mags = NULL;
		}

	}
	_net.lenergy_chunks = 0;
}

struct testdata *test(struct testing test, bool logging) {
	printf("Preparing to test ...\n");

	_relax = test.relax_params;
	_logging = logging;

	size_t max_count = test.num_samples == 0 ? db_get_count(db_test) : test.num_samples;

	db_dataset data[_net.ntargets];
	for (int i = 0; i < _net.ntargets; i++) {
		data[i] = db_get_dataset(db_test, _net.targets[i], _net.proc);

		if (data[i].count < max_count) {
			printf("Config states too many samples per target, reducing count from %ld to %ld\n", max_count, data[i].count);
			max_count = data[i].count;
		}
	}

	size_t num_samples = max_count * _net.ntargets;

	clear_block_data();
	_sample_i = 0;

	struct testdata *ret = NULL;
	if (logging) {
		ret = malloc(sizeof(struct testdata));
		ret->outputs = malloc(sizeof(gsl_vector *) * num_samples);
		ret->labels = gsl_vector_calloc(num_samples);
		ret->predictions = gsl_vector_calloc(num_samples);
		ret->costs = malloc(sizeof(gsl_vector *) * _net.ntargets);
		ret->ntargets = _net.ntargets;

		if (test.relax) {
			ret->iter_counts = gsl_vector_calloc(num_samples);
			ret->relax = true;
		} else {
			ret->relax = false;
		}
	}

	if (test.relax) {
		struct block *cblock;
		FLOOP(_net, cblock) {
			if (cblock->type == block_layer) {
				cblock->deltaw_mags = malloc(sizeof(double *) * num_samples);
				cblock->energies = malloc(sizeof(double *) * num_samples);
				cblock->nenergies = num_samples;
				for (int i = 0; i < num_samples; i++)
					cblock->energies[i] = NULL;
			}
		}
	}

	size_t counter = 0;
	size_t num_correct = 0;

	gsl_vector *label_vec = gsl_vector_calloc(_net.ntargets);
	gsl_vector *cost_vec = gsl_vector_calloc(_net.ntargets);

	_sample_i = 0;
	for (int target_i = 0; target_i < _net.ntargets; target_i++) {
		gsl_vector_set_basis(label_vec, target_i);

		if (logging)
			ret->costs[target_i] = gsl_vector_calloc(max_count);

		for (int i = 0; i < max_count; i++) {
			gsl_vector_memcpy(_net.head->layer, data[target_i].images[i]);
			if (test.relax) {
				int iters = _relaxation(false);
				if (logging)
					gsl_vector_set(ret->iter_counts, _sample_i, iters);
			} else {
				struct block *cblock;
				PLOOP(_net, cblock)
					_prop_layer(cblock, false, false);
			}

			int prediction_index = gsl_vector_max_index(_net.tail->layer);
			int prediction = _net.targets[prediction_index];

			gsl_vector_memcpy(cost_vec, _net.tail->layer);
			gsl_vector_sub(cost_vec, label_vec);
			double cost = gsl_blas_dnrm2(cost_vec);

			if (prediction == data[target_i].label)
				num_correct++;
			
			if (logging) {
				ret->outputs[_sample_i] = gsl_vector_calloc(_net.ntargets);
				gsl_vector_memcpy(ret->outputs[_sample_i], _net.tail->layer);
				gsl_vector_set(ret->labels, _sample_i, _net.targets[target_i]);
				gsl_vector_set(ret->predictions, _sample_i, prediction);
				gsl_vector_set(ret->costs[target_i], i, cost);
			}
			
			_sample_i++;
			_net.lenergy_chunks = 0;
		}
	}

	double accuracy = (double)num_correct / (double)num_samples;
	printf("Completed testing\n");
	printf("Summary: \n");
	printf("Tested on %ld images of ", num_samples);
	for(int i = 0; i < _net.ntargets; i++)
		printf("%ld (%ld),", _net.targets[i], data[i].count);
	printf("\b\n");
	printf("Accuracy = %ld/%ld = %.4f\n", num_correct, num_samples, accuracy);

	if (logging) {
		ret->num_correct = num_correct;
		ret->num_samples = num_samples;
	}

	gsl_vector_free(label_vec);
	gsl_vector_free(cost_vec);

	for (int i = 0; i < _net.ntargets; i++)
		db_free_dataset(data[i]);
	
	return ret;
}

void save_testdata(struct testdata *data, char *filename) {
	printf("Preparing to save testdata\n");
	if (!data) {
		printf("[Error] Invalid testdata\n");
		return;
	}
	
	gsl_vector ***lenergies;
	gsl_vector_view views[data->num_samples * (_net.nlayers-1)];

	if (data->relax) {
		lenergies = malloc(sizeof(gsl_vector **) * data->num_samples);

		int view_cnt = 0;

		for (int i = 0; i < data->num_samples; i++) {
			lenergies[i] = malloc(sizeof(gsl_vector *) * (_net.nlayers-1));
			int l = 0;
			struct block *cblock;
			PLOOP(_net, cblock) {
				views[view_cnt] = gsl_vector_view_array(cblock->energies[i], gsl_vector_get(data->iter_counts, i));
				lenergies[i][l] = &views[view_cnt].vector;
				// printf("lenergies[%d][%d] = %p\n", i, l , lenergies[i][l]);
				view_cnt++;
				l++;
			}
		}
	}

	FILE *file = fopen(filename, "w");
	if (!file) {
		printf("Error: Failed to open testdata file (%s)\n", filename);
		exit(ERR_FILE);
	}

	save_data(SAVE_TYPE, size_dt, PS(SAVE_TEST), 0, 1, NULL, file);

	save_data(SAVE_LABELS, double_dt, data->labels, 1, 1, PS(1), file);
	save_data(SAVE_PREDICTIONS, double_dt, data->predictions, 1, 1, PS(1), file);
	save_data(SAVE_COSTS, double_dt, data->costs, 1, 1, PS(_net.ntargets), file);
	save_data(SAVE_OUTPUTS, double_dt, data->outputs, 1, 1, PS(data->num_samples), file);
	if (data->relax) {
		printf("start tracking now\n");
		save_data(SAVE_LENERGIES, double_dt, lenergies, 1, 2, (size_t[]){data->num_samples, _net.nlayers-1}, file);
		save_data(SAVE_ITER_COUNTS, double_dt, data->iter_counts, 1, 1, PS(1), file);
	}
	
	fclose(file);

	if (data->relax) {
		for (int i = 0; i < data->num_samples; i++) {
			free(lenergies[i]);
		}
		free(lenergies);
	}

	printf("Finished saving testdata\n");
}

void free_testdata(struct testdata *data) {
	if (!data)
		return;
	
	for (int i = 0; i < data->ntargets; i++)
		gsl_vector_free(data->costs[i]);
	free(data->costs);

	gsl_vector_free(data->labels);
	gsl_vector_free(data->predictions);

	for (int i = 0; i < data->num_samples; i++)
		gsl_vector_free(data->outputs[i]);
	free(data->outputs);

	if (data->relax)
		gsl_vector_free(data->iter_counts);
	
	free(data);
}

void free_network(struct network *net) {
	free(net->targets);
	struct block *cblock = net->tail;
	while (cblock->prev) {
		cblock = cblock->prev;
		_free_block(cblock->next);
	}
	_free_block(cblock);

	free(net);
}

int _relaxation(bool training) {
	// printf("Starting relaxation\n");

	// reset layers, epsilons and deltax hidden layers
	struct block *cblock;
	HLOOP(_net, cblock) {
		gsl_vector_set_zero(cblock->layer);
		gsl_vector_set_zero(cblock->epsilon);
		gsl_vector_set_zero(cblock->deltax);
	}

	double prev_energy = 1e30;
	double energy;
	size_t gamma_count = _relax.gamma_count;
	double gamma = _relax.gamma;

	// check stop condition every iteration
	int iter = 0;
	bool stop = false;
	do {
		// calculate epsilon
		PLOOP(_net, cblock) {
			gsl_vector_memcpy(cblock->epsilon, cblock->layer);
			_prop_layer(cblock, false, true);
			gsl_vector_sub(cblock->epsilon, cblock->tlayer);
		}

		// adjust x
		HLOOP(_net, cblock) {
		// for (cblock=_net.head->next->next; cblock->next; cblock=cblock->next) {
			if (cblock->next->type == block_layer)  {
				activation_deriv_inplace(cblock->layer, cblock->tlayer, _net.act);
				gsl_blas_dgemv(CblasTrans, gamma, cblock->next->weights, cblock->next->epsilon, 0.0, cblock->deltax);
				gsl_vector_mul(cblock->deltax, cblock->tlayer);
			} else if (cblock->next->type == block_cnn) {
				int idx = 0;
				for (int c = 0; c < cblock->next->cnn->nchannels; c++) {
					for (int i = 0; i < cblock->next->cnn->image_length; i++) {
						size_t true_i = gsl_vector_get(cblock->next->cnn->pool_indices, i);
						gsl_vector_view row_view = gsl_matrix_row(cblock->next->cnn->dAdxP[c], true_i);

						gsl_matrix_set_row(cblock->next->cnn->dAdx, idx, &row_view.vector);
						idx++;
					}
				}

				gsl_blas_dgemv(CblasTrans, gamma, cblock->next->cnn->dAdx, cblock->next->epsilon, 0.0, cblock->deltax);
			}

			gsl_blas_daxpy(-gamma, cblock->epsilon, cblock->deltax);
			gsl_vector_add(cblock->layer, cblock->deltax);

			if (iter > 0 && _logging)
				cblock->deltax_mags[_sample_i][iter-1] = gsl_blas_dnrm2(cblock->deltax);
		}

		if (!training)
			gsl_blas_daxpy(-gamma, _net.tail->epsilon, _net.tail->layer);

		// check stop
		if (_relax.max_iters && iter > _relax.max_iters)
			break;

		if (_logging && (iter + 1) >= (_net.lenergy_chunks * CHUNK_SIZE)) {
			_net.lenergy_chunks++;

			struct block *cblock;
			PLOOP(_net, cblock) {
				double *new_ptr = realloc(cblock->energies[_sample_i], sizeof(double) * _net.lenergy_chunks * CHUNK_SIZE);
				if (new_ptr) {
					cblock->energies[_sample_i] = new_ptr;
				} else {
					printf("[Error] Failed to realloc lenergy data, expect errors\n");
					break;
				}
			}

			HLOOP(_net, cblock) {
				double *new_ptr = realloc(cblock->deltax_mags[_sample_i], sizeof(double) * _net.lenergy_chunks * CHUNK_SIZE);
				if (new_ptr) {
					cblock->deltax_mags[_sample_i] = new_ptr;
				} else {
					printf("[Error] Failed to realloc deltax mags data, expect errors\n");
					break;
				}
			}
		}

		energy = _calc_energies(iter);
		double res = prev_energy - energy;
		if (energy > 1e30) {
			printf("WTF thats massive %d:  %e\n", iter, energy);
			raise(SIGABRT);
			exit(100);
		}

		if (_relax.energy_res &&  res < _relax.energy_res) {
			if (res < 0) {
				// printf("prev: %.60f\n", prev_energy);
				// printf("curr: %.60f\n", energy);
				// _print_lenergies(_net, iter-1);
				// _print_lenergies(_net, iter);

				if (gamma_count > 0) {
					gamma *= _relax.gamma_rate;
					gamma_count--;
					// printf("Energy increasing, reducing gamma to %f\n", gamma);
					continue;
				}

			}
			break;
		}
		// printf(" ");
		// printf("[%5d] energy = %.30f, res = %.30f\n", iter, energy, res);
		prev_energy = energy;

		iter++;
	} while(!stop);

	printf("[%5d] Relaxation complete after %3d iterations with energy = %.30f\n", _sample_i, iter, energy);
	return iter;
}

void _prop_layer(struct block *ablock, bool temp_input, bool temp_output) {
	struct block *prev = ablock->prev;
	gsl_vector *input = temp_input ? prev->tlayer : prev->layer;
	gsl_vector *output = temp_output ? ablock->tlayer : ablock->layer;
	activation_inplace(input, prev->tlayer, _net.act);

	if (ablock->type == block_layer) {
		gsl_blas_dgemv(CblasNoTrans, 1.0, ablock->weights, prev->tlayer, 0.0, output);
	} else if (ablock->type == block_cnn) {
		struct block_cnn  *cnn = ablock->cnn;

		int prev_channels = prev->type == block_cnn ? prev->cnn->nchannels : 1;
		size_t prev_len = prev->type == block_cnn ? prev->cnn->image_length : prev->length;
		size_t inp_size = sqrt(prev_len);

		gsl_vector *act_deriv = activation_deriv(input, _net.act);
		gsl_matrix *padded_deriv[prev_channels];
		

		for (int i = 0; i < prev_channels; i++) {
			gsl_vector_view vview = gsl_vector_subvector(prev->tlayer, i*prev_len, prev_len);
			gsl_matrix_view inp_mat = gsl_matrix_view_vector(&vview.vector, inp_size, inp_size);

			gsl_matrix_view mview = gsl_matrix_submatrix(cnn->padded_input[i], cnn->padding, cnn->padding, inp_size, inp_size);
			gsl_matrix_memcpy(&mview.matrix, &inp_mat.matrix);

			gsl_vector_view dvview = gsl_vector_subvector(act_deriv, i*prev_len, prev_len);
			gsl_matrix_view dmview = gsl_matrix_view_vector(&dvview.vector, inp_size, inp_size);

			padded_deriv[i] = gsl_matrix_calloc(cnn->padded_input[0]->size1, cnn->padded_input[0]->size2);
			gsl_matrix_view pad_view = gsl_matrix_submatrix(padded_deriv[i], cnn->padding, cnn->padding, inp_size, inp_size);
			gsl_matrix_memcpy(&pad_view.matrix, &dmview.matrix);
		}

		for (int c = 0; c < cnn->nchannels; c++) {
			gsl_vector_set_zero(cnn->conv_layer);
			gsl_matrix_view cview = gsl_matrix_view_vector(cnn->conv_layer, cnn->conv_size, cnn->conv_size);
			gsl_matrix_set_zero(cnn->dAdxP[c]);

			for (int i = 0; i < cnn->conv_size; i++) {
				for (int j = 0; j < cnn->conv_size; j++) {
					double cview_val = gsl_matrix_get(&cview.matrix, i, j);
					double dadxp_val = gsl_matrix_get(cnn->dAdxP[c], i, j);
					for (int c2 = 0; c2 < prev_channels; c2++) {
						gsl_matrix_view xview = gsl_matrix_submatrix(cnn->padded_input[c2], i, j, cnn->kernel_size, cnn->kernel_size);
						gsl_matrix_view wview = gsl_matrix_submatrix(ablock->weights, 0, (c * prev_channels + c2) * cnn->kernel_size, cnn->kernel_size, cnn->kernel_size);
						gsl_matrix_view dxview = gsl_matrix_submatrix(padded_deriv[c2], i, j, cnn->kernel_size, cnn->kernel_size);

						cview_val += mat_dot(&wview.matrix, &xview.matrix);
						// gsl_matrix_set(&cview.matrix, i, j, gsl_matrix_get(&cview.matrix, i, j) + mat_dot(&wview.matrix, &xview.matrix));

						gsl_matrix_view dadwview = gsl_matrix_submatrix(cnn->dAdw[c * prev_channels + c2], 0, (i*cnn->conv_size+j)*cnn->kernel_size, cnn->kernel_size, cnn->kernel_size);
						gsl_matrix_memcpy(&dadwview.matrix, &xview.matrix);
						gsl_matrix_mul_elements(&dadwview.matrix, &dxview.matrix);

						dadxp_val += mat_dot(&dxview.matrix, &wview.matrix);

						// for (int k1 = 0; k1 < cnn->kernel_size; k1++) {
						// 	for (int k2 = 0; k2 < cnn->kernel_size; k2++) {
						// 		double prev_act = gsl_matrix_get(&xview.matrix, k1, k2);
						// 		double cur_weight = gsl_matrix_get(&wview.matrix, k1, k2);

						// 		dadxp_val += (prev_act * cur_weight);
						// 		gsl_matrix_set(cnn->dAdxP[c], i, j, gsl_matrix_get(cnn->dAdxP[c], i, j) + prev_act * cur_weight);
						// 	}
						// }
					}

					gsl_matrix_set(&cview.matrix, i, j, cview_val);
					gsl_matrix_set(cnn->dAdxP[c], i, j, dadxp_val);
				}
			}

			// pooling, currently only does max pooling
			gsl_vector_view ovview = gsl_vector_subvector(output, c*cnn->image_length, cnn->image_length);
			gsl_matrix_view oview = gsl_matrix_view_vector(&ovview.vector, cnn->image_size, cnn->image_size);
			for (int i = 0; i < cnn->image_size; i += cnn->pool_size) {
				for (int j = 0; j < cnn->image_size; j += cnn->pool_size) {
					gsl_matrix_view submat = gsl_matrix_submatrix(&cview.matrix, i, j, cnn->pool_size, cnn->pool_size);
					size_t maxi, maxj;
					gsl_matrix_max_index(&submat.matrix, &maxi, &maxj);
					gsl_vector_set(cnn->pool_indices, i*cnn->image_size+j, (i+maxi)*cnn->conv_size + (j+maxj));
					gsl_matrix_set(&oview.matrix, i, j, gsl_matrix_get(&submat.matrix, maxi, maxj));
				}
			}
		}

		for (int i = 0; i < prev_channels; i++)
			gsl_matrix_free(padded_deriv[i]);
		
		gsl_vector_free(act_deriv);
	}
}

void _adjust_w_layer(struct block *ablock) {
	gsl_matrix_set_zero(ablock->deltaw);

	activation_inplace(ablock->prev->layer, ablock->prev->tlayer, _net.act);

	gsl_blas_dger(_net.alpha, ablock->epsilon, ablock->prev->tlayer, ablock->deltaw);
	gsl_matrix_add(ablock->weights, ablock->deltaw);

	if (_logging)
		ablock->deltaw_mags[_sample_i] = frobenius_norm(ablock->deltaw);

}

void _adjust_w_cnn(struct block *ablock) {
	double mag = 0;
	size_t prev_channels = ablock->prev->type == block_cnn ? ablock->prev->cnn->nchannels : 1;
	gsl_matrix_set_zero(ablock->deltaw);
	for (int c1 = 0; c1 < ablock->cnn->nchannels; c1++) {
		for (int c2 = 0; c2 < prev_channels; c2++) {
			gsl_matrix_view dwview = gsl_matrix_submatrix(ablock->deltaw, 0, (c1*prev_channels + c2)*ablock->cnn->kernel_size, ablock->cnn->kernel_size, ablock->cnn->kernel_size);
			for (int i = 0; i < ablock->cnn->image_length; i++) {
				size_t true_i = gsl_vector_get(ablock->cnn->pool_indices, i);
				gsl_matrix_view dadw = gsl_matrix_submatrix(ablock->cnn->dAdw[c1*prev_channels + c2], 0, true_i*ablock->cnn->kernel_size, ablock->cnn->kernel_size, ablock->cnn->kernel_size);
				gsl_matrix_scale(&dadw.matrix, gsl_vector_get(ablock->epsilon, c1*ablock->cnn->image_length + i));
				gsl_matrix_add(&dwview.matrix, &dadw.matrix);
			}
		}
	}

	gsl_matrix_scale(ablock->deltaw, _net.alpha);
	gsl_matrix_add(ablock->weights, ablock->deltaw);

	double norm = frobenius_norm(ablock->deltaw);

	if (_logging)
		ablock->deltaw_mags[_sample_i] = norm;
	
	// if (_logging)
		// printf("DeltaW mag (%ld mats) = %.5f, normalized per mat = %.5f\n", ablock->cnn->nmats, norm, norm / ablock->cnn->nmats);
}

double _calc_energies(int iter) {
	double ret = 0.0;

	struct block *cblock;
	PLOOP(_net, cblock) {
		gsl_vector_memcpy(cblock->tepsilon, cblock->layer);
		_prop_layer(cblock, false, true);
		gsl_vector_sub(cblock->tepsilon, cblock->tlayer);

		double dot;
		gsl_blas_ddot(cblock->tepsilon, cblock->tepsilon, &dot);
		dot *= 0.5;

		if (_logging)
			cblock->energies[_sample_i][iter] = dot;

		ret += dot;
	}

	return ret;
}

void _print_lenergies(int iter) {
	// printf("lenergies at iter %d\n", iter);
	struct block *cur_block = _net.head->next;
	printf("\t[%3d] ", iter);
	while (cur_block) {
	if (cur_block->type == block_layer) {
			printf("%.40f ", cur_block->energies[_sample_i][iter-1]);
		}
		cur_block = cur_block->next;
	}
	printf("\n");
}

gsl_vector *restriction(gsl_vector *input, size_t out_size) {

}

struct network *_build_coarse_net() {
	struct network *ret = malloc(sizeof(struct network));
	ret->alpha = _net.alpha;
	ret->act = _net.act;
	ret->weight_init = _net.weight_init;
	ret->proc = _net.proc;

	ret->ntargets = _net.ntargets;
	ret->targets = _net.targets;

	ret->lenergy_chunks = 0;
	ret->save = _net.save;

	// create head block (special restricted block layer)
	ret->head = malloc(sizeof(struct block));
	struct block *rblock = ret->head;
	rblock->type = block_layer;
	rblock->blayer = malloc(sizeof(struct block_layer));
	rblock->length = 23456789876543245678908976; // fix this

	rblock->next = malloc(sizeof(struct block));
	rblock->next->prev = rblock;

	rblock->layer = restriction(_net.head->layer, rblock->length);
	rblock->epsilon = gsl_vector_calloc(rblock->length);
	rblock->deltax = gsl_vector_calloc(rblock->length);
	rblock->tlayer = gsl_vector_calloc(rblock->length);
	rblock->tepsilon = gsl_vector_calloc(rblock->length);

	struct block *cblock;
	PLOOP(_net, cblock) {
		if (cblock->type == block_layer) {
			// single layers are not restricted
			rblock->type = block_layer;
			rblock->blayer = malloc(sizeof(struct block_layer));
			rblock->length = cblock->length;


	// pick out weights from finer mesh
	// need to choose the correct rows so itll fit here


			
		} else if (cblock->type == block_cnn) {
			// cnn blocks are restricted
			rblock->type = block_cnn;
			rblock->cnn = malloc(sizeof(struct block_cnn));
			rblock->cnn->kernel_size = cblock->cnn->kernel_size;
			rblock->cnn->stride = cblock->cnn->stride;
			rblock->cnn->padding = cblock->cnn->padding;
			rblock->cnn->nchannels = cblock->cnn->nchannels;
			rblock->cnn->pool_size = cblock->cnn->pool_size;
			rblock->cnn->pool_type = cblock->cnn->pool_type;
			rblock->weights = cblock->weights; // not going to be adjusted so should be fine
			rblock->deltaw = NULL;

			// restrict channels
			for (int i = 0; i < rblock->cnn->nchannels; i++) {
				gsl_vector_view cview = gsl_vector_subvector(cblock->layer, i*cblock->cnn->image_length, cblock->cnn->image_length);
				gsl_vector_view rview = gsl_vector_subvector(rblock->layer, i*rblock->cnn->image_length, rblock->cnn->image_length);

				gsl_vector *restricted = restriction(&cview.vector, rblock->cnn->image_length);
				gsl_vector_memcpy(&rview.vector, restricted);
			}
		}

		if (cblock->next) {
			rblock->next = malloc(sizeof(struct block));
			rblock->next->prev = rblock;
			rblock = rblock->next;
		}
	}

	ret->tail = rblock;
}

void _free_block(struct block *ablock) {
	gsl_vector_free(ablock->layer);
	gsl_vector_free(ablock->epsilon);
	gsl_vector_free(ablock->deltax);

	gsl_vector_free(ablock->tlayer);
	gsl_vector_free(ablock->tepsilon);
	
	if (ablock->energies)
		free(ablock->energies);

	if (ablock->type == block_layer) {
		if (ablock->prev) {
			gsl_matrix_free(ablock->weights);
			gsl_matrix_free(ablock->deltaw);
		}
		free(ablock->blayer);
	} else if (ablock->type == block_cnn) {
		gsl_vector_free(ablock->cnn->pool_indices);
		gsl_vector_free(ablock->cnn->conv_layer);

		gsl_matrix_free(ablock->weights);
		gsl_matrix_free(ablock->deltaw);

		gsl_matrix_free(ablock->cnn->dAdx);

		for (int c = 0; c < ablock->cnn->nmats; c++)
			gsl_matrix_free(ablock->cnn->dAdw[c]);
		free(ablock->cnn->dAdw);

		for (int c = 0; c < ablock->cnn->nchannels; c++)
			gsl_matrix_free(ablock->cnn->dAdxP[c]);
		free(ablock->cnn->dAdxP);

		for (int c = 0; c < (ablock->prev->type == block_cnn ? ablock->prev->cnn->nchannels : 1); c++)
			gsl_matrix_free(ablock->cnn->padded_input[c]);
		free(ablock->cnn->padded_input);
		
		free(ablock->cnn);
	}

	free(ablock);
}