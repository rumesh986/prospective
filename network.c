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
static int _sample_i;
static bool _logging = false;

void _prop_layer(struct block *ablock, bool temp_input, bool temp_output);
int _relaxation(bool training);
void _adjust_eps();
// void _adjust_x(bool training);
void _adjust_w_layer(struct block *ablock);
void _adjust_w_cnn(struct block *ablock);
void _adjust_x_layer(struct block *ablock, int iter);
void _adjust_x_cnn(struct block *ablock, int iter);
bool _check_stop(int iter, bool reset);
void _free_block(struct block *ablock);
double _calc_energies(int iter);

// main code

void init_network(struct network *net) {
	printf("Initializing network...\n");

	// update input length
	net->head->blayer->length = db_get_input_length();

	net->nlayers = 0;
	// for (struct block *cur_block = net.head; cur_block; cur_block=cur_block->next) {
	for (struct block *cur_block = net->head; cur_block; cur_block=cur_block->next) {
		size_t layer_size = 0;
		if (cur_block->type == block_layer) {
			struct block_layer *layer = cur_block->blayer;

			layer_size = layer->length;
			

			if (cur_block->next && cur_block->next->type == block_layer) {
				cur_block->weights = gsl_matrix_calloc(cur_block->next->blayer->length, layer->length);
				cur_block->deltaw = gsl_matrix_calloc(cur_block->next->blayer->length, layer->length);
				// cur_block->out = gsl_vector_calloc(cur_block->next->blayer->length);
				weight_init(cur_block->weights, net->weight_init);
			} 
		} else if (cur_block->type == block_cnn) {
			size_t layer_size = cur_block->prev->layer->size;
			cur_block->layer = gsl_vector_calloc(layer_size);
			cur_block->weights = gsl_matrix_calloc(cur_block->cnn->kernel_size, cur_block->cnn->kernel_size);
		}

		cur_block->layer = gsl_vector_calloc(layer_size);
		cur_block->tlayer = gsl_vector_calloc(layer_size);
		cur_block->epsilon = gsl_vector_calloc(layer_size);
		cur_block->deltax = gsl_vector_calloc(layer_size);
		cur_block->energies = NULL;
		cur_block->deltaw_mags = NULL;
		cur_block->tepsilon = gsl_vector_calloc(layer_size);

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
			if (cblock->type == block_layer) {
				cblock->deltaw_mags = malloc(sizeof(double *) * num_samples);
				cblock->energies = malloc(sizeof(double *) * num_samples);
				cblock->deltax_mags = malloc(sizeof(double *) * num_samples);
				cblock->nenergies = num_samples;
				for (int i = 0; i < num_samples; i++)
					cblock->energies[i] = NULL;
				
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
	
		int iters = _relaxation(true);
		// _adjust_w();
		
		// update weights
		struct block *cblock;
		PLOOP2(_net, cblock) {
			switch (cblock->type) {
				case block_layer:	_adjust_w_layer(cblock);	break;
				case block_cnn:		_adjust_w_cnn(cblock);		break;
			}
		}
		
		if (train.test_samples_per_iters != 0) {
			double train_cost = 0.0;
			struct block *cblock;
			
			for (int test_i = 0; test_i < train.test_samples_per_iters; test_i++) {
				gsl_vector_memcpy(_net.head->tlayer, data_test[test_i % _net.ntargets].images[test_i]);
				gsl_vector_set_zero(_net.tail->tlayer);
				gsl_vector_set_basis(test_label_vec, test_i % _net.ntargets);

				PLOOP2(_net, cblock) 
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
	// gsl_vector_view **delta_w_mags = malloc(sizeof(gsl_vector_view) * _net.nlayers-1);
	// gsl_vector *lenergies[data->num_samples][_net.nlayers-1];
	gsl_vector *delta_w_mags[_net.nlayers-1];
	// gsl_vector *iter_counts = gsl_vector_calloc(data->num_samples);
	// gsl_vector *train_costs = gsl_vector_calloc(data->num_samples);

	gsl_vector_view iter_counts = gsl_vector_view_array(data->iter_counts, data->num_samples);
	gsl_vector_view train_costs = gsl_vector_view_array(data->train_costs, data->num_samples);

	gsl_vector_view views[data->num_samples * (_net.nlayers-1)];
	gsl_vector_view deltax_views[data->num_samples * (_net.nlayers-2)];
	int view_counter = 0;
	int deltax_view_counter = 0;

	struct block *cblock;
	int l = 0;
	PLOOP2(_net, cblock) {
		views[view_counter] = gsl_vector_view_array(cblock->deltaw_mags, data->num_samples);
		delta_w_mags[l] = &views[view_counter].vector;
		
		// print_vec(delta_w_mags[l], "deltawmags", false);
		l++;
		view_counter++;
	}

	view_counter = 0;

	for (int i = 0; i < data->num_samples; i++) {
		lenergies[i] = malloc(sizeof(gsl_vector *) * (_net.nlayers-1));
		deltax_mags[i] = malloc(sizeof(gsl_vector *) * (_net.nlayers-2));
		l = 0;
		PLOOP(_net, cblock) {
			views[view_counter] = gsl_vector_view_array(cblock->energies[i], data->iter_counts[i]);
			lenergies[i][l] = &views[view_counter].vector;
			// char vectitle[32];
			// sprintf(vectitle, "i=%d,l=%d,L=%ld", i, l, lenergies[i][l]->size);
			// print_vec(lenergies[i][l], vectitle, false);
			// printf("compiling lenergies i = %d, l = %d, p = %p\n", i, l, lenergies[i][l]);
			view_counter++;
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

	for (int i = 0; i < data->num_samples; i++)
		free(lenergies[i]);
	free(lenergies);
}

void free_traindata(struct traindata *data) {
	if (data) {
		free(data->iter_counts);
		free(data->train_costs);
		free(data);
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
				PLOOP2(_net, cblock)
					_prop_layer(cblock, false, false);
					// gsl_vector_memcpy(cblock->next->layer, cblock->out);
					
				// gsl_vector_memcpy(_net.tail->layer, _net.tail->prev->out);
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
			// for (int l = 0; l < _net.nlayers-1; l++)
				// gsl_vector_free(lenergies[i][l]);
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
	struct block *cblock = net->head;
	while (cblock->next) {
		cblock = cblock->next;
		_free_block(cblock->prev);
	}
	_free_block(cblock);

	free(net);
}

int _relaxation(bool training) {
	// printf("Starting relaxation\n");

	// reset layers, epsilons and deltax hidden layers
	struct block *cur_block;
	HLOOP(_net, cur_block) {
		gsl_vector_set_zero(cur_block->layer);
		gsl_vector_set_zero(cur_block->epsilon);
		gsl_vector_set_zero(cur_block->deltax);
	}

	double prev_energy = 1e30;
	size_t gamma_count = _relax.gamma_count;
	double gamma = _relax.gamma;

	// check stop condition every iteration
	int iter = 0;
	bool stop = false;
	// bool reset = true;
	struct block *cblock;
	do {
		// calculate epsilon
		PLOOP(_net, cblock) {
			gsl_vector_memcpy(cblock->epsilon, cblock->layer);
			_prop_layer(cblock->prev, false, true);
			gsl_vector_sub(cblock->epsilon, cblock->tlayer);
		}

		// adjust x
		HLOOP(_net, cblock) {
			switch (cblock->type) {
				case block_layer:	_adjust_x_layer(cblock, iter);	break;
				case block_cnn:		_adjust_x_cnn(cblock, iter);		break;
			}
		}

		if (!training)
			gsl_blas_daxpy(-gamma, _net.tail->epsilon, _net.tail->layer);

		// check stop
		if (_relax.max_iters && iter > _relax.max_iters)
			break;
			// stop = true;

		if (_logging && (iter + 1) >= (_net.lenergy_chunks * CHUNK_SIZE)) {
			_net.lenergy_chunks++;

			struct block *cblock;
			PLOOP(_net, cblock) {
				double *new_ptr = realloc(cblock->energies[_sample_i], sizeof(double) * _net.lenergy_chunks * CHUNK_SIZE);
				if (new_ptr) {
					cblock->energies[_sample_i] = new_ptr;
				} else {
					printf("[Error] Failed to realloc lenergy data, expect errors\n");
					// stop = true;
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

		double energy = _calc_energies(iter);
		double res = prev_energy - energy;
		if (energy > 1e30) {
			printf("WTF thats massive %d:  %e\n", iter, energy);
			exit(100);
		}

		if (_relax.energy_res &&  res < _relax.energy_res) {
			if (res < 0) {
				printf("prev: %.60f\n", prev_energy);
				printf("curr: %.60f\n", energy);
				// _print_lenergies(_net, iter-1);
				// _print_lenergies(_net, iter);

				if (gamma_count > 0) {
					gamma *= _relax.gamma_rate;
					gamma_count--;
					printf("Energy increasing, reducing gamma to %f\n", gamma);
					// stop = false;
					continue;
				}

			}
			// stop = true;
			break;
		}
		
		prev_energy = energy;

		// _adjust_eps();
		// _adjust_x(training);
		// stop = _check_stop(iter, reset);
		iter++;
		// reset = false;
	} while(!stop);

	// if (training)
	// 	_trainresults->iter_counts[_sample_i] = iter;

	// printf("[%5d] Relaxation complete after %d iterations\n", _sample_i, iter);
	return iter;
}

void _prop_layer(struct block *ablock, bool temp_input, bool temp_output) {
	if (ablock->type == block_layer) {
		struct block_layer *layer = ablock->blayer;

		activation_inplace(temp_input ? ablock->tlayer : ablock->layer, ablock->tlayer, _net.act);
		gsl_blas_dgemv(CblasNoTrans, 1.0, ablock->weights, ablock->tlayer, 0.0, temp_output ? ablock->next->tlayer : ablock->next->layer);
	} else if (ablock->type == block_cnn) {
		for (int i = 0; i < ablock->next->layer->size; i++) {
			struct block_cnn *cnn = ablock->cnn;
			struct db_image_info img_shape = db_get_image_info();
			cnn->layer_mat = gsl_matrix_view_vector(ablock->layer, img_shape.size1, img_shape.size2);
			gsl_matrix_view mat_view = gsl_matrix_submatrix(&cnn->layer_mat.matrix, i / img_shape.size2, i % img_shape.size2 , cnn->kernel_size, cnn->kernel_size);

			gsl_vector_set(temp_output ? ablock->next->tlayer : ablock->next->layer, i, mat_dot(ablock->weights, &mat_view.matrix));
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

// void _adjust_eps() {
// 	struct block *cblock;
// 	PLOOP(_net, cblock) {
// 		gsl_vector_memcpy(cblock->epsilon, cblock->layer);
// 		_prop_layer(cblock->prev, false, true);
// 		gsl_vector_sub(cblock->epsilon, cblock->tlayer);

// 		// switch (cblock->type) {
// 		// 	case block_layer:	_adjust_eps_layer(cblock);	break;
// 		// 	case block_cnn: 	_adjust_eps_cnn(cblock);	break;
// 		// }
// 	}
// }

void _adjust_x_layer(struct block *ablock, int iter) {
	struct block_layer *layer = ablock->blayer;
	
	activation_deriv_inplace(ablock->layer, ablock->tlayer, _net.act);
	gsl_blas_dgemv(CblasTrans, _relax.gamma, ablock->weights, ablock->next->epsilon, 0.0, ablock->deltax);
	gsl_vector_mul(ablock->deltax, ablock->tlayer);

	gsl_blas_daxpy(-_relax.gamma, ablock->epsilon, ablock->deltax);
	gsl_vector_add(ablock->layer, ablock->deltax);

	if (iter > 0 && _logging) {
		ablock->deltax_mags[_sample_i][iter-1] = gsl_blas_dnrm2(ablock->deltax);
	}
}

void _adjust_x_cnn(struct block *ablock, int iter) {

}

// void _adjust_x(bool training) {
// 	struct block *cblock;
// 	// for (struct block *cblock = _net.head->next; training ? cblock->next : cblock; cblock=cblock->next) {
// 	HLOOP(_net, cblock) {
// 		switch (cblock->type) {
// 			case block_layer:	_adjust_x_layer(cblock);	break;
// 			case block_cnn:		_adjust_x_cnn(cblock);		break;
// 		}
// 	}

// 	if (!training)
// 		gsl_blas_daxpy(-_relax.gamma, _net.tail->epsilon, _net.tail->layer);
// }

void _adjust_w_layer(struct block *ablock) {
	activation_inplace(ablock->layer, ablock->tlayer, _net.act);

	gsl_blas_dger(_net.alpha, ablock->next->epsilon, ablock->tlayer, ablock->deltaw);
	gsl_matrix_add(ablock->weights, ablock->deltaw);

	if (_logging)
		ablock->deltaw_mags[_sample_i] = frobenius_norm(ablock->deltaw);

	gsl_matrix_set_zero(ablock->deltaw);
}

void _adjust_w_cnn(struct block *ablock) {

}

// void _adjust_w() {

// }

double _calc_energies(int iter) {
	double ret = 0.0;

	struct block *cblock;
	PLOOP(_net, cblock) {
		if (cblock->type == block_layer) {
			struct block_layer *layer = cblock->blayer;

			gsl_vector_memcpy(cblock->tepsilon, cblock->layer);
			_prop_layer(cblock->prev, false, true);
			gsl_vector_sub(cblock->tepsilon, cblock->tlayer);

			double dot;
			gsl_blas_ddot(cblock->tepsilon, cblock->tepsilon, &dot);
			dot *= 0.5;

			if (_logging)
				cblock->energies[_sample_i][iter] = dot;
			ret += dot;
		}
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

// bool _check_stop(int iter, bool reset) {
// 	if (_relax.max_iters && iter > _relax.max_iters)
// 		return true;
	
// 	if (_logging && (iter + 1) >= (_net.lenergy_chunks * CHUNK_SIZE)) {
// 		_net.lenergy_chunks++;

// 		struct block *cblock;
// 		PLOOP(_net, cblock) {
// 			double *new_ptr = realloc(cblock->energies[_sample_i], sizeof(double) * _net.lenergy_chunks * CHUNK_SIZE);
// 			if (new_ptr) {
// 				cblock->energies[_sample_i] = new_ptr;
// 			} else {
// 				printf("[Error] Failed to realloc lenergy data, expect errors\n");
// 				return true;
// 			}
// 		}
// 	}
	
// 	static double prev_energy = 1e30;
// 	static size_t gamma_count = 0;
// 	static double gamma = 0.1;

// 	if (reset) {
// 		prev_energy = 1000000000000000000000000000000.0;
// 		gamma_count = _relax.gamma_count;
// 		gamma = _relax.gamma;
// 		return false;
// 	}

// 	double energy = _calc_energies(iter);
// 	_print_lenergies(iter);
// 	double res = prev_energy - energy;

// 	if (energy > 1e30) {
// 		printf("WTF thats massive %d:  %e\n", iter, energy);
// 		exit(100);
// 	}

// 	// energy does crazy high (>1e30) if printf is commented out
// 	// absolutely no clue why
// 	// printf(" ");
// 	// printf("[%3d] energy: %e, res = %e\n", iter, energy, res);
// 	if (_relax.energy_res &&  res < _relax.energy_res) {
// 		if (res < 0) {
// 			printf("prev: %.60f\n", prev_energy);
// 			printf("curr: %.60f\n", energy);
// 			// _print_lenergies(_net, iter-1);
// 			// _print_lenergies(_net, iter);

// 			if (gamma_count > 0) {
// 				gamma *= _relax.gamma_rate;
// 				gamma_count--;
// 				printf("Energy increasing, reducing gamma to %f\n", gamma);
// 				return false;
// 			}

// 		}
// 		return true;

// 	}
	
// 	prev_energy = energy;
// }

void _free_block(struct block *ablock) {
	gsl_vector_free(ablock->layer);
	gsl_vector_free(ablock->epsilon);
	gsl_vector_free(ablock->deltax);

	gsl_vector_free(ablock->tlayer);
	gsl_vector_free(ablock->tepsilon);

	if (ablock->next) {
		gsl_matrix_free(ablock->weights);
		gsl_matrix_free(ablock->deltaw);
	}
	
	if (ablock->energies)
		free(ablock->energies);

	if (ablock->type == block_layer) {
		free(ablock->blayer);
	} else {
		free(ablock->cnn);
	}

	free(ablock);
}