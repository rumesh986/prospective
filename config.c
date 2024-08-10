#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

#include <cjson/cJSON.h>

#include "include/all.h"
#include "include/config.h"

// local header

#define CMP_STR(a, b) strcmp(a, b) == 0
#define CMP_VAL(a, b) CMP_STR(a->valuestring, b)

enum dtype {
	int_t,
	size_tt,
	double_tt,
	bool_tt
};

struct config_option {
	char *key;
	int value;
	// enum dtype type;
};

struct mapping {
	char *label;
	void *ptr;
};

// struct network **all_structures;
size_t num_blocks;
struct mapping *block_mapping;

// struct relaxations **all_relaxations;
size_t num_relaxations;
struct mapping *relaxation_mapping;

size_t num_networks = 0;
struct mapping *network_mapping;


// store original config data as parsed
// makes it easier to save file in the future
// remember to free at end of program
cJSON *_config;

cJSON *_parse_file(char *filename);
void set_enum(cJSON *main_config, char *key, int *setting, size_t num_options, struct config_option *options, int default_val );
void set_val(cJSON *main_config, char *key, void *setting, void * default_val, enum dtype type);
void *get_ptr(struct mapping *mapping, size_t len, char *label);
void _print_network(struct network *net, char *level);
void _print_relaxation(struct relaxation_params *relax, char *level);

// main code

struct config parse_config(char *filename) {
	_config = _parse_file(filename);

	struct config ret;

	cJSON *general = cJSON_GetObjectItem(_config, "general");
	cJSON *blocks = cJSON_GetObjectItem(_config, "blocks");
	cJSON *relaxations = cJSON_GetObjectItem(_config, "relaxations");
	cJSON *operations = cJSON_GetObjectItem(_config, "operations");

	// To do: support for loading networks
	// cJSON *networks = cJSON_GetObjectItem(_config, "networks");

	if (!general || !blocks || !relaxations || !operations) {
		printf("[Config][Error] Invalid configuration file\n");
		exit(ERR_INVALID_CONFIG);
	}

	struct config_option db_options[2] = {
		{"mnist", MNIST},
		{"fashion_mnist", FashionMNIST}
	};
	set_enum(general, "database", (int *)&ret.db, 2, db_options, MNIST);
	set_val(general, "logging", &ret.logging, PB(true), bool_tt);

	if (cJSON_HasObjectItem(general, "label")) {
		char *gen_label = cJSON_GetObjectItem(general, "label")->valuestring;

		ret.label = malloc(strlen(gen_label)+2);
		// strcpy(ret.label, gen_label);
		sprintf(ret.label, "%s ", gen_label);
	} else {
		ret.label = malloc(1);
		strcpy(ret.label, "");
	}

	struct config_option proc_options[5] = {
		{"original", proc_original},
		{"normalized", proc_normalize},
		{"normalised", proc_normalize},
		{"binarised", proc_binarize},
		{"binarized", proc_binarize}
	};

	struct config_option act_options[3] = {
		{"linear", act_linear},
		{"sigmoid", act_sigmoid},
		{"relu", act_relu}
	};

	struct config_option weights_options[4] = {
		{"zero", weights_zero},
		{"one", weights_one},
		{"xavier_normal", weights_xavier_normal},
		{"xavier_uniform", weights_xavier_uniform}
	};

	cJSON *element;
	int arr_count = 0;

	// handle blocks
	num_blocks = cJSON_GetArraySize(blocks);
	block_mapping = malloc(sizeof(struct mapping) * num_blocks);
	cJSON_ArrayForEach(element, blocks) {
		cJSON *type = cJSON_GetObjectItem(element, "type");

		// standard MLP type structure
		if (CMP_VAL(type, "dense")) {
			cJSON *db = cJSON_GetObjectItem(element, "dense");
			cJSON *label = cJSON_GetObjectItem(db, "label");
			if (!db || !label) {
				printf("Invalid structure defintion. Skipping....\n");
				continue;
			}

			struct dense_block *block = malloc(sizeof(struct dense_block));
			
			cJSON *layers = cJSON_GetObjectItem(db, "layers");
			if (!layers) {
				printf("Invalid structure definition, no layers specified. Skipping...\n");
				free(block);
				continue;
			}

			block->nlayers = cJSON_GetArraySize(layers);
			block->lengths = malloc(sizeof(size_t) * block->nlayers);
			for (int i = 0; i < block->nlayers; i++)
				block->lengths[i] = cJSON_GetArrayItem(layers, i)->valueint;
			
			set_enum(db, "activation", (int *)&block->act, 3, act_options, act_sigmoid);
			set_enum(db, "weights", (int *)&block->weight_init, 4, weights_options, weights_xavier_uniform);
			set_val(db, "alpha", &block->alpha, PD(0.1), double_tt);

			// initial value for weights, will be changed later if this particular block is used in a network
			block->weights = NULL;
			block->deltaw = NULL;
			block->layers = NULL;
			block->epsilons = NULL;
			block->deltax = NULL;

			struct block *abstract_block = malloc(sizeof(struct block));
			abstract_block->type = block_dense;
			abstract_block->dense = block;
			abstract_block->next = NULL;
			abstract_block->prev = NULL;

			block_mapping[arr_count] = (struct mapping){cJSON_GetStringValue(label), abstract_block};
			// all_structures[arr_count] = structure;
		} else {
			printf("[Config][Error] Unknown structure type specified, skipping...\n");
			continue;
		}
		arr_count++;
	}

	// handle relaxations
	arr_count = 0;
	num_relaxations = cJSON_GetArraySize(relaxations);
	relaxation_mapping = malloc(sizeof(struct mapping) * num_relaxations);
	cJSON_ArrayForEach(element, relaxations) {
		cJSON *label = cJSON_GetObjectItem(element, "label");
		if (!label) {
			printf("Invalid relaxation configuration, skipping...\n");
			continue;
		}

		struct relaxation_params *rel = malloc(sizeof(struct relaxation_params));

		set_val(element, "gamma", &rel->gamma, PS(0.1), double_tt);
		set_val(element, "gamma_rate", &rel->gamma_rate, PS(0.5), double_tt);
		set_val(element, "gamma_count", &rel->gamma_count, PS(0), size_tt);
		set_val(element, "energy_residual",  &rel->energy_res, PS(0.0), double_tt);
		set_val(element, "max_iterations", &rel->max_iters, PS(0), size_tt);

		relaxation_mapping[arr_count] = (struct mapping){cJSON_GetStringValue(label), rel};
		// all_relaxations[arr_count] = rel;
		arr_count++;
	}

	// handle operations
	arr_count = 0;
	ret.num_operations = cJSON_GetArraySize(operations);
	ret.operations = malloc(sizeof(struct operation) * ret.num_operations);
	// size_t cur_num_network = 0;
	network_mapping = malloc(1);
	cJSON_ArrayForEach(element, operations) {
		cJSON *type = cJSON_GetObjectItem(element, "type");
		if (CMP_VAL(type, "training")) {
			cJSON *op = cJSON_GetObjectItem(element, "training");

			ret.operations[arr_count].type = op_training;
			struct training *train = &ret.operations[arr_count].training;

			cJSON *relax_label = cJSON_GetObjectItem(op, "relaxation");
			cJSON *network_label = cJSON_GetObjectItem(op, "label");
			train->params.relax = get_ptr(relaxation_mapping, num_relaxations, cJSON_GetStringValue(relax_label));
			if (!train->params.relax) {
				printf("[Config][Error] Relaxation object not set, skipping...\n");
				continue;
			}

			// handle networks
			cJSON *net_blocks = cJSON_GetObjectItem(op, "network");
			if (!net_blocks || cJSON_GetArraySize(net_blocks) == 0 || !network_label) {
				printf("[Config][Error] Invalid training configuration, missing network labels. skipping...\n");
				continue;
			}

			int arr_count2 = 0;
			struct network *net = malloc(sizeof(struct network));
			net->nblocks = cJSON_GetArraySize(net_blocks);
			net->blocks = malloc(sizeof(struct block) * net->nblocks);
			cJSON *element2;
			cJSON_ArrayForEach(element2, net_blocks) {
				struct block *block = get_ptr(block_mapping, num_blocks, cJSON_GetStringValue(element2));
				if (!block) {
					printf("[Config][Error] Block not found, please check labels. exiting...\n");
					exit(2);
				}

				net->blocks[arr_count2] = block;
				arr_count2++;
			}

			num_networks++;
			network_mapping = realloc(network_mapping, sizeof(struct mapping) * num_networks);
			network_mapping[num_networks-1] = (struct mapping){cJSON_GetStringValue(network_label), net};

			// struct network *net = malloc(sizeof(struct network));
			net->head = get_ptr(block_mapping, num_blocks, cJSON_GetStringValue(cJSON_GetArrayItem(net_blocks, 0)));
			struct block *block = net->head;
			for (int i = 1; i < net->nblocks; i++) {
				block->next = get_ptr(block_mapping, num_blocks, cJSON_GetStringValue(cJSON_GetArrayItem(net_blocks, i)));
				block->next->prev = block;
				block = block->next;
			}
			
			net->training = train;
			train->net = net;
			set_enum(op, "processing", (int *)&train->params.proc, 5, proc_options, proc_normalize);
			set_val(op, "train_samples", &train->params.num_samples, PS(0), size_tt);
			set_val(op, "seed", &train->params.seed, PS(0), size_tt);
			set_val(op, "test_samples_per_iter", &train->params.test_samples_per_iters, PS(0), size_tt);
			
			cJSON *targets = cJSON_GetObjectItem(op, "targets");
			if (!targets || cJSON_GetArraySize(targets) == 0) {
				printf("Targets not specified, assuming all\n");
				train->ntargets = 0;
				train->targets = NULL;
			} else {
				train->ntargets = cJSON_GetArraySize(targets);
				train->targets = malloc(sizeof(size_t) * train->ntargets);
				for (int i = 0; i < train->ntargets; i++)
					train->targets[i] = cJSON_GetArrayItem(targets, i)->valueint;
			}
		} else if (CMP_VAL(type, "testing")) {
			cJSON *op = cJSON_GetObjectItem(element, "testing");
			ret.operations[arr_count].type = op_testing;

			struct testing *test = &ret.operations[arr_count].testing;
			set_val(op, "num_samples", &test->num_samples, PS(0), size_tt);

			cJSON *net_label = cJSON_GetObjectItem(op, "network");
			test->net = get_ptr(network_mapping, num_networks, cJSON_GetStringValue(net_label));

			cJSON *relax_label = cJSON_GetObjectItem(op, "relaxation");
			if (relax_label && !cJSON_IsNull(relax_label))
				test->relax = get_ptr(relaxation_mapping, num_relaxations, cJSON_GetStringValue(relax_label));
			else
				test->relax = NULL;

		} else {
			printf("Invalid operation type specified, skipping...\n");
			continue;
		}

		arr_count++;
	}

	ret.num_networks = num_networks;
	ret.networks = malloc(sizeof(struct network *) * ret.num_networks);
	for (int i = 0; i < ret.num_networks; i++)
		ret.networks[i] = network_mapping[i].ptr;

	return ret;
}

void free_config(struct config config) {
	free(config.label);

	for (int i = 0; i < config.num_operations; i++) {
		if (config.operations[i].type == op_training) {
			free(config.operations[i].training.targets);
		}
	}	
	free(config.operations);
	free(config.networks);

	for (int i = 0; i < num_blocks; i++) {
		struct block *ablock = block_mapping[i].ptr;

		if (ablock->type == block_dense) {
			struct dense_block *dblock = ablock->dense;

			if (dblock->weights) {
				for (int j = 1; j < dblock->nlayers+1; j++) {
					if (dblock->weights[j])
						gsl_matrix_free(dblock->weights[j]);
				}
				free(dblock->weights);
			}

			if (dblock->deltaw) {
				for (int j = 0; j < dblock->nlayers; j++) {
					if (dblock->deltaw[j])
						gsl_matrix_free(dblock->deltaw[j]);
				}
				free(dblock->deltaw);
			}

			if (dblock->layers) {
				for (int j = 1; j < dblock->nlayers+1; j++) {
					if (dblock->layers[j])
						gsl_vector_free(dblock->layers[j]);
				}
				free(dblock->layers);
			}

			if (dblock->epsilons) {
				for (int j = 1; j < dblock->nlayers+1; j++) {
					if (dblock->epsilons[j])
						gsl_vector_free(dblock->epsilons[j]);
				}
				free(dblock->epsilons);
			}

			if (dblock->deltax) {
				for (int j = 0; j < dblock->nlayers; j++) {
					if (dblock->deltax[j])
						gsl_vector_free(dblock->deltax[j]);
				}
				free(dblock->deltax);
			}

			free(dblock->lengths);
			free(dblock);
		}
		free(ablock);
	}
	free(block_mapping);

	for (int i = 0; i < num_relaxations; i++)
		free(relaxation_mapping[i].ptr);
	free(relaxation_mapping);
	
	for (int i = 0; i < num_networks; i++) {
		free(((struct network *)network_mapping[i].ptr)->blocks);
		free(network_mapping[i].ptr);
	}

	free(network_mapping);
	cJSON_Delete(_config);
}

void save_config(struct config config, char *filename) {
	cJSON *ops = cJSON_GetObjectItem(_config, "operations");
	cJSON *blocks = cJSON_GetObjectItem(_config, "blocks");
	cJSON *elem;
	cJSON_ArrayForEach(elem, ops) {
		cJSON *op = cJSON_GetObjectItem(elem, "training");
		if (!op)
			continue;

		cJSON *label = cJSON_GetObjectItem(op, "label");
		struct network *net = get_ptr(network_mapping, num_networks, cJSON_GetStringValue(label));

		cJSON *seed = cJSON_GetObjectItem(op, "seed");
		if (cJSON_GetNumberValue(seed) == 0)
			cJSON_SetNumberValue(seed, net->training->params.seed);
	}


	// cJSON *name_config = cJSON_GetObjectItem(_config, "name");
	// if (!cJSON_HasObjectItem(_config, "name"))
	// 	cJSON_AddStringToObject(_config, "name", config.net_name);
	
	// cJSON *training_config = cJSON_GetObjectItem(_config, "training");
	// cJSON *seed_config = cJSON_GetObjectItem(training_config, "seed");
	// if (cJSON_GetNumberValue(seed_config) == 0) 
	// 	cJSON_SetNumberValue(seed_config, config.train_params.seed);
	
	// cJSON *network_config = cJSON_GetObjectItem(_config, "network");
	// cJSON *layers_config = cJSON_GetObjectItem(network_config, "layers");
	// cJSON *first_layer = cJSON_CreateNumber(config.net_params.lengths[0]);
	// cJSON *last_layer = cJSON_CreateNumber(config.net_params.lengths[config.net_params.nlayers-1]);
	// cJSON_InsertItemInArray(layers_config, 0, first_layer);
	// cJSON_AddItemToArray(layers_config, last_layer);
	

	char *print = cJSON_Print(_config);
	FILE *file = fopen(filename, "w");
	if (!file) {
		printf("[Config] Error failed to open file to save config (%s)\n", filename);
		return;
	} 

	fwrite(print, 1, strlen(print), file);

	free(print);
	fclose(file);
}

void print_config(struct config config) {
	printf("DB: ");
	switch (config.db) {
		case MNIST: 		printf("MNIST");		break;
		case FashionMNIST:	printf("FashionMNIST");	break;
		default:			printf("Invalid");		break;
	}
	printf("\n");
	printf("num_networks: %ld\n", config.num_networks);

	for (int i = 0; i < config.num_networks; i++) {
		printf("Network %d\n", i);
		_print_network(config.networks[i], "");
	}

	printf("num_operations: %ld\n", config.num_operations);
	for (int i = 0; i < config.num_operations; i++) {
		printf("Operation %d, type: ", i);

		if (config.operations[i].type == op_training) {
			printf("Training\n");
			struct training training = config.operations[i].training;
			printf("\tntargets: %ld\n", training.ntargets);
			printf("\ttargets: [");
			for (int j = 0; j < training.ntargets; j++) 
				printf("%ld ", training.targets[j]);
			printf("]\n");

			printf("\tprocessing: ");
			switch (training.params.proc) {
				case proc_original: 	printf("None");			break;
				case proc_normalize:	printf("Normalize");	break;
				case proc_binarize:		printf("Binarize");		break;
				default:				printf("Unknown");		break;
			}
			printf("\n");

			printf("\tnum_samples: %ld\n", training.params.num_samples);
			printf("\tseed: %ld\n", training.params.seed);
			printf("\ttest_samples_per_iter: %ld\n", training.params.test_samples_per_iters);
			_print_relaxation(training.params.relax, "\t");
			printf("\tNetwork\n");
			_print_network(training.net, "\t");
		} else if (config.operations[i].type == op_testing) {
			printf("Testing\n");
			
			struct testing testing = config.operations[i].testing;
			printf("\tnum_samples: %ld\n", testing.num_samples);
			printf("\tNetwork\n");
			_print_network(testing.net, "\t");
			_print_relaxation(testing.relax, "\t");
		} else {
			printf("Unknown Operation\n");
		}
	}

}

struct block *new_dense_block(size_t nlayers, size_t *lengths) {
	struct block *ret = malloc(sizeof(struct block));
	ret->type = block_dense;
	ret->dense = malloc(sizeof(struct dense_block));
	ret->dense->nlayers = nlayers;
	ret->dense->lengths = lengths;
	ret->prev = NULL;
	ret->next = NULL;

	num_blocks++;
	struct mapping *new_block_map_ptr = realloc(block_mapping, sizeof(struct mapping) * num_blocks);
	if (new_block_map_ptr)
		block_mapping = new_block_map_ptr;
	else {
		printf("Failed to change block mapping size\n");
		exit(4);
	}
	block_mapping[num_blocks-1] = (struct mapping){"", ret};
	return ret;
}

// private functions

cJSON *_parse_file(char *filename) {
	FILE *file = fopen(filename, "r");
	if (!file) {
		printf("ERROR: Failed to open config file (%s)\n", filename);
		return NULL;
	}

	size_t chunk_size = 32;
	size_t nchunks = 1;
	char *filestring = calloc(1, 1);
	char buffer[chunk_size];
	size_t nread;
	do {
		nread = fread(buffer, 1, chunk_size, file);
		filestring = realloc(filestring, nchunks * chunk_size + 1);
		strncat(filestring, buffer, nread);
		nchunks ++;
	} while (nread == chunk_size);

	fclose(file);

	cJSON *ret = cJSON_Parse(filestring);
	if (!ret) {
		printf("[Config] Error reading file\n");
	}

	free(filestring);

	return ret;
}

void set_enum(cJSON *main_config, char *key, int *setting, size_t num_options, struct config_option *options, int default_val ) {
	cJSON *config = cJSON_GetObjectItem(main_config, key);
	if (!config) {
		printf("[Config] %s not specified, defaulting to %d\n", key, default_val);
	} else {
		for (int i = 0; i < num_options; i++) {
			if (CMP_VAL(config, options[i].key)) {
				*setting = options[i].value;
			}
		}
	}
}

void set_val(cJSON *main_config, char *key, void *setting, void * default_val, enum dtype type) {
	cJSON *config = cJSON_GetObjectItem(main_config, key);
	if (!config) {
		if (type == int_t) {
			printf("[Config] %s not specified, defaulting to %d\n", key, *(int *)default_val);
			*(int *)setting = *(int *)default_val;
		} else if (type == size_tt) {
			printf("[Config] %s not specified, defaulting to %ld\n", key, *(size_t *)default_val);
			*(size_t *)setting = *(size_t *)default_val;
		} else if (type == double_tt) {
			printf("[Config] %s not specified, defaulting to %f\n", key, *(double *)default_val);
			*(double *)setting = *(double *)default_val;
		} else if (type == bool_tt) {
			printf("[Config] %s not specified, defaulting to %s\n", key, *(bool *)default_val ? "true" : "false");
		} else {
			printf("[Config] %s not specified, invalid type provided, unable to set default value\n", key);
			return;
		}
	} else {
		if (type == int_t)
			*(int *)setting = config->valueint;
		else if (type == size_tt)
			*(size_t *)setting = (size_t)config->valueint;
		else if (type == double_tt)
			*(double *)setting = config->valuedouble;
		else if (type == bool_tt)
			*(bool *)setting = cJSON_IsTrue(config);
		else {
			printf("Invalid command specify type\n");
		}
	}
}

void *get_ptr(struct mapping *mapping, size_t len, char *label) {
	// printf("Looking for %s among %ld mappings\n", label, len);
	for (int i = 0; i < len; i++) {
		if (CMP_STR(mapping[i].label, label))
			return mapping[i].ptr;
	}
	return NULL;
}

void _print_network(struct network *net, char *level) {
	if (net) {
		printf("%snblocks: %ld\n", level, net->nblocks);
		int index = 0;
		printf("Linked list form:\n");
		struct block *block = net->head;
		while(block) {
			printf("%sblock %d, type: ", level, index);
			if (block->type == block_dense) {
				printf("Dense\n");
				struct dense_block *dblock = block->dense;
				printf("%s\talpha: %.4f\n", level, dblock->alpha);

				printf("%s\tactivation: ", level);
				switch (dblock->act) {
					case act_linear:	printf("Linear");	break;
					case act_sigmoid:	printf("Sigmoid");	break;
					case act_relu:		printf("ReLU");		break;
					default:			printf("Unknonw");	break;
				}
				printf("\n");

				printf("%s\tWeight init: ", level);
				switch (dblock->weight_init) {
					case weights_zero:				printf("Zeros");			break;
					case weights_one:				printf("Ones");				break;
					case weights_xavier_normal:		printf("Xavier normal");	break;
					case weights_xavier_uniform:	printf("Xavier uniform");	break;
					default:						printf("Unknown");			break;
				}
				printf("\n");

				printf("%s\tnlayers: %ld\n", level, dblock->nlayers);
				printf("%s\tlayers: [", level);
				for (int k = 0; k < dblock->nlayers; k++)
					printf("%ld ", dblock->lengths[k]);
				printf("]\n");
			}
			block = block->next;
			index++;
		}
	}
}

void _print_relaxation(struct relaxation_params *relax, char *level) {
	printf("%sRelaxation\n", level);
	if (relax) {
		printf("%s\tgamma: %.4f\n", level, relax->gamma);
		printf("%s\tgamma_rate: %.4f\n", level, relax->gamma_rate);
		printf("%s\tgamma_count: %ld\n", level, relax->gamma_count);
		printf("%s\tenergy_res: %.4f\n", level, relax->energy_res);
		printf("%s\tmax_iters: %ld\n", level, relax->max_iters);
	} else {
		printf("%s\tNULL\n", level);
	}
}