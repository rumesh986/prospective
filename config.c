#include <string.h>

#include <cjson/cJSON.h>

#include "include/all.h"
#include "include/config.h"

// local header

#define CMP_STR(a, b) strcmp(a, b) == 0
#define CMP_VAL(a, b) CMP_STR(a->valuestring, b)

enum dtype {
	int_dt,
	size_dt,
	double_dt,
	bool_dt
};

struct _map_pair {
	char *key;
	void *value;
	enum dtype type;
};

struct _map {
	size_t num;
	struct _map_pair *data;
};

struct _map _db_map = {2, (struct _map_pair[]) {
	{"mnist", PI(MNIST), int_dt},
	{"fashion_mnist", PI(FashionMNIST), int_dt}
}};

struct _map _proc_map = {5, (struct _map_pair[]) {
	{"original", PI(proc_original), int_dt},
	{"normalized", PI(proc_normalize), int_dt},
	{"normalised", PI(proc_normalize), int_dt},
	{"binarised", PI(proc_binarize), int_dt},
	{"binarized", PI(proc_binarize), int_dt}
}};

struct _map _act_map = {3, (struct _map_pair[]) {
	{"linear", PI(act_linear), int_dt},
	{"sigmoid", PI(act_sigmoid), int_dt},
	{"relu", PI(act_relu), int_dt}
}};

struct _map _weights_map = {4, (struct _map_pair[]) {
	{"zero", PI(weights_zero), int_dt},
	{"one", PI(weights_one), int_dt},
	{"xavier_normal", PI(weights_xavier_normal), int_dt},
	{"xavier_uniform", PI(weights_xavier_uniform), int_dt}
}};

struct _map *block_map;
struct _map *relax_map;
struct _map *net_map;

cJSON *_config_json;
struct config _config;

cJSON *_parse_file(char *filename);
void _set_relax(struct relaxation_params *params, char *label);
void _set_network(struct network *net, char *label);
struct block *_get_block(char *label);

void _set_enum(cJSON *main_config, char *key, void *setting, struct _map map, int default_val );
void _set_val(cJSON *main_config, char *key, void *setting, void *default_val, enum dtype type);

void _print_network(struct network net, char *level);
void _print_relaxation(struct relaxation_params relax, char *level);

// main code

struct config parse_config(char *filename) {
	_config_json = _parse_file(filename);

	cJSON *general = cJSON_GetObjectItem(_config_json, "general");
	cJSON *blocks = cJSON_GetObjectItem(_config_json, "blocks");
	cJSON *relaxations = cJSON_GetObjectItem(_config_json, "relaxations");
	cJSON *operations = cJSON_GetObjectItem(_config_json, "operations");
	cJSON *networks = cJSON_GetObjectItem(_config_json, "networks");

	if (!general || !blocks || !relaxations || !operations || !networks) {
		printf("[Config][Error] Invalid configuration file\n");
		exit(ERR_INVALID_CONFIG);
	}

	_set_enum(general, "database", &_config.db, _db_map, MNIST);
	_set_val(general, "logging", &_config.logging, PB(true), bool_dt);

	if (cJSON_HasObjectItem(general, "label")) {
		char *gen_label = cJSON_GetObjectItem(general, "label")->valuestring;
		_config.label = malloc(strlen(gen_label)+2);
		sprintf(_config.label, "%s ", gen_label);
	} else {
		// different from previous code
		// previous code set it to "" with length 1
		_config.label = NULL;
	}

	// handle operations
	int arr_count = 0;
	cJSON *elem;
	_config.num_operations = cJSON_GetArraySize(operations);
	_config.operations = malloc(sizeof(struct operation) * _config.num_operations);
	cJSON_ArrayForEach(elem, operations) {
		cJSON *type = cJSON_GetObjectItem(elem, "type");
		if (CMP_VAL(type, "training")) {
			cJSON *op = cJSON_GetObjectItem(elem, "training");

			_config.operations[arr_count].type = op_training;
			struct training *train = &_config.operations[arr_count].training;
			
			cJSON *relax_label = cJSON_GetObjectItem(op, "relaxation");
			cJSON *network_label = cJSON_GetObjectItem(op, "network");

			_set_relax(&train->relax, cJSON_GetStringValue(relax_label));
			_set_network(&train->net, cJSON_GetStringValue(network_label));

			_set_enum(op, "processing", &train->proc, _proc_map, proc_normalize);
			_set_val(op, "train_samples", &train->num_samples, PS(0), size_dt);
			_set_val(op, "seed", &train->seed, PS(0), size_dt);
			_set_val(op, "test_samples_per_iter", &train->test_samples_per_iters, PS(0), size_dt);

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
			cJSON *op = cJSON_GetObjectItem(elem, "testing");
			_config.operations[arr_count].type = op_testing;
			struct testing *test = &_config.operations[arr_count].testing;

			_set_val(op, "num_samples", &test->num_samples, PS(0), size_dt);

			cJSON *net_label = cJSON_GetObjectItem(op, "network");
			cJSON *relax_label = cJSON_GetObjectItem(op, "relaxation");

			_set_network(&test->net, cJSON_GetStringValue(net_label));

			if (relax_label && !cJSON_IsNull(relax_label)) {
				test->relax = true;
				_set_relax(&test->relax_params, cJSON_GetStringValue(relax_label));
			} else {
				test->relax = false;
			}
		} else {
			printf("Invalid operation type specified, skipping...\n");
			continue;
		}
		arr_count++;
	}

	return _config;
}

void print_config() {
	printf("DB: ");
	switch (_config.db) {
		case MNIST: 		printf("MNIST");		break;
		case FashionMNIST:	printf("FashionMNIST");	break;
		default:			printf("Invalid");		break;
	}
	printf("\n");

	printf("num_operations: %ld\n", _config.num_operations);
	for (int i = 0; i < _config.num_operations; i++) {
		printf("Operation %d, type: ", i);
		if (_config.operations[i].type == op_training) {
			printf("Training\n");
			struct training training = _config.operations[i].training;
			printf("\tntargets: %ld\n", training.ntargets);
			printf("\ttargets: [");
			for (int j = 0; j < training.ntargets; j++) 
				printf("%ld ", training.targets[j]);
			printf("]\n");

			printf("\tprocessing: ");
			switch (training.proc) {
				case proc_original: 	printf("None");			break;
				case proc_normalize:	printf("Normalize");	break;
				case proc_binarize:		printf("Binarize");		break;
				default:				printf("Unknown");		break;
			}
			printf("\n");

			printf("\tnum_samples: %ld\n", training.num_samples);
			printf("\tseed: %ld\n", training.seed);
			printf("\ttest_samples_per_iter: %ld\n", training.test_samples_per_iters);
			_print_relaxation(training.relax, "\t");
			printf("\tNetwork\n");
			_print_network(training.net, "\t");
		} else if (_config.operations[i].type == op_testing) {
			printf("Testing\n");
			
			struct testing testing = _config.operations[i].testing;
			printf("\tnum_samples: %ld\n", testing.num_samples);
			printf("\tNetwork\n");
			_print_network(testing.net, "\t");
			if (testing.relax)
				_print_relaxation(testing.relax_params, "\t");
			else
				printf("\tRelaxation\n\t\tNULL\n");
		} else {
			printf("Unknown Operation\n");
		}
	}
}

void free_config(struct config config) {

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

void _set_enum(cJSON *main_config, char *key, void *setting, struct _map map, int default_val ) {
	cJSON *config = cJSON_GetObjectItem(main_config, key);
	if (!config) {
		printf("[Config] %s not specified, defaulting to %d\n", key, default_val);
	} else {
		for (int i = 0; i < map.num; i++) {
			if (CMP_VAL(config, map.data[i].key)) {
				*(int *)setting = *(int *)map.data[i].value;
			}
		}
	}
}

void _set_val(cJSON *main_config, char *key, void *setting, void *default_val, enum dtype type) {
	cJSON *config = cJSON_GetObjectItem(main_config, key);
	if (!config) {
		switch (type) {
			case int_dt:	
				printf("[Config] %s not specified, defaulting to %d\n", key, *(int *)default_val);
				*(int *)setting = *(int *)default_val;
				break;
			case size_dt:	
				printf("[Config] %s not specified, defaulting to %ld\n", key, *(size_t *)default_val);
				*(size_t *)setting = *(size_t *)default_val;
				break;
			case double_dt:	
				printf("[Config] %s not specified, defaulting to %f\n", key, *(double *)default_val);
				*(double *)setting = *(double *)default_val;
				break;
			case bool_dt:	
				printf("[Config] %s not specified, defaulting to %s\n", key, *(bool *)default_val ? "true" : "false");
				*(bool *)setting = *(bool *)default_val;
				break;
			default:
				printf("[Config] %s not specified, invalid type provided, unable to set default value\n", key);
				return;
		}
	} else {
		switch(type) {
			case int_dt:	*(int *)setting = config->valueint;				break;
			case size_dt:	*(size_t *)setting = (size_t)config->valueint;	break;
			case double_dt:	*(double *)setting = config->valuedouble;		break;
			case bool_dt:	*(bool *)setting = cJSON_IsTrue(config);		break;
			default:		printf("Invalid command specify type\n");		break;
		}
	}
}

void _set_relax(struct relaxation_params *params, char *label) {
	cJSON *relax = cJSON_GetObjectItem(_config_json, "relaxations");
	cJSON *elem;
	cJSON_ArrayForEach(elem, relax) {
		if (CMP_VAL(cJSON_GetObjectItem(elem, "label"), label)) {
			_set_val(elem, "gamma", &params->gamma, PS(0.1), double_dt);
			_set_val(elem, "gamma_rate", &params->gamma_rate, PS(0.5), double_dt);
			_set_val(elem, "gamma_count", &params->gamma_count, PS(0), size_dt);
			_set_val(elem, "energy_residual", &params->energy_res, PS(0.0), double_dt);
			_set_val(elem, "max_iterations", &params->max_iters, PS(0), size_dt);
			return;
		}
	}

	// relaxation not found
	printf("[Config][Error] unable to find specified relaxation, please check config\n");
	exit(ERR_INVALID_CONFIG);
}

void _set_network(struct network *net, char *label) {
	cJSON *networks = cJSON_GetObjectItem(_config_json, "networks");
	cJSON *elem;
	cJSON_ArrayForEach(elem, networks) {
		if (CMP_VAL(cJSON_GetObjectItem(elem, "label"), label)) {
			_set_val(elem, "alpha", &net->alpha, PD(0.1), double_dt);
			_set_enum(elem, "activation", &net->act, _act_map, act_sigmoid);
			_set_enum(elem, "weights", &net->weight_init, _weights_map, weights_xavier_uniform);

			net->head = malloc(sizeof(struct block));
			net->head->type = block_layer;
			net->head->layer = malloc(sizeof(struct block_layer));
			net->head->layer->length = 0; // initial value, will be updated in init_network later on
			net->head->prev = NULL;
			net->head->next = NULL;

			cJSON *blocks = cJSON_GetObjectItem(elem, "blocks");
			cJSON *cjson_block = blocks->child;
			struct block *cur_block = net->head;
			// cur_block = _get_block(cJSON_GetStringValue(blocks->child));

			while (cjson_block) {
				struct block *new_block = _get_block(cJSON_GetStringValue(cjson_block));
				new_block->prev = cur_block;
				cur_block->next = new_block;
				// cur_block = _get_block(cJSON_GetStringValue(cjson_block));

				cur_block = new_block;
				cjson_block = cjson_block->next;
			}

			net->tail = malloc(sizeof(struct block));
			net->tail->prev = cur_block;
			net->tail->next = NULL;
			cur_block->next = net->tail;

			net->tail->type = block_layer;
			net->tail->layer = malloc(sizeof(struct block_layer));
			net->tail->layer->length = 0; // initial value, will be updated in init_network later on

		}
	}
}

struct block *_get_block(char *label) {
	struct block *ret = malloc(sizeof(struct block));
	cJSON *blocks = cJSON_GetObjectItem(_config_json, "blocks");
	cJSON *elem;
	cJSON_ArrayForEach(elem, blocks) {
		cJSON *type = cJSON_GetObjectItem(elem, "type");
		if (CMP_VAL(type, "layer")) {
			cJSON *layer = cJSON_GetObjectItem(elem, "layer");
			if (CMP_VAL(layer, "label"), label) {
				ret->type = block_layer;
				ret->layer = malloc(sizeof(struct block_layer));
				_set_val(layer, "length", &ret->layer->length, PS(0), size_dt);

				ret->prev = NULL;
				ret->next = NULL;
				return ret;
			}
		}
	}
}

void _print_relaxation(struct relaxation_params relax, char *level) {
	printf("%sRelaxation\n", level);
	printf("%s\tgamma: %.4f\n", level, relax.gamma);
	printf("%s\tgamma_rate: %.4f\n", level, relax.gamma_rate);
	printf("%s\tgamma_count: %ld\n", level, relax.gamma_count);
	printf("%s\tenergy_res: %.4f\n", level, relax.energy_res);
	printf("%s\tmax_iters: %ld\n", level, relax.max_iters);
}

void _print_network(struct network net, char *level) {
	printf("%s\talpha: %.4f\n", level, net.alpha);

	printf("%s\tactivation: ", level);
	switch (net.act) {
		case act_linear:	printf("Linear");	break;
		case act_sigmoid:	printf("Sigmoid");	break;
		case act_relu:		printf("ReLU");		break;
		default:			printf("Unknonw");	break;
	}
	printf("\n");

	printf("%s\tWeight init: ", level);
	switch (net.weight_init) {
		case weights_zero:				printf("Zeros");			break;
		case weights_one:				printf("Ones");				break;
		case weights_xavier_normal:		printf("Xavier normal");	break;
		case weights_xavier_uniform:	printf("Xavier uniform");	break;
		default:						printf("Unknown");			break;
	}
	printf("\n");

	int index = 0;
	printf("Linked list form:\n");
	struct block *block = net.head;
	while(block) {
		printf("%sblock %d, type: ", level, index);

		if (block->type == block_layer) {
			printf("Layer\n");
			printf("%s\tLength: %ld\n", level, block->layer->length);
		}
		block = block->next;
		index++;
	}
}