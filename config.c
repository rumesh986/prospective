#include <string.h>

#include <cjson/cJSON.h>

#include "include/all.h"
#include "include/config.h"

// local header

#define CMP_STR(a, b) strcmp(a, b) == 0
#define CMP_VAL(a, b) CMP_STR(a->valuestring, b)

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

struct _map _poolt_map = {3, (struct _map_pair[]) {
	{"max", PI(pool_max), int_dt},
	{"average", PI(pool_avg), int_dt},
	{"avg", PI(pool_avg), int_dt}
}};

struct _map net_map = {0, NULL};

cJSON *_config_json;
struct config _config;

cJSON *_parse_file(char *filename);
void _set_relax(struct relaxation_params *params, char *label);
struct network *_set_network(char *net_label, char *op_label);
struct network *_get_network(char *label);
struct block *_get_block(char *label);

void _set_enum(cJSON *main_config, char *key, void *setting, struct _map map, int default_val );
void _set_val(cJSON *main_config, char *key, void *setting, void *default_val, enum dtype type);

void _print_network(struct network net, char *level);
void _print_relaxation(struct relaxation_params relax, char *level);

void *_search_map(struct _map map, char *label);
void _append_map(struct _map *map, char *label, void *obj, enum dtype type);

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

	cJSON *label = cJSON_GetObjectItem(general, "label");
	if (!label) {
		printf("[Error][Config] config missing label. exiting...\n");
		exit(ERR_INVALID_CONFIG);
	}

	_config.label = malloc(strlen(cJSON_GetStringValue(label))+1);
	sprintf(_config.label, "%s", cJSON_GetStringValue(label));

	cJSON *results_dir = cJSON_GetObjectItem(general, "results_dir");
	if (results_dir && !cJSON_IsNull(results_dir)) {
		_config.results_dir = malloc(strlen(cJSON_GetStringValue(results_dir))+1);
		sprintf(_config.results_dir, "%s", cJSON_GetStringValue(results_dir));
	} else {
		_config.results_dir = malloc(11);
		sprintf(_config.results_dir, "./results");
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
			cJSON *label = cJSON_GetObjectItem(op, "label");

			if (!label) {
				printf("[Error][Config] Operation requires a label, exiting...\n");
				exit(ERR_INVALID_CONFIG);
			}

			_config.operations[arr_count].label = malloc(strlen(cJSON_GetStringValue(label))+1);
			strcpy(_config.operations[arr_count].label, cJSON_GetStringValue(label));

			_config.operations[arr_count].type = op_training;
			struct training *train = &_config.operations[arr_count].training;
			
			cJSON *relax_label = cJSON_GetObjectItem(op, "relaxation");
			cJSON *network_label = cJSON_GetObjectItem(op, "network");

			_set_relax(&train->relax, cJSON_GetStringValue(relax_label));
			train->net = _set_network(cJSON_GetStringValue(network_label), cJSON_GetStringValue(label));

			_set_val(op, "train_samples", &train->num_samples, PS(0), size_dt);
			_set_val(op, "seed", &train->seed, PS(0), size_dt);
			_set_val(op, "test_samples_per_iter", &train->test_samples_per_iters, PS(0), size_dt);

		} else if (CMP_VAL(type, "testing")) {
			cJSON *op = cJSON_GetObjectItem(elem, "testing");
			cJSON *label = cJSON_GetObjectItem(op, "label");

			if (!label) {
				printf("[Error][Config] Operation requires a label, exiting...\n");
				exit(ERR_INVALID_CONFIG);
			}

			_config.operations[arr_count].label = malloc(strlen(cJSON_GetStringValue(label))+1);
			strcpy(_config.operations[arr_count].label, cJSON_GetStringValue(label));

			_config.operations[arr_count].type = op_testing;
			struct testing *test = &_config.operations[arr_count].testing;

			_set_val(op, "num_samples", &test->num_samples, PS(0), size_dt);

			cJSON *net_label = cJSON_GetObjectItem(op, "network");
			cJSON *relax_label = cJSON_GetObjectItem(op, "relaxation");

			test->net = _get_network(cJSON_GetStringValue(net_label));

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
	printf("trial for wtf is going\n");
	printf("DB: ");
	printf("on like seriously\n");
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

			printf("\tnum_samples: %ld\n", training.num_samples);
			printf("\tseed: %ld\n", training.seed);
			printf("\ttest_samples_per_iter: %ld\n", training.test_samples_per_iters);
			_print_relaxation(training.relax, "\t");
			printf("\tNetwork\n");
			_print_network(*training.net, "\t");
		} else if (_config.operations[i].type == op_testing) {
			printf("Testing\n");
			
			struct testing testing = _config.operations[i].testing;
			printf("\tnum_samples: %ld\n", testing.num_samples);
			printf("\tNetwork\n");
			_print_network(*testing.net, "\t");
			if (testing.relax)
				_print_relaxation(testing.relax_params, "\t");
			else
				printf("\tRelaxation\n\t\tNULL\n");
		} else {
			printf("Unknown Operation\n");
		}
	}
}

void save_config(char *filename) {
	printf("Saving configuration file\n");
	cJSON *elem;
	cJSON *networks = cJSON_GetObjectItem(_config_json, "networks");
	for (int i = 0; i < net_map.num; i++) {
		cJSON_ArrayForEach(elem, networks) {
			if (CMP_VAL(cJSON_GetObjectItem(elem, "label"), net_map.data[i].key)) {
				struct network *net = (struct network *) net_map.data[i].value;
				cJSON_AddNumberToObject(elem, "input_length", net->head->length);
				cJSON_AddNumberToObject(elem, "output_length", net->tail->length);
			}
		}
	}

	cJSON *ops = cJSON_GetObjectItem(_config_json, "operations");
	for (int i = 0; i < _config.num_operations; i++) {
		cJSON_ArrayForEach(elem, ops) {
			if (CMP_VAL(cJSON_GetObjectItem(elem, "type"), "training")) {
				cJSON *op = cJSON_GetObjectItem(elem, "training");
				if (_config.operations[i].label && CMP_VAL(cJSON_GetObjectItem(op, "label"), _config.operations[i].label)) {
					cJSON *seed = cJSON_GetObjectItem(op, "seed");
					if (cJSON_GetNumberValue(seed) == 0)
						cJSON_SetNumberValue(seed, _config.operations[i].training.seed);

				}
			}
		}
	}

	char *config_str = cJSON_Print(_config_json);
	FILE *file = fopen(filename, "w");
	if (!file) {
		printf("[Error][Config] Failed to open file to save config\n");
		return;
	}

	fwrite(config_str, 1, strlen(config_str), file);
	fclose(file);
	free(config_str);
	printf("Finished saving configuration file\n");
}

void free_config() {
	for (int i = 0; i < net_map.num; i++)
		free_network((struct network *)net_map.data[i].value);
	free(net_map.data);

	free(_config.label);
	free(_config.results_dir);
	
	for (int i = 0; i < _config.num_operations; i++)
		free(_config.operations[i].label);

	free(_config.operations);

	cJSON_Delete(_config_json);
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

struct network * _get_network(char *label) {
	// check if network has already been built
	struct network *existing_net = _search_map(net_map, label);
	if (existing_net)
		return existing_net;
	
	printf("Unable to find network, check config. exiting...\n");
	exit(ERR_INVALID_CONFIG);
}

struct network *_set_network(char *net_label, char *op_label) {
	cJSON *networks = cJSON_GetObjectItem(_config_json, "networks");
	cJSON *elem;
	cJSON_ArrayForEach(elem, networks) {
		if (CMP_VAL(cJSON_GetObjectItem(elem, "label"), net_label)) {
			struct network *net = malloc(sizeof(struct network));
			_set_val(elem, "alpha", &net->alpha, PD(0.1), double_dt);
			_set_enum(elem, "activation", &net->act, _act_map, act_sigmoid);
			_set_enum(elem, "weights", &net->weight_init, _weights_map, weights_xavier_uniform);
			_set_enum(elem, "processing", &net->proc, _proc_map, proc_normalize);

			cJSON *targets = cJSON_GetObjectItem(elem, "targets");
			if (!targets || cJSON_IsNull(targets)) {
				printf("Targets not specified, assuming all\n");
				net->ntargets = 10;
				net->targets = malloc(sizeof(size_t) * net->ntargets);
				for (int i = 0; i < net->ntargets; i++)
					net->targets[i] = i;
			} else {
				net->ntargets = cJSON_GetArraySize(targets);
				net->targets = malloc(sizeof(size_t) * net->ntargets);
				for (int i = 0; i < net->ntargets; i++)
					net->targets[i] = cJSON_GetArrayItem(targets, i)->valueint;
			}

			net->head = malloc(sizeof(struct block));
			net->head->type = block_layer;
			net->head->blayer = malloc(sizeof(struct block_layer));
			net->head->length = 0; // initial value, will be updated in init_network later on
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

				cur_block = new_block;
				cjson_block = cjson_block->next;
			}

			net->tail = malloc(sizeof(struct block));
			net->tail->prev = cur_block;
			net->tail->next = NULL;
			cur_block->next = net->tail;

			net->tail->type = block_layer;
			net->tail->blayer = malloc(sizeof(struct block_layer));
			net->tail->length = net->ntargets;

			_append_map(&net_map, op_label, net, network_dt);
			return net;
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
			if (CMP_VAL(cJSON_GetObjectItem(layer, "label"), label)) {
				ret->type = block_layer;
				ret->blayer = malloc(sizeof(struct block_layer));
				_set_val(layer, "length", &ret->length, PS(0), size_dt);

				ret->prev = NULL;
				ret->next = NULL;
				return ret;
			}
		} else if (CMP_VAL(type, "cnn")) {
			cJSON *cnn = cJSON_GetObjectItem(elem, "cnn");
			if (CMP_VAL(cJSON_GetObjectItem(cnn, "label"), label)) {
				ret->type = block_cnn;
				ret->cnn = malloc(sizeof(struct block_cnn));
				_set_val(cnn, "kernel_size", &ret->cnn->kernel_size, PS(3), size_dt);
				_set_val(cnn, "stride", &ret->cnn->stride, PS(1), size_dt);
				_set_val(cnn, "padding", &ret->cnn->padding, PS(0), size_dt);
				_set_val(cnn, "nchannels", &ret->cnn->nchannels, PS(1), size_dt);
				
				cJSON *pool_cfg = cJSON_GetObjectItem(cnn, "pool");
				if (pool_cfg) {
					_set_val(pool_cfg, "size", &ret->cnn->pool_size, PS(2), size_dt);
					_set_enum(pool_cfg, "type", &ret->cnn->pool_type, _poolt_map, pool_max);
				} else {
					printf("CNN pool information, assuming max pooling of size 2\n");
					ret->cnn->pool_size = 2;
					ret->cnn->pool_type = pool_max;
				}

				return ret;
			}
		}
	}

	printf("block %s not found, check config file\n", label);
	exit(ERR_INVALID_CONFIG);
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

	printf("\tprocessing: ");
			switch (net.proc) {
				case proc_original: 	printf("None");			break;
				case proc_normalize:	printf("Normalize");	break;
				case proc_binarize:		printf("Binarize");		break;
				default:				printf("Unknown");		break;
			}
			printf("\n");

	printf("\tntargets: %ld\n", net.ntargets);
	printf("\ttargets: [");
	for (int j = 0; j < net.ntargets; j++) 
		printf("%ld ", net.targets[j]);
	printf("]\n");

	int index = 0;
	struct block *block = net.head;
	while(block) {
		printf("%sblock %d, type: ", level, index);

		if (block->type == block_layer) {
			printf("Layer\n");
			printf("%s\tLength: %ld\n", level, block->length);
		} else if (block->type == block_cnn) {
			printf("CNN\n");
			printf("%s\tkernel size: %ld\n", level, block->cnn->kernel_size);
			printf("%s\tstride: %ld\n", level, block->cnn->stride);
			printf("%s\tpadding: %ld\n", level, block->cnn->padding);
		}
		block = block->next;
		index++;
	}
}

void *_search_map(struct _map map, char *label) {
	for (int i = 0; i < map.num; i++) {
		if (CMP_STR(map.data[i].key, label))
			return map.data[i].value;
	}
	return NULL;
}

void _append_map(struct _map *map, char *label, void *obj, enum dtype type) {
	map->num++;

	struct _map_pair *new_ptr = realloc(map->data, sizeof(struct _map_pair) * map->num);
	if (!new_ptr) {
		printf("[Error] Unable to realloc network map, exiting\n");
		exit(ERR_MEM);
	} else {
		map->data = new_ptr;
	}

	map->data[map->num-1] = (struct _map_pair){label, obj, type};
}