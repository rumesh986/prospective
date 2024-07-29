#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

#include <cjson/cJSON.h>

#include "include/all.h"
// #include "include/network.h"
// #include "include/mnist.h"
#include "include/config.h"

// local header

#define CMP_VAL(a, b) strcmp(a->valuestring, b) == 0

enum dtype {
	int_t,
	size_tt,
	double_tt
};

struct config_option {
	char *key;
	int value;
	// enum dtype type;
};

cJSON *_config;

cJSON *_parse_file(char *filename);
void set_enum(cJSON *main_config, char *key, int *setting, size_t num_options, struct config_option *options, int default_val );
void set_val(cJSON *main_config, char *key, void *setting, void * default_val, enum dtype type);

// main code

struct config parse_config(char *filename) {
	_config = _parse_file(filename);
	
	// struct net_params *ret = malloc(sizeof(struct net_params));
	struct config ret;

	cJSON *mnist_configs = cJSON_GetObjectItem(_config, "mnist");
	cJSON *net_configs = cJSON_GetObjectItem(_config, "network");
	cJSON *oper_configs = cJSON_GetObjectItem(_config, "operations");
	cJSON *train_configs = cJSON_GetObjectItem(_config, "training");
	cJSON *early_stop_configs = cJSON_GetObjectItem(train_configs, "early_stop");
	cJSON *name_config = cJSON_GetObjectItem(_config, "name");

	struct config_option mnist_options[2] = {
		{"number", mnist_numbers},
		{"fashion", mnist_fashion}
	};
	set_enum(mnist_configs, "type", (int *)&ret.net_params.mnist, 2, mnist_options, mnist_numbers);

	struct config_option mnist_proc_options[5] = {
		{"original", mnist_original},
		{"normalized", mnist_normalized},
		{"normalised", mnist_normalized},
		{"binarised", mnist_binarised},
		{"binarized", mnist_binarised}
	};
	set_enum(mnist_configs, "processing", (int *)&ret.net_params.mnist_proc, 3, mnist_proc_options, mnist_original);

	set_val(net_configs, "alpha", &ret.net_params.alpha, &(double){0.05}, double_tt);
	set_val(net_configs, "gamma", &ret.net_params.gamma, &(double){0.1}, double_tt);
	set_val(net_configs, "tau", &ret.net_params.tau, &(size_t){128}, size_tt);

	struct config_option act_options[3] = {
		{"linear", act_linear},
		{"sigmoid", act_sigmoid},
		{"relu", act_relu}
	};
	set_enum(net_configs, "activation", (int *)&ret.net_params.act, 3, act_options, act_linear);
	
	struct config_option weights_options[4] = {
		{"zero", weights_zero},
		{"one", weights_one},
		{"xavier_normal", weights_xavier_normal},
		{"xavier_uniform", weights_xavier_uniform}
	};
	set_enum(net_configs, "weights", (int *)&ret.net_params.weights, 4, weights_options, weights_xavier_uniform);

	cJSON *layers_config = cJSON_GetObjectItem(net_configs, "layers");
	if (!layers_config) {
		printf("[Config] Layers not specified, exiting...");
		exit(3);
	}
	ret.net_params.nlayers = cJSON_GetArraySize(layers_config);
	ret.net_params.lengths = malloc(sizeof(size_t) * ret.net_params.nlayers);
	for (int i = 0; i < ret.net_params.nlayers; i++)
		ret.net_params.lengths[i] = cJSON_GetArrayItem(layers_config, i)->valueint;
	
	cJSON *targets_config = cJSON_GetObjectItem(net_configs, "targets");
	if (!targets_config) {
		printf("[Config] Layers not specified, exiting...");
		exit(3);
	}
	ret.net_params.ntargets = cJSON_GetArraySize(targets_config);
	ret.net_params.targets = malloc(sizeof(size_t) * ret.net_params.ntargets);
	for (int i = 0; i < ret.net_params.ntargets; i++)
		ret.net_params.targets[i] = cJSON_GetArrayItem(targets_config, i)->valueint;

	if (!oper_configs || cJSON_GetArraySize(oper_configs) == 0) {
		printf("[Config] No operations specified, defaulting to train\n");
		bool should_train = true;
	} else {
		cJSON *train_config = cJSON_GetObjectItem(oper_configs, "train");
		if (cJSON_IsBool(train_config)) {
			ret.should_train = cJSON_IsTrue(train_config);
		}

		cJSON *test_config = cJSON_GetObjectItem(oper_configs, "test");
		if (cJSON_IsBool(test_config)) {
			ret.should_test = cJSON_IsTrue(test_config);
		}
	}

	set_val(train_configs, "num_samples", &ret.train_params.num_samples, &(size_t){0}, size_tt);
	set_val(train_configs, "seed", &ret.train_params.seed, &(size_t){0}, size_tt);
	set_val(train_configs, "gamma_rate", &ret.train_params.gamma_rate, &(double){0}, double_tt);

	if (early_stop_configs) {
		set_val(early_stop_configs, "gamma_count", &ret.train_params.gamma_count, &(size_t){0}, size_tt);
		set_val(early_stop_configs, "energy_residual", &ret.train_params.energy_res, &(double){0}, double_tt);
	}

	if (name_config) {
		// char *namestr = name_config->valuestring;
		ret.net_name = malloc(strlen(name_config->valuestring)+1);
		strcpy(ret.net_name, name_config->valuestring);
	} else {
		char tempstr[512];

		char db[4];
		switch (ret.net_params.mnist) {
			case mnist_numbers:	strcpy(db, "num");	break;
			case mnist_fashion:	strcpy(db, "fas");	break;
			default:			strcpy(db, "err");	break;
		}

		char act[4];
		switch (ret.net_params.act) {
			case act_linear: 	strcpy(act, "lin");	break;
			case act_sigmoid:	strcpy(act, "sig");	break;
			case act_relu:		strcpy(act, "rel");	break;
			default:			strcpy(act, "err");	break;
		}

		size_t targets = 0;
		for (int i = 0; i < ret.net_params.ntargets; i++)
			targets += ret.net_params.targets[i] * pow(10, ret.net_params.ntargets - i - 1);
		sprintf(tempstr, "net_%s_a%s_%ldx%ld_t%ld", db, act, ret.net_params.nlayers, ret.net_params.lengths[0], targets);

		ret.net_name = malloc(strlen(tempstr)+1);
		strcpy(ret.net_name, tempstr);
	}

	return ret;
}

void free_config() {
	cJSON_Delete(_config);
}

void save_config(char *filename) {
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
	printf("MNIST DB: ");
	switch (config.net_params.mnist) {
		case mnist_numbers: printf("numbers\n");	break;
		case mnist_fashion: printf("fashion\n");	break;
		default: 			printf("ERROR\n");		break;
	}

	printf("MNIST Processing: ");
	switch (config.net_params.mnist_proc) {
		case mnist_original:	printf("original\n");	break;
		case mnist_normalized:	printf("normalized\n");	break;
		case mnist_binarised:	printf("binarised\n");	break;
		default:				printf("ERROR\n");		break;
	}

	printf("NET alpha: %f\n", config.net_params.alpha);
	printf("NET gamma: %f\n", config.net_params.gamma);
	printf("NET tau: %ld\n", config.net_params.tau);
	printf("NET activation: ");
	switch (config.net_params.act) {
		case act_linear:	printf("Linear\n");		break;
		case act_sigmoid:	printf("Sigmoid\n");	break;
		case act_relu:		printf("ReLu\n");		break;
		default:			printf("ERROR\n");		break;
	}

	printf("NET weights: ");
	switch (config.net_params.weights) {
		case weights_zero:				printf("zero\n");			break;
		case weights_one:				printf("one\n");			break;
		case weights_xavier_normal:		printf("xavier normal\n");	break;
		case weights_xavier_uniform:	printf("xavier uniform\n");	break;
	}

	printf("NET nlayers: %ld [\n", config.net_params.nlayers);
	for (int i = 0; i < config.net_params.nlayers; i++)
		printf("%ld, \n", config.net_params.lengths[i]);
	printf("]\n");

	printf("NET ntargets: %ld [", config.net_params.ntargets);
	for (int i = 0; i < config.net_params.ntargets; i++)
		printf("%ld, ", config.net_params.targets[i]);
	printf("]\n");

	printf("Config name: %s\n", config.net_name);
	printf("Config train: %s\n", config.should_train ? "True" : "False");
	printf("Config test: %s\n", config.should_test ? "True" : "False");
	printf("Config num samples: %ld\n", config.train_params.num_samples);
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
		} else {
			printf("[Config] %s not specified, invalid type provided, unable to set default value\n", key);
			return;
		}
	} else {
		if (type == int_t)
			*(int *)setting = config->valueint;
		else if (type == size_tt)
			*(size_t *)setting = (size_t)config->valueint;
		else if (double_tt)
			*(double *)setting = config->valuedouble;
		else {
			printf("Invalid command specify type\n");
		}
	}
}

void save_config_old(struct config config, char *filename) {
	cJSON *final = cJSON_CreateObject();
	cJSON *mnist = cJSON_CreateObject();
	cJSON *net = cJSON_CreateObject();
	cJSON *operations = cJSON_CreateObject();
	cJSON *training = cJSON_CreateObject();
	cJSON *early_stop = cJSON_CreateObject();

	char str[128];
	switch (config.net_params.mnist) {
		case mnist_numbers: strcpy(str, "number");	break;
		case mnist_fashion:	strcpy(str, "fashion");	break;
	}
	cJSON_AddStringToObject(mnist, "type", str);

	switch (config.net_params.mnist_proc) {
		case mnist_original: 	strcpy(str, "original");	break;
		case mnist_normalized:	strcpy(str, "normalized");	break;
		case mnist_binarised:	strcpy(str, "binarised");	break;
	}
	cJSON_AddStringToObject(mnist, "processing", str);

	cJSON_AddItemToObject(final, "mnist", mnist);

	cJSON_AddNumberToObject(net, "alpha", config.net_params.alpha);
	cJSON_AddNumberToObject(net, "gamma", config.net_params.gamma);
	cJSON_AddNumberToObject(net, "tau", config.net_params.tau);

	switch (config.net_params.act) {
		case act_linear: 	strcpy(str, "linear");	break;
		case act_sigmoid:	strcpy(str, "sigmoid");	break;
		case act_relu:	strcpy(str, "relu");	break;
	}
	cJSON_AddStringToObject(net, "activation", str);

	switch (config.net_params.weights) {
		case weights_zero:				strcpy(str, "zero");			break;
		case weights_one:				strcpy(str, "one");				break;
		case weights_xavier_normal:		strcpy(str, "xavier_normal");	break;
		case weights_xavier_uniform:	strcpy(str, "xavier_uniform");	break;
	}
	cJSON_AddStringToObject(net, "weights", str);

	cJSON *layers = cJSON_AddArrayToObject(net, "layers");
	for (int i = 0; i < config.net_params.nlayers; i++) {
		cJSON *temp = cJSON_CreateNumber(config.net_params.lengths[i]);
		cJSON_AddItemToArray(layers, temp);
	}

	cJSON *targets = cJSON_AddArrayToObject(net, "targets");
	for (int i = 0; i < config.net_params.ntargets; i++) {
		cJSON *temp = cJSON_CreateNumber(config.net_params.targets[i]);
		cJSON_AddItemToArray(targets, temp);
	}

	cJSON_AddItemToObject(final, "network", net);

	cJSON_AddBoolToObject(operations, "train", config.should_train);
	cJSON_AddBoolToObject(operations, "test", config.should_test);

	cJSON_AddItemToObject(final, "operations", operations);

	cJSON_AddNumberToObject(training, "num_samples", config.train_params.num_samples);
	cJSON_AddNumberToObject(training, "seed", config.train_params.seed);
	cJSON_AddNumberToObject(training, "gamma_rate", config.train_params.gamma_rate);
	
	cJSON_AddNumberToObject(early_stop, "gamma_count", config.train_params.gamma_count);
	cJSON_AddNumberToObject(early_stop, "energy_residual", config.train_params.energy_res);
	cJSON_AddItemToObject(training, "early_stop", early_stop);
	
	cJSON_AddItemToObject(final, "training", training);


	// cJSON *namestr = cJSON_CreateString(config.net_name);
	// cJSON_AddItemToObject(final, "name", namestr);
	cJSON_AddStringToObject(final, "name", config.net_name);

	char *print = cJSON_Print(final);
	FILE *file = fopen(filename, "w");
	if (!file) {
		printf("[Config] Error failed to open file to save config (%s)\n", filename);
		return;
	} 

	fwrite(print, 1, strlen(print), file);

	free(print);
	cJSON_Delete(final);
	fclose(file);
}