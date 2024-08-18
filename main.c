#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#include <getopt.h>

// #include <gsl/gsl_matrix.h>

#include "include/all.h"
#include "include/config.h"

static bool config_read = false;
struct config config;

static bool save_net = false;

void trial() {
	config = parse_config("config.json");
	print_config();
	exit(0);
}

int main(int argc, char **argv) {
	int c;
	while ((c = getopt(argc, argv, "c:t")) != -1) {
		switch (c) {
			case 'c':
				config = parse_config(optarg);
				config_read = true;
				break;
			case 't':
				trial();
		}
	}

	if (!config_read) {
		config = parse_config("configs/config.json");
		config_read = true;
	}

	if (config.num_operations == 0) {
		printf("No operations specified, exiting,...\n");
		exit(0);
	}

	// print_config(config);

	char results_dir[256];
	time_t cur_time = time(NULL);
	struct tm *cur_tm = localtime(&cur_time);
	char time_str[128];
	strftime(time_str, 256, "%d-%m-%y_%H:%M:%S", cur_tm);
	sprintf(results_dir, "%s/%s %s", config.results_dir, config.label, time_str);
	mkdir(results_dir, 0700);

	load_db(config.db);

	for (int i = 0; i < config.num_operations; i++) {
		if (config.operations[i].type == op_training) {
			struct training *train_params = &config.operations[i].training;
			printf("Starting train operation\n");
			// if seed not provided or equal to zero, seed with current time
			if (train_params->seed == 0)
				train_params->seed = cur_time;
		
			srand(train_params->seed);

			init_network(train_params->net);
			set_network(train_params->net);
			char netname[512];
			sprintf(netname, "%s/%s.net", results_dir, config.operations[i].label);
			save_network(netname);
			struct traindata *train_data = train(*train_params, true);

			char trainfile[512];
			sprintf(trainfile, "%s/%s.traindata", results_dir, config.operations[i].label);
			save_traindata(train_data, trainfile);
			free_traindata(train_data);
			// clear_block_data();
			// struct traindata *traindata = train(config.networks[i], true);
			// save_traindata(config.networks[i]);
			// build network
			// train
			// save network
			// save results 

		} else if (config.operations[i].type == op_testing) {
			set_network(config.operations[i].testing.net);
			struct testdata *test_data = test(config.operations[i].testing, config.logging);
			char testfile[512];
			sprintf(testfile, "%s/%s.testdata", results_dir, config.operations[i].label);

			save_testdata(test_data, testfile);
			free_testdata(test_data);
			// test
			// save results
		}

	}

	char conf_save_file[512];
	sprintf(conf_save_file, "%s/%s.json", results_dir, config.label);
	save_config(conf_save_file);

	printf("Freeing DB\n");
	free_db();
	printf("Freeing config\n");
	free_config(config);
}