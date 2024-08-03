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
	load_mnist(mnist_numbers);
	free_mnist();
	exit(0);
}

int main(int argc, char **argv) {
	int c;
	while ((c = getopt(argc, argv, "c:st")) != -1) {
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
		config = parse_config("config.json");
		config_read = true;
	}

	if (config.num_operations == 0) {
		printf("No operations specified, exiting,...\n");
		exit(0);
	}

	char results_dir[256];
	time_t cur_time = time(NULL);
	struct tm *cur_tm = localtime(&cur_time);
	char time_str[128];
	strftime(time_str, 256, "%d-%m-%y_%H:%M:%S", cur_tm);
	sprintf(results_dir, "./results/%s%s", config.label, time_str);
	mkdir(results_dir, 0700);

	load_db(config.db);



	for (int i = 0; i < config.num_operations; i++) {
		if (config.operations[i].type == op_training) {
			
			// if seed not provided or equal to zero, seed with current time
			if (config.operations[i].training.params.seed == 0)
				config.operations[i].training.params.seed = cur_time;
		
			srand(config.operations[i].training.params.seed);

			// build network
			// train
			// save network
			// save results 

		} else if (config.operations[i].type == op_testing) {
			// test
			// save results
		}

	}

	free_db();
	// free_network();
	free_config(config);
}