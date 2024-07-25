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

int main(int argc, char **argv) {
	int c;
	while ((c = getopt(argc, argv, "c:n:s")) != -1) {
		switch (c) {
			case 'c':
				config = parse_config(optarg);
				config_read = true;
				break;
			case 'n':
				config.params = *load_network(optarg);
				config.should_test = true;
				config.should_train = false;
				break;
			case 's':
				save_net = true;
				break;
		}
	}

	if (!config.should_train && !config.should_test) {
		printf("Nothing to do, exiting,...\n");
		exit(0);
	}

	char results_dir[256];
	time_t cur_time = time(NULL);
	struct tm *cur_tm = localtime(&cur_time);
	strftime(results_dir, 256, "./results/%d-%m-%y_%H:%M:%S", cur_tm);
	mkdir(results_dir, 0700);

	load_mnist(config.params.mnist);

	if (config.should_train) {
		build_network(mnist_get_input_length(), config.params.ntargets, &config.params);

		struct traindata *train_results = train(config.num_samples);
		if (train_results) {
			char trainfile[512];
			sprintf(trainfile, "%s/%s.traindata", results_dir, config.net_name);
			save_traindata(train_results, trainfile);
			free_traindata(train_results);
		}

		if (save_net) {
			printf("Saving\n");
			char netname[512];
			sprintf(netname, "%s/%s", results_dir, config.net_name);
			save_network(netname);
			strcat(netname, ".json");
			save_config(config, netname);
		}
	}

	if (config.should_test) {
		struct testdata *test_results = test();
		
		if (test_results) {
			char testfile[512];
			sprintf(testfile, "%s/%s.testdata", results_dir, config.net_name);
			save_testdata(test_results, testfile);
			free_testdata(test_results);
		}

	}

	free_mnist();
	free_network();
	free(config.net_name);
}