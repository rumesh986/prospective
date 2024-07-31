#ifndef __CONFIG_H__
#define __CONFIG_H__

// extern struct net_params params;

struct config {
	struct net_params net_params;
	struct training_params train_params;
	struct testing_params test_params;
	struct relaxation_params relax_params;

	char *net_name;
	bool should_train;
	bool should_test;
};

struct config parse_config(char *filename);
void save_config(struct config config, char *filename);
void free_config();
void print_config(struct config config);

#endif