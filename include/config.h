#ifndef __CONFIG_H__
#define __CONFIG_H__

// extern struct net_params params;

struct config {
	struct net_params params;

	char *net_name;
	bool should_train;
	bool should_test;

	size_t num_samples;
};

struct config parse_config(char *filename);
void save_config(struct config config, char *filename);
void print_config(struct config config);

#endif