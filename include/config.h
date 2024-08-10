#ifndef __CONFIG_H__
#define __CONFIG_H__

// extern struct net_params params;

enum op_type {
	op_training,
	op_testing
};

struct operation {
	enum op_type type;

	// maybe put this in a union?
	union {
		struct training training;
		struct testing testing;
	};
};

struct config {
	enum db db;

	size_t num_networks;
	struct network **networks;

	size_t num_operations;
	struct operation *operations;

	bool logging;
	char *label;
};

struct config parse_config(char *filename);
void save_config(struct config config, char *filename);
void free_config(struct config config);
void print_config(struct config config);

struct block *new_dense_block(size_t nlayers, size_t *lengths);

#endif