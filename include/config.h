#ifndef __CONFIG_H__
#define __CONFIG_H__

enum op_type {
	op_training,
	op_testing
};

struct operation {
	enum op_type type;

	union {
		struct training training;
		struct testing testing;
	};

	// struct network net;
	char *label;
};

struct config {
	enum db db;

	size_t num_operations;
	struct operation *operations;

	bool logging;
	char *label;
};

struct config parse_config(char *filename);
void save_config(char *filename);
void free_config();
void print_config();

// void free_config(struct config config);
#endif