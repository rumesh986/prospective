#ifndef __MNIST_H__
#define __MNIST_H__

// temporary definition (will move to other file when new db added)
enum db {
	MNIST,
	FashionMNIST
};

enum db_proc {
	proc_original,
	proc_normalize,
	proc_binarize
};

void load_db(enum db db);
void free_db();

typedef enum {
	mnist_numbers,
	mnist_fashion
} mnist_db;

typedef enum {
	mnist_train,
	mnist_test
} mnist_set;

typedef enum {
	mnist_original,
	mnist_normalized,
	mnist_binarised
} mnist_processing;

typedef struct {
	gsl_vector **images;
	gsl_vector *label_vec;

	int label;
	size_t count;
} mnist_dataset;

void load_mnist(mnist_db type);
mnist_dataset mnist_get_dataset(mnist_set set, int label, mnist_processing processing);

size_t mnist_get_count(mnist_set set);
size_t mnist_get_input_length();
int mnist_get_label(mnist_set set, int index);

void free_mnist();
void free_mnist_dataset(mnist_dataset data);

#endif