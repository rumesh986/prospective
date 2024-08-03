#ifndef __DATABASE_H__
#define __DATABASE_H__

enum db {
	MNIST,
	FashionMNIST
};

enum db_proc {
	proc_original,
	proc_normalize,
	proc_binarize
};

enum db_set {
	db_train,
	db_test
};

typedef struct {
	gsl_vector **images;
	gsl_vector *label_vec;

	int label;
	size_t count;
} db_dataset;

void load_db(enum db db);
db_dataset db_get_dataset(enum db_set set, int label, enum db_proc processing);
size_t db_get_count(enum db_set set);
size_t db_get_input_length();
void free_db();

#endif