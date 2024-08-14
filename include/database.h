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

struct db_image_info {
	size_t size1;
	size_t size2;
	size_t length;
};

void load_db(enum db db);
db_dataset db_get_dataset(enum db_set set, int label, enum db_proc processing);
size_t db_get_count(enum db_set set);
size_t db_get_input_length();
struct db_image_info db_get_image_info();
void free_db();
void db_free_dataset(db_dataset data);

#endif