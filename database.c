// #include "include/mnist.h"
#include "include/all.h"
#include "include/mnist.h"
#include "include/database.h"

static enum db _db;


void load_db(enum db db) {
	_db = db;
	switch (db) {
		case MNIST:			load_mnist(mnist_numbers);	break;
		case FashionMNIST:	load_mnist(mnist_fashion);	break;
	}
}

db_dataset db_get_dataset(enum db_set set, int label, enum db_proc processing) {
	mnist_set mset;
	mnist_processing mproc;

	switch (set) {
		case db_train:	mset = mnist_train;	break;
		case db_test:	mset = mnist_test;	break;
		default:		mset = mnist_train;	break;
	}

	switch (processing) {
		case proc_original:		mproc = mnist_original;		break;
		case proc_normalize:	mproc = mnist_normalized;	break;
		case proc_binarize:		mproc = mnist_binarised;	break;
		default:				mproc = mnist_normalized;	break;
	}
	mnist_dataset data = mnist_get_dataset(mset, label, mproc);
	return (db_dataset){data.images, data.label_vec, data.label, data.count};
}

size_t db_get_count(enum db_set set) {
	mnist_set mset;
	switch (set) {
		case db_train:	mset = mnist_train;	break;
		case db_test:	mset = mnist_test;	break;
	}

	return mnist_get_count(mset);
}

size_t db_get_input_length() {
	return mnist_get_input_length();
}

struct db_image_info db_get_image_info() {
	return mnist_get_image_info();
}


void free_db() {
	free_mnist();
}

void db_free_dataset(db_dataset data) {
	for (int i = 0; i < data.count; i++)
		gsl_vector_free(data.images[i]);
	
	free(data.images);
	gsl_vector_free(data.label_vec);
}
