#include <gsl/gsl_matrix.h>

#include "include/mnist.h"

// local header

struct _mnist_set {
	unsigned char **images;
	unsigned char *labels;

	size_t count_total;
	size_t counts[10];
};

struct _mnist {
	struct _mnist_set train;
	struct _mnist_set test;

	size_t img_shape[2];
	size_t img_length;
};

static void _load_mnist_set(struct _mnist_set *set, char *image_filename, char *label_filename);
static gsl_vector *_get_vec(struct _mnist_set set, int index, mnist_processing processing);

// main code

static struct _mnist _mnist;

void load_db(enum db db) {
	switch (db) {
		case MNIST:			load_mnist(mnist_numbers);	break;
		case FashionMNIST:	load_mnist(mnist_fashion);	break;
	}
}

void free_db() {
	free_mnist();
}

void load_mnist(mnist_db type) {
	switch (type) {
		case mnist_numbers:
			printf("Loading MNIST Numbers DB\n");
			_load_mnist_set(&_mnist.train, "database/train-images-idx3-ubyte", "database/train-labels-idx1-ubyte");
			_load_mnist_set(&_mnist.test, "database/t10k-images-idx3-ubyte", "database/t10k-labels-idx1-ubyte");
			break;
		case mnist_fashion:
			printf("Loading MNIST Fashion DB\n");
			_load_mnist_set(&_mnist.train, "database/fashion-train-images-idx3-ubyte", "database/fashion-train-labels-idx1-ubyte");
			_load_mnist_set(&_mnist.test, "database/fashion-t10k-images-idx3-ubyte", "database/fashion-t10k-labels-idx1-ubyte");
			break;
		default:
			printf("Invalid MNIST DB type provided, exiting\n");
			exit(1);
	}
}

mnist_dataset mnist_get_dataset(mnist_set set, int label, mnist_processing processing) {
	struct _mnist_set mnist;
	switch (set) {
		case mnist_train:	mnist = _mnist.train;	break;
		case mnist_test:	mnist = _mnist.test;	break;
		default:
			printf("Invalid MNIST set provided, exiting...\n");
			return (mnist_dataset){};
	}

	int counter = 0;
	// gsl_vector *images[mnist.counts[label]];
	gsl_vector **images = malloc(sizeof(gsl_vector *) * mnist.counts[label]);

	for (int i = 0; i < mnist.count_total; i++) {
		if (mnist.labels[i] == label) {
			images[counter] = _get_vec(mnist, i, processing);
			counter++;

			if (counter == mnist.counts[label])
				break;
		}
	}

	gsl_vector *label_vec = gsl_vector_calloc(10);
	gsl_vector_set(label_vec, label, 1);

	return (mnist_dataset){images, label_vec, label, mnist.counts[label]};
}

size_t mnist_get_count(mnist_set set) {
	switch (set) {
		case mnist_train:	return _mnist.train.count_total;
		case mnist_test:	return _mnist.test.count_total;
		default:			return 0;
	}
}

size_t mnist_get_input_length() {
	return _mnist.img_length;
}

int mnist_get_label(mnist_set set, int index) {
	switch (set) {
		case mnist_train:	return _mnist.train.labels[index];
		case mnist_test:	return _mnist.test.labels[index];
		default: 			return	-1;
	}
}

void free_mnist() {
	for (int i = 0; i < _mnist.train.count_total; i++)
		free(_mnist.train.images[i]);
	
	free(_mnist.train.images);
	free(_mnist.train.labels);

	for (int i = 0; i < _mnist.test.count_total; i++)
		free(_mnist.test.images[i]);

	free(_mnist.test.images);
	free(_mnist.test.labels);
}

void free_mnist_dataset(mnist_dataset data) {
	for (int i = 0; i < data.count; i++) {
		gsl_vector_free(data.images[i]);
	}
	free(data.images);
	
	gsl_vector_free(data.label_vec);
}

// private functions

// load indivial data set (train or test)
static void _load_mnist_set(struct _mnist_set *set, char *image_filename, char *label_filename) {
	FILE *imagesf = fopen(image_filename, "rb");
	FILE *labelsf = fopen(label_filename, "rb");

	if (!imagesf || !labelsf) {
		printf("Error opening MNIST DB File %s (or) %s\n", image_filename, label_filename);
		exit(2);
	}

	int img_header[4];
	int label_header[2];

	fread(&img_header, 4, 4, imagesf);
	fread(&label_header, 4, 2, labelsf);

	for (int i = 0; i < 4; i++) 
		img_header[i] = __builtin_bswap32(img_header[i]);
	
	for (int i = 0; i < 2; i++)
		label_header[i] = __builtin_bswap32(label_header[i]);
	
	set->images = malloc(sizeof(unsigned char *) * img_header[1]);
	set->labels = malloc(sizeof(unsigned char *) * label_header[1]);

	set->count_total = img_header[1];

	_mnist.img_shape[0] = img_header[2];
	_mnist.img_shape[1] = img_header[3];
	_mnist.img_length = _mnist.img_shape[0] * _mnist.img_shape[1];

	for (int index = 0; index < set->count_total; index++) {
		fread(&set->labels[index], 1, 1, labelsf);

		set->images[index] = malloc(sizeof(unsigned char) * _mnist.img_length);
		fread(set->images[index], 1, _mnist.img_length, imagesf);
		set->counts[set->labels[index]]++;
	}

	fclose(imagesf);
	fclose(labelsf);
}

// returns gsl_vector with processed image
static gsl_vector *_get_vec(struct _mnist_set set, int index, mnist_processing processing) {
	// store image in vector
	gsl_vector *ret = gsl_vector_calloc(_mnist.img_length);
	for (int i = 0; i < ret->size; i++)
		gsl_vector_set(ret, i, set.images[index][i]);
	
	// process vector
	switch (processing) {
		case mnist_original:	break;
		case mnist_normalized:	gsl_vector_scale(ret, (float)1/(float)255);	break;
		case mnist_binarised:
			for (int i = 0; i < ret->size; i++)
				gsl_vector_set(ret, i, gsl_vector_get(ret, i) > 128 ? 1 : 0);
			break;
		default:
			printf("ERROR: Invalid MNIST processing specified, returning original vector\n");
			break;
	}

	return ret;
}