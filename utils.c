#include <math.h>
#include <stdbool.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include "include/utils.h"

void print_vec(gsl_vector *vec, char *title, bool print_index) {
	printf("########### %s ###########\n", title);
	if (print_index) {
		printf("  ");
		for (int i = 0; i < vec->size; i++)
			printf(" [%3d] ", i);
		printf("\n");
	}
	printf("| ");
	for (int i = 0; i< vec->size; i++)
		printf("%.5f ", gsl_vector_get(vec, i));
	printf("|\n");
}

void print_img(gsl_vector *vec, char *title) {
	printf("########### %s ###########\n", title);
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			double val = gsl_vector_get(vec, 28*i + j);
			printf("%s", val == 0.0 ? " " : ".");
			// if (val == 0.0)
			// 	printf("    ");
			// else
			// 	printf("%.3f ", val);
		}
		printf("\n");
	}
}

gsl_vector *vec_ops(gsl_vector *inp, double(*op)(double)) {
	gsl_vector *ret = gsl_vector_calloc(inp->size);
	for (int i = 0; i < inp->size; i++)
		gsl_vector_set(ret, i, op(gsl_vector_get(inp, i)));
	return ret;
}

double frobenius_norm(gsl_matrix *mat) {
	double ret = 0.0;
	for (int i = 0; i < mat->size1; i++) {
		gsl_vector_view row = gsl_matrix_row(mat, i);
		double temp;
		gsl_blas_ddot(&row.vector, &row.vector, &temp);

		ret += temp;
	}

	return sqrt(ret);
}

void vec2file(gsl_vector *vec, FILE *file) {
	size_t datainfo[2] = {1, vec->size};
	fwrite(datainfo, sizeof(size_t), 2, file);

	gsl_vector_fwrite(file, vec);
}

void _recursive_vec2file(void *data, size_t ndims, size_t *dims, int depth, FILE *file) {
	for (int i = 0; i < dims[depth]; i++) {
		if (depth == ndims-1) {
			vec2file( ((gsl_vector **)data)[i], file);
		} else {
			_recursive_vec2file(( (void **) data)[i],  ndims, dims, depth+1, file);

		}
	}
}

void vecs2file(void *data, size_t type, size_t ndims, size_t *dims, FILE *file) {
	size_t header[2] = {type, ndims};

	fwrite(header, sizeof(size_t), 2, file);
	fwrite(dims, sizeof(size_t), ndims, file);

	_recursive_vec2file(data, ndims, dims, 0, file);
}

gsl_matrix *file2mat(FILE *file) {
	size_t datainfo[2];
	fread(datainfo, sizeof(size_t), 3, file);

	gsl_matrix *ret = gsl_matrix_alloc(datainfo[1], datainfo[2]);
	gsl_matrix_fread(file, ret);

	return ret;
}

void mat2file(gsl_matrix *mat, size_t type, FILE *file) {
	size_t datainfo[2] = {mat->size1, mat->size2};
	fwrite(datainfo, sizeof(size_t), 2, file);

	gsl_matrix_fwrite(file, mat);
}