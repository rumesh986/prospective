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
	size_t sizeinfo[2] = {1, vec->size};
	fwrite(sizeinfo, sizeof(size_t), 2, file);

	gsl_vector_fwrite(file, vec);
}

gsl_matrix *file2mat(FILE *file) {
	size_t sizeinfo[2];
	fread(sizeinfo, sizeof(size_t), 2, file);

	gsl_matrix *ret = gsl_matrix_alloc(sizeinfo[0], sizeinfo[1]);
	gsl_matrix_fread(file, ret);

	return ret;
}

void mat2file(gsl_matrix *mat, FILE *file) {
	size_t sizeinfo[2] = {mat->size1, mat->size2};
	fwrite(sizeinfo, sizeof(size_t), 2, file);

	gsl_matrix_fwrite(file, mat);
}