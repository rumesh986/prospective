#include <math.h>
#include <stdbool.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include "include/all.h"
#include "include/utils.h"
#include "include/savefiles.h"

// local header

void _recursive_tensor2file(void *data, size_t tensor_dim, size_t ndims, size_t *dims, int depth, FILE *file);

// main code

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

void print_mat(gsl_matrix *mat, char *title, bool print_index) {
	printf("########### %s ###########\n", title);
	if (print_index) {
		printf("    ");
		for (int j = 0; j < mat->size2; j++)
			printf(" [%3d] ", j);
		printf("\n");
	}
	for (int i = 0; i< mat->size1; i++) {
		if (print_index)
			printf("[%3d] |", i);
		for (int j = 0; j < mat->size2; j++) 
			printf(" %.1f ", gsl_matrix_get(mat, i, j));
		printf("|\n");
	}
}


gsl_vector *vec_ops(gsl_vector *inp, double(*op)(double)) {
	gsl_vector *ret = gsl_vector_calloc(inp->size);
	for (int i = 0; i < inp->size; i++)
		gsl_vector_set(ret, i, op(gsl_vector_get(inp, i)));
	return ret;
}
void vec_ops_inplace(gsl_vector *inp, gsl_vector *out, double(*op)(double)) {
	if (inp->size != out->size) {
		printf("[Error] vector sizes dont match");
		exit(ERR_VEC_OPS);
	}

	for (int i = 0; i < inp->size; i++)
		gsl_vector_set(out, i, op(gsl_vector_get(inp, i)));
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

double mat_dot(gsl_matrix *A, gsl_matrix *B) {
	gsl_matrix *temp = gsl_matrix_calloc(A->size1, A->size2);
	gsl_matrix_memcpy(temp, A);
	gsl_matrix_mul_elements(temp, B);

	gsl_vector_view temp_vec = gsl_vector_view_array(temp->data, A->size1 * A->size2);

	double ret = gsl_vector_sum(&temp_vec.vector);
	gsl_matrix_free(temp);

	return ret;
}

void vec2file(gsl_vector *vec, FILE *file) {
	size_t datainfo[2] = {1, vec->size};
	fwrite(datainfo, sizeof(size_t), 2, file);

	gsl_vector_fwrite(file, vec);
}

void mat2file(gsl_matrix *mat, FILE *file) {
	size_t datainfo[2] = {mat->size1, mat->size2};
	fwrite(datainfo, sizeof(size_t), 2, file);

	gsl_matrix_fwrite(file, mat);
}

void save_data(size_t label, enum dtype dtype, void *data, size_t tensor_dim, size_t ndims, size_t *dims, FILE *file) {
	// no data to write exit without error
	if (!data)
		return;

	// write array of scalars information needed to read rest of system essentially
	if (label == SAVE_ARRAY) {
		if (tensor_dim != 0) {
			printf("Possible error in saving data (non-scalar array specified), skipping. Please check code\n");
			return;
		}

		for (int i = 0; i < ndims; i++) {
			size_t label_i = *((size_t *)data + 3*i);
			size_t dtype_i = *((size_t *)data + 3*i + 1);
			void *data_i;
			switch (dtype_i) {
				case size_dt:	data_i = ((size_t *)data + 3*i + 2);	break;
				case double_dt:	data_i = ((double *)data + 3*i + 2);	break;
				default:		data_i = ((size_t *)data + 3*i + 2);	break;
			}

			if (label_i == SAVE_ALPHA) {
				printf("PASSING alpha: dtype == %lx, data_i = %p\n", dtype_i, data + 3*i + 2);
			}

			save_data(label_i, dtype_i, data_i, tensor_dim, 1, NULL, file);
		}
		return;
	}

	size_t header[4] = {label, dtype, tensor_dim, ndims};
	fwrite(header, sizeof(size_t), 4, file);

	if (tensor_dim == 0) {
		fwrite(data, sizeof(size_t), 1, file);
		return;
	}

	fwrite(dims, sizeof(size_t), ndims, file);

	_recursive_tensor2file(data, tensor_dim, ndims, dims, 0, file);
}


gsl_matrix *file2mat(FILE *file) {
	size_t datainfo[2];
	fread(datainfo, sizeof(size_t), 3, file);

	gsl_matrix *ret = gsl_matrix_alloc(datainfo[1], datainfo[2]);
	gsl_matrix_fread(file, ret);

	return ret;
}

// private functions
void _recursive_tensor2file(void *data, size_t tensor_dim, size_t ndims, size_t *dims, int depth, FILE *file) {
	for (int i = 0; i < dims[depth]; i++) {
		if (depth == ndims-1) {
			void *data_ptr = dims[depth] > 1 ? ((void **)data)[i] : data;
			switch (tensor_dim) {
				case 1:	vec2file((gsl_vector *) data_ptr, file);	break;
				case 2: mat2file((gsl_matrix *) data_ptr, file);	break;
			}
		} else {
			_recursive_tensor2file(((void **)data)[i], tensor_dim, ndims, dims, depth+1, file);
		}
	}
}