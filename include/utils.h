#ifndef __UTILS_H__
#define __UTILS_H__

#define PS(x) &(size_t){x}
#define PB(x) &(bool){x}
#define PI(x) &(int){x}
#define PD(x) &(double){x}

enum dtype {
	int_dt,
	size_dt,
	double_dt,
	bool_dt,
	network_dt
};

void print_vec(gsl_vector *vec, char *title, bool print_index);
void print_img(gsl_vector *vec, char *title);
void print_mat(gsl_matrix *mat, char *title, bool print_index);

gsl_vector *vec_ops(gsl_vector *inp, double(*op)(double));
void vec_ops_inplace(gsl_vector *inp, gsl_vector *out, double(*op)(double));

double frobenius_norm(gsl_matrix *mat);
double mat_dot(gsl_matrix *A, gsl_matrix *B);

void vec2file(gsl_vector *vec, FILE *file);
void mat2file(gsl_matrix *mat, FILE *file);

void save_data(size_t label, enum dtype dtype, void *data, size_t tensor_dim, size_t ndims, size_t *dims, FILE *file);

gsl_matrix *file2mat(FILE *file);

#endif