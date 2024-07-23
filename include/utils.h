#ifndef __UTILS_H__
#define __UTILS_H__

void print_vec(gsl_vector *vec, char *title, bool print_index);

gsl_vector *vec_ops(gsl_vector *inp, double(*op)(double));

double frobenius_norm(gsl_matrix *mat);

void vec2file(gsl_vector *vec, FILE *file);

gsl_matrix *file2mat(FILE *file);
void mat2file(gsl_matrix *mat, FILE *file);

#endif