#ifndef __NETWORK_H__
#define __NETWORK_H__

struct net_params {
	mnist_db mnist;
	mnist_processing mnist_proc;

	double gamma;
	double alpha;
	
	size_t tau;
	enum activation act;
	enum weights_init weights;
	
	size_t nlayers;
	size_t *lengths; // must be alloc'd

	size_t ntargets;
	size_t *targets; // must be alloc'd
};

struct lenergy {
	size_t layer;
	gsl_vector *epsilon;
};

struct traindata {
	gsl_vector **delta_w_mags;
	gsl_vector *iter_counts;
	gsl_vector **energies;

	// struct lenergy *lenergies;
	gsl_vector ***lenergies;

	size_t num_samples;
};

struct testdata {
	gsl_matrix *confusion;
	gsl_vector **costs;

	gsl_vector *labels;
	gsl_vector **outputs;

	size_t num_correct;
	size_t num_samples;
};

void build_network(size_t inp_len, size_t out_len, struct net_params *params);
struct net_params *load_network(char *filename);
int save_network(char *filename);
void free_network();

struct traindata *train(size_t num_samples);
int save_traindata(struct traindata *data, char *filename);
void free_traindata(struct traindata *data);

struct testdata *test();
int save_testdata(struct testdata *data, char *filename);
void free_testdata(struct testdata *data);

void trial_network();

#endif