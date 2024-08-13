#ifndef __NETWORK_H__
#define __NETWORK_H__

struct relaxation_params {
	double gamma;
	double gamma_rate;

	size_t gamma_count;
	double energy_res;
	size_t max_iters;
};

// type of block
// needed for eventual support of CNNs
enum block_t {
	block_layer
};

// abstract block contains pointer to a specific block and its type
struct block {
	enum block_t type;

	union {
		struct dense_block *dense;
		struct block_layer *layer;
	};

	struct block *prev;
	struct block *next;
};

// collection of blocks
struct network {
	double alpha;
	enum activation act;
	enum weights_init weight_init;

	struct block *head; // points to input layer, initial value is 0, real value set in init_network
	struct block *tail; // points to output layer, initial value is 0, real value set in init_network

	size_t nlayers;
	size_t ntargets;
	size_t *targets;

	size_t lenergy_chunks;

	bool save;
};

struct block_layer {
	size_t length;

	gsl_vector *layer;
	gsl_vector *act;	// used to calculate activated versions or derivative of activation versions
	gsl_vector *epsilon;
	gsl_vector *out;

	gsl_matrix *weights;

	gsl_vector *deltax;
	gsl_matrix *deltaw;

	gsl_vector *epsilon2;	// used when calculating energies, without disturbing epsilon

	double **energies;
	double *deltaw_mags;
};

struct training {
	struct relaxation_params relax;
	struct network *net;

	enum db_proc proc;
	size_t num_samples;
	size_t seed;
	size_t test_samples_per_iters;
};

struct testing {
	bool relax;
	struct relaxation_params relax_params;
	struct network *net;

	size_t num_samples;
};

// back to old stuff
struct traindata {
	// double **delta_w_mags;
	size_t * iter_counts;

	// double ***lenergies;
	double *train_costs;

	size_t num_samples;
};

struct testdata {
	gsl_matrix *confusion;
	gsl_vector **costs;

	gsl_vector *labels;
	gsl_vector *predictions;
	gsl_vector **outputs;

	gsl_vector **energies;
	gsl_vector ***lenergies;
	gsl_vector *iter_counts;

	size_t num_correct;
	size_t num_samples;
};

void init_network(struct network *net);
void set_network(struct network *net);
struct traindata *train(struct training train, bool logging);
void save_traindata(struct traindata *data, char *filename);
void free_network(struct network *net);

// struct net_params *load_network(char *filename);
// int save_network(char *filename);
// void free_network();

// struct traindata *train(struct training_params train_params);
// int save_traindata(struct traindata *data, char *filename);
// void free_traindata(struct traindata *data);

// struct testdata *test(struct testing_params test_params, bool relaxation);
// int save_testdata(struct testdata *data, char *filename);
// void free_testdata(struct testdata *data);

void trial_network();

#endif