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
	block_layer,
	block_cnn
};

// abstract block contains pointer to a specific block and its type
struct block {
	enum block_t type;

	union {
		struct block_layer *blayer;
		struct block_cnn *cnn;
	};

	struct block *prev;
	struct block *next;

	gsl_vector *layer;
	gsl_vector *epsilon;

	// gsl_vector *act;	// used to calculate activated versions or derivative of activation versions
	// gsl_vector *out;

	gsl_matrix *weights;

	gsl_vector *deltax;
	gsl_matrix *deltaw;

	gsl_vector *tlayer;
	gsl_vector *tepsilon;	// used when calculating energies, without disturbing epsilon


	double **energies;
	double *deltaw_mags;
};

// collection of blocks
struct network {
	double alpha;
	enum activation act;
	enum weights_init weight_init;
	enum db_proc proc;

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
};

struct block_cnn {
	size_t kernel_size;
	size_t stride;
	size_t padding;

	size_t length;
	gsl_matrix_view layer_mat;
};

struct training {
	struct relaxation_params relax;
	struct network *net;

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
	double *iter_counts;
	double *train_costs;
	size_t num_samples;
};

struct testdata {
	gsl_vector **costs;

	gsl_vector *labels;
	gsl_vector *predictions;
	gsl_vector **outputs;

	gsl_vector *iter_counts;

	size_t num_correct;
	size_t num_samples;

	bool relax;
};

void init_network(struct network *net);
void set_network(struct network *net);
void save_network(char *filename);
void free_network(struct network *net);
void clear_block_data();

// struct net_params *load_network(char *filename);

struct traindata *train(struct training train, bool logging);
void save_traindata(struct traindata *data, char *filename);

// void free_traindata(struct traindata *data);

struct testdata *test(struct testing test, bool logging);
void save_testdata(struct testdata *data, char *filename);

// void free_testdata(struct testdata *data);

void trial_network();

#endif