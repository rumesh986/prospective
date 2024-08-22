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

enum pool_t {
	pool_max,
	pool_avg
};

// collection of blocks
struct network {
	double alpha;
	enum activation act;
	enum weights_init weight_init;
	enum db_proc proc;

	struct block *head; // points to input layer, initial value is 0, real value set in init_network
	struct block *tail; // points to output layer, set to number of targets

	size_t nlayers;
	size_t ntargets;
	size_t *targets;

	size_t lenergy_chunks; // used for managing memory for data collection in child blocks

	bool save;
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

	size_t length;

	gsl_vector *layer;
	gsl_vector *epsilon;
	gsl_vector *deltax;

	gsl_matrix *weights;
	gsl_matrix *deltaw;

	// temp vectors, used to avoid corrupting main vectors
	gsl_vector *tlayer;
	gsl_vector *tepsilon;

	// data to be collected
	size_t nenergies;
	double **energies;
	double *deltaw_mags;
	double **deltax_mags;
};

struct block_layer {};

struct block_cnn {
	size_t kernel_size;
	size_t stride;
	size_t padding;

	size_t nchannels;
	size_t nmats;
	
	size_t pool_size;
	enum pool_t pool_type;
	gsl_vector *pool_indices; // holds location of max index if needed

	size_t image_size;
	size_t image_length;
	
	size_t conv_size;
	size_t conv_length;
	gsl_vector *conv_layer;
	gsl_matrix **padded_input;

	gsl_matrix **dAdxP;
	gsl_matrix *dAdx;
	gsl_matrix **dAdw;
};

struct amg {
	struct network **nets;
	size_t depth;
	// gsl_vector *
};

struct training {
	struct relaxation_params relax;
	struct network *net;

	size_t num_samples;
	size_t seed;
	size_t test_samples_per_iters;

	struct amg amg;
};

struct testing {
	bool relax;
	struct relaxation_params relax_params;
	struct network *net;

	size_t num_samples;
};

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
	size_t ntargets;

	bool relax;
};

void init_network(struct network *net);
void set_network(struct network *net);
void save_network(char *filename);
void free_network(struct network *net);
void clear_block_data();

struct traindata *train(struct training train, bool logging);
void save_traindata(struct traindata *data, char *filename);
void free_traindata(struct traindata *data);

struct testdata *test(struct testing test, bool logging);
void save_testdata(struct testdata *data, char *filename);
void free_testdata(struct testdata *data);

void trial_network();
// struct net_params *load_network(char *filename);

#endif