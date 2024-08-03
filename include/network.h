#ifndef __NETWORK_H__
#define __NETWORK_H__

struct relaxation_params {
	double gamma;
	double gamma_rate;

	size_t gamma_count;
	double energy_res;
	size_t max_iters;
};

struct net_params {
	mnist_db mnist;
	mnist_processing mnist_proc;

	double alpha;
	
	size_t tau;
	enum activation act;
	enum weights_init weights;
	
	size_t nlayers;
	size_t *lengths; // must be alloc'd

	size_t ntargets;
	size_t *targets; // must be alloc'd

	struct relaxation_params relax_params;
};

struct training_params {
	size_t num_samples;
	size_t seed;

	size_t test_samples;

	struct relaxation_params relax_params;

	bool logging;
};

struct testing_params {
	size_t num_samples;

	struct relaxation_params relax_params;

	bool logging;
};

// new stuff

// type of block
enum block_t {
	block_dense
};

// abstract block contains pointer to a specific block and its type
struct block {
	enum block_t type;
	void *block;
};

// collection of blocks
struct network {
	size_t nblocks;
	struct block **blocks;

	struct training *training;

	bool save;
};

// standard densely connected network
struct dense_block {
	double alpha;
	enum activation act;
	enum weights_init weight_init;

	size_t nlayers;
	size_t *layers;
};

struct training {
	int id;

	struct network *net;

	size_t ntargets;
	size_t *targets;

	struct {
		enum db_proc proc;
		size_t num_samples;
		size_t seed;
		size_t test_samples_per_iters;

		struct relaxation_params *relax;
	} params;
};

struct testing {
	struct network *net;

	size_t num_samples;
	struct relaxation_params *relax;
};

// back to old stuff
struct traindata {
	gsl_vector **delta_w_mags;
	gsl_vector *iter_counts;
	gsl_vector **energies;

	gsl_vector ***lenergies;
	gsl_vector *train_costs;

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

void build_network(size_t inp_len, size_t out_len, struct net_params *params);
struct net_params *load_network(char *filename);
int save_network(char *filename);
void free_network();

struct traindata *train(struct training_params train_params);
int save_traindata(struct traindata *data, char *filename);
void free_traindata(struct traindata *data);

struct testdata *test(struct testing_params test_params, bool relaxation);
int save_testdata(struct testdata *data, char *filename);
void free_testdata(struct testdata *data);

void trial_network();

#endif