#ifndef __INCLUDEALL_H__
#define __INCLUDEALL_H__

// #define LOGGING
#define ERR_INVALID_CONFIG	1
#define ERR_VEC_OPS			2
#define ERR_MEM				3
#define ERR_FILE			4

#define CHUNK_SIZE	100

#include <stdbool.h>
#include <gsl/gsl_matrix.h>

#include "database.h"
#include "network_utils.h"
#include "network.h"
#include "utils.h"
#include "config.h"

#endif