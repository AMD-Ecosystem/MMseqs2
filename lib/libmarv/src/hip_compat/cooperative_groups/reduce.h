#ifndef MARV_HIP_COMPAT_COOPERATIVE_GROUPS_REDUCE_H
#define MARV_HIP_COMPAT_COOPERATIVE_GROUPS_REDUCE_H
// ROCm 7.2.1 has no <cooperative_groups/reduce.h> and no cooperative_groups::reduce.
// libmarv's reduce_max call sites are replaced with a tile-local butterfly under
// USE_HIP (see mathops.cuh), so this header only needs to exist and pull in the
// base cooperative-groups header.
#include <hip/hip_cooperative_groups.h>
#endif
