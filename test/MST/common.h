
#ifndef _COMMON_H_
#define _COMMON_H_

#include "lonestargpu/lonestargpu.h"

#define BLOCK_DIM 1024

void launch_find_kernel(unsigned int nb, unsigned int nt, unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid);

void launch_verify_kernel(unsigned int nb, unsigned int nt, unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid);

#endif

