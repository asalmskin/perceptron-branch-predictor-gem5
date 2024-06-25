#ifndef __PARAMS__PerceptronBP__
#define __PARAMS__PerceptronBP__

class PerceptronBP;

#include <cstddef>
#include "base/types.hh"
#include <cstddef>
#include "base/types.hh"

#include "params/BranchPredictor.hh"

struct PerceptronBPParams
    : public BranchPredictorParams
{
    PerceptronBP * create();
    size_t numPerceptrons;
    size_t numWeights;
    int threshold;
};

#endif // __PARAMS__PerceptronBP__