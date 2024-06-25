#include "cpu/pred/perceptron.hh"

#include "base/intmath.hh"
#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/Fetch.hh"

namespace gem5
{

namespace branch_prediction
{

PerceptronBP::PerceptronBP(const PerceptronBPParams &params)
    : BPredUnit(params),
      numPerceptrons(params.numPerceptrons),
      numWeights(params.numWeights),
      threshold(params.threshold),
      indexMask(numPerceptrons - 1),
      perceptrons(numPerceptrons, std::vector<int>(numWeights, 0)),
      globalHistory(numWeights, 1)
{
    if (!isPowerOf2(numPerceptrons)) {
        fatal("Invalid number of perceptrons! Must be a power of 2.\n");
    }

    if (!isPowerOf2(numWeights + 1)) {
        fatal("Invalid history length! Must be a power of 2 - 1.\n");
    }

    DPRINTF(Fetch, "index mask: %#x\n", indexMask);
    DPRINTF(Fetch, "number of perceptrons: %i\n", numPerceptrons);
    DPRINTF(Fetch, "number of weights: %i\n", numWeights);
    DPRINTF(Fetch, "threshold: %i\n", threshold);
}

void
PerceptronBP::updateHistories(ThreadID tid, Addr pc, bool uncond,
                         bool taken, Addr target, void * &bp_history)
{
    globalHistory.pop_back();
    globalHistory.insert(globalHistory.begin(), taken ? 1 : -1);
}

bool
PerceptronBP::lookup(ThreadID tid, Addr branch_addr, void * &bp_history)
{
    unsigned index = getLocalIndex(branch_addr);
    int sum = 0;
    for (size_t i = 0; i < numWeights; ++i) {
        sum += perceptrons[index][i] * globalHistory[i];
    }

    DPRINTF(Fetch, "Lookup: index=%#x, sum=%d\n", index, sum);

    return getPrediction(sum);
}

void
PerceptronBP::update(ThreadID tid, Addr branch_addr, bool taken, void *&bp_history,
                bool squashed, const StaticInstPtr & inst, Addr target)
{
    assert(bp_history == NULL);
    
    if (squashed) {
        return;
    }

    unsigned index = getLocalIndex(branch_addr);
    int sum = 0;
    for (size_t i = 0; i < numWeights; ++i) {
        sum += perceptrons[index][i] * globalHistory[i];
    }

    int outcome = taken ? 1 : -1;
    if ((getPrediction(sum) != taken) || (std::abs(sum) <= threshold)) {
        for (size_t i = 0; i < numWeights; ++i) {
            perceptrons[index][i] += outcome * globalHistory[i];
        }
    }

    DPRINTF(Fetch, "Update: index=%#x, outcome=%d, sum=%d\n", index, outcome, sum);

    updateHistories(tid, branch_addr, false, taken, target, bp_history);
}

inline
bool
PerceptronBP::getPrediction(uint8_t &count)
{
    return sum >= 0;
}

inline
unsigned
PerceptronBP::getLocalIndex(Addr &branch_addr)
{
    return (branch_addr >> instShiftAmt) & indexMask;
}

} // namespace branch_prediction
} // namespace gem5