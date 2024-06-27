#include "cpu/pred/perceptron.hh"

#include "base/intmath.hh"
#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/Fetch.hh"

#include <math.h>

namespace gem5
{

namespace branch_prediction
{

PerceptronBP::PerceptronBP(const PerceptronBPParams &params)
    : BPredUnit(params),
      numPerceptrons(params.numPerceptrons),
      historyLength(params.historyLength),
      threshold(params.threshold),
      indexMask(numPerceptrons - 1),
      perceptrons(numPerceptrons, std::vector<int>(historyLength + 1, 0)),
      globalHistory(historyLength, 0)
{
    if (!isPowerOf2(numPerceptrons)) {
        fatal("Number of perceptrons must be a power of 2!\n");
    }

    DPRINTF(Fetch, "Number of perceptrons: %i\n", numPerceptrons);
    DPRINTF(Fetch, "History length: %i\n", historyLength);
    DPRINTF(Fetch, "Threshold: %i\n", threshold);
}

void
PerceptronBP::updateHistories(ThreadID tid, Addr pc, bool uncond,
                              bool taken, Addr target, void * &bp_history)
{
    // Shift the global history to the left and insert the new outcome at the end
    globalHistory.insert(globalHistory.begin(), taken ? 1 : -1);
    globalHistory.pop_back();
}

bool
PerceptronBP::lookup(ThreadID tid, Addr branch_addr, void * &bp_history)
{
    unsigned perceptronIndex = getLocalIndex(branch_addr);
    int sum = perceptrons[perceptronIndex][0];  // Bias weight

    DPRINTF(Fetch, "Perceptron index: %u\n", perceptronIndex);

    // Calculate the dot product of the perceptron weights and global history
    for (size_t i = 0; i < historyLength; ++i) {
        sum += perceptrons[perceptronIndex][i + 1] * globalHistory[i];
    }

    DPRINTF(Fetch, "Sum: %d\n", sum);
    return getPrediction(sum);
}

void
PerceptronBP::update(ThreadID tid, Addr branch_addr, bool taken,
                     void * &bp_history, bool squashed,
                     const StaticInstPtr & inst, Addr target)
{
    if (squashed) {
        return;
    }

    unsigned perceptronIndex = getLocalIndex(branch_addr);
    int sum = perceptrons[perceptronIndex][0];  // Bias weight

    // Calculate the dot product of the perceptron weights and global history
    for (size_t i = 0; i < historyLength; ++i) {
        sum += perceptrons[perceptronIndex][i + 1] * globalHistory[i];
    }

    bool prediction = getPrediction(sum);
    if (prediction != taken || std::abs(sum) <= threshold) {
        // Update the bias weight
        int updatedVal = perceptrons[perceptronIndex][0] + (taken ? 1 : -1);

        if (updatedVal >= pow(2, floor(log2(threshold) + 1))) {
            perceptrons[perceptronIndex][0] = pow(2, floor(log2(threshold) + 1)) - 1;
        }
        else if (updatedVal <= -pow(2, floor(log2(threshold) + 1))) {
            perceptrons[perceptronIndex][0] = -pow(2, floor(log2(threshold) + 1)) + 1;
        }
        else
            perceptrons[perceptronIndex][0] += taken ? 1 : -1;

        // Update the weights
        for (size_t i = 0; i < historyLength; ++i) {
            updatedVal = perceptrons[perceptronIndex][i + 1] + (taken ? globalHistory[i] : -globalHistory[i]);

            if (updatedVal >= pow(2, floor(log2(threshold) + 1))) {
                perceptrons[perceptronIndex][i + 1] = pow(2, floor(log2(threshold) + 1)) - 1;
            }
            else if (updatedVal <= -pow(2, floor(log2(threshold) + 1))) {
                perceptrons[perceptronIndex][i + 1] = -pow(2, floor(log2(threshold) + 1)) + 1;
            }
            else
                perceptrons[perceptronIndex][i + 1] += taken ? globalHistory[i] : -globalHistory[i];
        }
    }

    // Update the global history
    updateHistories(tid, branch_addr, false, taken, target, bp_history);
}

inline bool
PerceptronBP::getPrediction(int sum)
{
    return sum >= 0;
}

inline unsigned
PerceptronBP::getLocalIndex(Addr &branch_addr)
{
    return (branch_addr >> instShiftAmt) & indexMask;
}

} // namespace branch_prediction
} // namespace gem5