// Same scalar operation order as the tensor reference; do not enable fast-math.
#include <algorithm>
#include <cmath>
#include <cstdint>
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
extern "C" EXPORT std::int64_t sparse_observe(
    std::int64_t old_count, std::int64_t new_count, std::int64_t causal,
    float eta, float eta_reward, float decay, float pair_decay, float epsilon, float reward,
    const std::int64_t* old_keys, const float* old_values, const float* old_traces,
    std::int64_t* incoming, const std::int64_t* post, const std::int64_t* pathway,
    const float* signs, const float* current_decay, const float* observed,
    const float* expected, const float* post_trace, const bool* spikes, float* proposals,
    std::int64_t* keys, float* values, float* traces) {
    if (new_count > 1) std::sort(incoming, incoming+new_count);
    std::int64_t i=0,j=0,out=0;
    while (i<old_count || j<new_count) {
        const auto edge = j==new_count ? old_keys[i] : i==old_count ? incoming[j]
                         : std::min(old_keys[i],incoming[j]);
        const auto target=post[edge];
        const bool behavior=pathway[edge]==2;
        float value=0.f,trace=0.f,count=0.f;
        if (i<old_count && old_keys[i]==edge) {
            value=old_values[i];
            if (causal && !behavior) {
                const float error=observed[target]-expected[target];
                const float scaled=eta*error;
                proposals[edge] += scaled*value;
            }
            value *= causal && !behavior ? current_decay[target] : decay;
            trace=old_traces[i]*pair_decay;
            ++i;
        }
        while (j<new_count && incoming[j]==edge) {count+=1.f;++j;}
        trace += count;
        if (behavior) {
            const float positive=float(spikes[target])*trace;
            const float negative=count*post_trace[target];
            value += positive-negative;
        } else value += count*signs[edge];
        if (std::abs(value)>epsilon || trace>epsilon) {
            float proposal=0.f;
            if (behavior) proposal=(eta_reward*reward)*value;
            else if (!causal) proposal=(eta*(observed[target]-expected[target]))*value;
            proposals[edge] += proposal;
            keys[out]=edge;values[out]=value;traces[out]=trace;++out;
        }
    }
    return out;
}

extern "C" EXPORT void neural_step(std::int64_t n, std::int64_t arrivals,
    std::int64_t classes, std::int64_t refractory_steps, std::int64_t sensory_filter,
    float current_leak, float membrane_leak, float membrane_gain, float sensory_leak,
    float adaptation_leak, float threshold, float jump,
    const std::int64_t* edges, const std::int64_t* post, const std::int64_t* pathway,
    const float* weights, const float* signs, const float* current_decay, const float* membrane_decay,
    const float* rest, const float* injection, const bool* silenced,
    float* ff, float* pred, float* behavior, float* sensory, float* adaptation,
    float* voltage, std::int64_t* refractory, float* observed, float* predicted,
    float* increments, bool* spikes) {
    for (std::int64_t i=0;i<n;++i) {
        const float leak=classes ? current_decay[i] : current_leak;
        ff[i]*=leak;pred[i]*=leak;behavior[i]*=leak;
        if (increments) increments[i]=0.f;
    }
    for (std::int64_t k=0;k<arrivals;++k) {
        const auto edge=edges[k], target=post[edge];
        const float weight=weights[edge]*signs[edge];
        if (pathway[edge]==0) {ff[target]+=weight;if (increments) increments[target]+=weight;}
        else if (pathway[edge]==1) pred[target]+=weight;
        else behavior[target]+=weight;
    }
    for (std::int64_t i=0;i<n;++i) {
        sensory[i]=sensory_filter ? sensory[i]*sensory_leak+injection[i] : injection[i];
        observed[i]=ff[i]+sensory[i];
        predicted[i]=pred[i];
        const float current=((observed[i]+predicted[i])+behavior[i])+rest[i];
        adaptation[i]*=adaptation_leak;
        const bool eligible=refractory[i]==0;
        refractory[i]=std::max(std::int64_t(0),refractory[i]-1);
        const float decay=classes ? membrane_decay[i] : membrane_leak;
        const float gain=classes ? 1.f-decay : membrane_gain;
        voltage[i]=voltage[i]*decay+current*gain;
        if (!eligible || silenced[i]) voltage[i]=0.f;
        spikes[i]=eligible && !silenced[i] && voltage[i]>=threshold+adaptation[i];
        if (spikes[i]) {voltage[i]=0.f;refractory[i]=refractory_steps;}
        adaptation[i]+=float(spikes[i])*jump;
    }
}
