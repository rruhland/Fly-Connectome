// Experimental only. Preserve native_cpu.cpp operation order and float32 rounding.
#include <algorithm>
#include <cstdint>
#include <cstring>
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
static bool same(float a, float b) { return std::memcmp(&a,&b,sizeof(float))==0; }
extern "C" EXPORT void sleeping_step(std::int64_t n, std::int64_t arrivals,
    std::int64_t classes, std::int64_t refractory_steps, std::int64_t sensory_filter, std::int64_t threads,
    float current_leak, float membrane_leak, float membrane_gain, float sensory_leak,
    float adaptation_leak, float threshold, float jump,
    const std::int64_t* edges, const std::int32_t* post, const std::uint8_t* pathway,
    const float* weights, const std::int8_t* signs, const float* current_decay, const float* membrane_decay,
    const float* rest, const float* injection, const bool* silenced,
    float* ff, float* pred, float* behavior, float* sensory, float* adaptation,
    float* voltage, std::int64_t* refractory, float* observed, float* predicted,
    float* increments, bool* spikes, std::uint8_t* status) {
    // 0 = awake candidate, 1 = confirmed sleeping, 2 = changed/input this tick.
    for (std::int64_t k=0;k<arrivals;++k) status[post[edges[k]]]=2;
    #pragma omp parallel for num_threads(threads) if(n>=16384 && threads>1)
    for (std::int64_t i=0;i<n;++i) {
        if (!same(injection[i],0.f)) status[i]=2;
        if (increments) increments[i]=0.f;
        if (status[i]==1) continue;
        const float old_ff=ff[i],old_pred=pred[i],old_behavior=behavior[i];
        const float leak=classes ? current_decay[i] : current_leak;
        ff[i]*=leak;pred[i]*=leak;behavior[i]*=leak;
        if (!same(old_ff,ff[i]) || !same(old_pred,pred[i]) || !same(old_behavior,behavior[i])) status[i]=2;
    }
    for (std::int64_t k=0;k<arrivals;++k) {
        const auto edge=edges[k];
        const auto target=post[edge];
        const float weight=weights[edge]*signs[edge];
        if (pathway[edge]==0) {ff[target]+=weight;if (increments) increments[target]+=weight;}
        else if (pathway[edge]==1) pred[target]+=weight;
        else behavior[target]+=weight;
    }
    #pragma omp parallel for num_threads(threads) if(n>=16384 && threads>1)
    for (std::int64_t i=0;i<n;++i) {
        if (status[i]==1) {
            observed[i]=ff[i]+sensory[i];predicted[i]=pred[i];spikes[i]=false;
            continue;
        }
        const float old_sensory=sensory[i],old_adaptation=adaptation[i],old_voltage=voltage[i];
        const auto old_refractory=refractory[i];
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
        status[i]=(status[i]!=2 && !spikes[i] && same(old_sensory,sensory[i]) &&
            same(old_adaptation,adaptation[i]) && same(old_voltage,voltage[i]) &&
            old_refractory==refractory[i]) ? 1 : 0;
    }
}
