// Experimental regrouping of the discrete neuron recurrence. No fast-math/FMA.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

namespace {
const double* coeff(const double* table, std::int32_t group,
                    std::int64_t horizon, std::int64_t k) {
    return table+(group*(horizon+1)+k)*7;
}

void advance(std::int64_t i, std::int64_t k, const double* p,
             const bool* silenced, float* ff, float* pred, float* behavior,
             float* sensory, float* adaptation, float* voltage,
             std::int64_t* refractory, const float* rest) {
    if (!k) return;
    // The scheduler splits intervals at refractory expiry; it never skips a spike.
    const double currents=double(ff[i])+pred[i]+behavior[i];
    voltage[i]=silenced[i] || refractory[i]>0 ? 0.f
        : float(double(voltage[i])*p[0]+double(rest[i])*p[4]
                +currents*p[5]+double(sensory[i])*p[6]);
    ff[i]=float(double(ff[i])*p[1]);
    pred[i]=float(double(pred[i])*p[1]);
    behavior[i]=float(double(behavior[i])*p[1]);
    sensory[i]=float(double(sensory[i])*p[2]);
    adaptation[i]=float(double(adaptation[i])*p[3]);
    refractory[i]=std::max(std::int64_t(0),refractory[i]-k);
}

double margin(std::int64_t k, double magnitude) {
    const double e=16.*double(k)*std::numeric_limits<float>::epsilon();
    if (e>=1.) return std::numeric_limits<double>::infinity();
    // Absolute underflow allowance is an error bound, never state pruning.
    return e/(1.-e)*magnitude
        +16.*double(k)*std::numeric_limits<float>::denorm_min();
}

std::int64_t schedule(std::int64_t horizon, const double* table,
    const double* bounds, std::int32_t group, float voltage, float rest,
    float ff, float pred, float behavior, float sensory, float adaptation,
    float threshold, std::int64_t refractory, bool silenced, bool& certified) {
    certified=false;
    const double absolute=std::abs(double(voltage))+std::abs(double(rest))
        +std::abs(double(ff))+std::abs(double(pred))+std::abs(double(behavior))
        +std::abs(double(sensory))+std::abs(double(adaptation))+std::abs(double(threshold));
    // Roundoff bounds presume finite intermediate float operations. Fail closed
    // before ordered current sums or threshold arithmetic can overflow.
    if (!std::isfinite(absolute) || absolute>double(std::numeric_limits<float>::max())/16.) return 1;
    if (silenced) return horizon;
    if (refractory>0) return std::min(horizon,refractory);
    const double values[]={voltage,rest,double(ff)+pred+behavior,sensory};
    const double magnitudes[]={std::abs(double(voltage)),std::abs(double(rest)),
        std::abs(double(ff))+std::abs(double(pred))+std::abs(double(behavior)),
        std::abs(double(sensory))};
    const auto* end=coeff(table,group,horizon,horizon);
    const auto* first=coeff(table,group,horizon,1);
    const auto* b=bounds+(group*(horizon+1)+horizon)*8;
    double upper=0.,magnitude=std::abs(double(threshold))+std::abs(double(adaptation));
    for (int j=0;j<4;++j) {
        upper+=values[j]*(values[j]>=0. ? b[2*j+1] : b[2*j]);
        magnitude+=magnitudes[j]*std::max(std::abs(b[2*j]),std::abs(b[2*j+1]));
    }
    const double lower=double(threshold)+std::min(double(adaptation)*first[3],
                                                 double(adaptation)*end[3]);
    if (upper+margin(horizon,magnitude)<lower) {
        certified=true;
        return horizon;
    }
    // Mixed-sign transients need not be monotone: inspect every integer tick.
    const int indices[]={0,4,5,6};
    for (std::int64_t k=1;k<=horizon;++k) {
        const auto* p=coeff(table,group,horizon,k);
        double v=0.,scale=std::abs(double(threshold))+std::abs(double(adaptation)*p[3]);
        for (int j=0;j<4;++j) {
            v+=values[j]*p[indices[j]];
            scale+=magnitudes[j]*std::abs(p[indices[j]]);
        }
        // NaNs/nonfinite bounds must fail closed to a candidate, never certify.
        if (!(v+margin(k,scale)<double(threshold)+double(adaptation)*p[3])) return k;
    }
    certified=true;
    return horizon;
}
}

extern "C" EXPORT void event_step(std::int64_t n, std::int64_t arrivals,
    std::int64_t classes, std::int64_t refractory_steps, std::int64_t sensory_filter, std::int64_t threads,
    float current_leak, float membrane_leak, float membrane_gain, float sensory_leak,
    float adaptation_leak, float threshold, float jump,
    const std::int64_t* edges, const std::int32_t* post, const std::uint8_t* pathway,
    const float* weights, const std::int8_t* signs, const float* current_decay, const float* membrane_decay,
    const float* rest, const float* injection, const bool* silenced,
    float* ff, float* pred, float* behavior, float* sensory, float* adaptation,
    float* voltage, std::int64_t* refractory, float* observed, float* predicted,
    float* increments, bool* spikes,
    std::int64_t step, std::int64_t horizon, const std::int32_t* groups,
    const double* coefficients, const double* bounds, std::int64_t* last,
    std::int64_t* due, std::uint8_t* awake, std::int64_t* counters) {
    #pragma omp parallel for num_threads(threads) if(n>=16384 && threads>1)
    for (std::int64_t i=0;i<n;++i) {
        std::uint32_t bits;
        std::memcpy(&bits,injection+i,sizeof(bits));
        awake[i]=due[i]<=step || bits!=0;
        spikes[i]=false;
        if (increments) increments[i]=0.f;
    }
    for (std::int64_t k=0;k<arrivals;++k) awake[post[edges[k]]]=1;
    std::int64_t active=0,skipped=0,candidates=0,certificates=0;
    #pragma omp parallel for num_threads(threads) reduction(+:active,skipped) if(n>=16384 && threads>1)
    for (std::int64_t i=0;i<n;++i) {
        if (!awake[i]) continue;
        const auto gap=step-last[i]-1;
        advance(i,gap,coeff(coefficients,groups[i],horizon,gap),silenced,
                ff,pred,behavior,sensory,adaptation,voltage,refractory,rest);
        ++active;skipped+=gap;
        const float leak=classes ? current_decay[i] : current_leak;
        ff[i]*=leak;pred[i]*=leak;behavior[i]*=leak;
    }
    // Preserve the reference's arrival accumulation order, including duplicates.
    for (std::int64_t k=0;k<arrivals;++k) {
        const auto edge=edges[k];
        const auto target=post[edge];
        const float weight=weights[edge]*signs[edge];
        if (pathway[edge]==0) {ff[target]+=weight;if (increments) increments[target]+=weight;}
        else if (pathway[edge]==1) pred[target]+=weight;
        else behavior[target]+=weight;
    }
    #pragma omp parallel for num_threads(threads) reduction(+:candidates,certificates) if(n>=16384 && threads>1)
    for (std::int64_t i=0;i<n;++i) {
        if (!awake[i]) {
            const auto* p=coeff(coefficients,groups[i],horizon,step-last[i]);
            observed[i]=float(double(ff[i])*p[1])+float(double(sensory[i])*p[2]);
            predicted[i]=float(double(pred[i])*p[1]);
            continue;
        }
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
        last[i]=step;
        bool certified;
        due[i]=step+schedule(horizon,coefficients,bounds,groups[i],voltage[i],rest[i],
            ff[i],pred[i],behavior[i],sensory[i],adaptation[i],threshold,
            refractory[i],silenced[i],certified);
        certificates+=certified;
        candidates+=!certified && !silenced[i] && refractory[i]==0;
    }
    counters[0]+=active;counters[1]+=skipped;
    counters[2]+=candidates;counters[3]+=certificates;
}

extern "C" EXPORT void event_materialize(std::int64_t n, std::int64_t step,
    std::int64_t horizon, const std::int32_t* groups, const double* coefficients,
    bool* silenced, float* ff, float* pred, float* behavior, float* sensory,
    float* adaptation, float* voltage, std::int64_t* refractory,
    const float* rest, std::int64_t* last, std::int64_t* due) {
    for (std::int64_t i=0;i<n;++i) {
        const auto gap=step-last[i];
        advance(i,gap,coeff(coefficients,groups[i],horizon,gap),silenced,
                ff,pred,behavior,sensory,adaptation,voltage,refractory,rest);
        last[i]=step;due[i]=step+1;
    }
}
