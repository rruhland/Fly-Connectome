// Same scalar operation order as the tensor reference; do not enable fast-math.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
extern "C" EXPORT int native_abi_version() { return 8; }

extern "C" EXPORT std::int64_t sparse_observe(
    std::int64_t old_count, std::int64_t new_count, std::int64_t causal, std::int64_t threads, std::int64_t n,
    float eta, float eta_reward, float decay, float pair_decay, float epsilon, float reward,
    const std::int64_t* old_keys, const float* old_values, const float* old_traces,
    std::int64_t* incoming, const std::int32_t* post, const std::uint8_t* pathway,
    const std::int8_t* signs, const float* current_decay, const float* observed,
    const float* expected, const float* post_trace, const bool* spikes, float* proposals,
    std::int64_t* keys, float* values, float* traces) {
    if (!old_count && !new_count) return 0;
    if (new_count > 1) std::sort(incoming, incoming+new_count);
    // The local error is identical for every existing synapse onto this neuron.
    std::vector<float> errors(n);
    #pragma omp parallel for num_threads(threads) if(n>=16384 && threads>1)
    for (std::int64_t neuron=0;neuron<n;++neuron)
        errors[neuron]=eta*(observed[neuron]-expected[neuron]);
    const auto chunks=std::min(threads,old_count/4096+1);
    std::vector<std::int64_t> offsets(chunks),counts(chunks);
    #pragma omp parallel for num_threads(chunks) if(chunks>1)
    for (std::int64_t chunk=0;chunk<chunks;++chunk) {
    const auto begin=old_count*chunk/chunks,end=old_count*(chunk+1)/chunks;
    const auto first_new=chunk==0 || new_count==0 ? 0 : std::lower_bound(incoming,incoming+new_count,old_keys[begin])-incoming;
    const auto end_new=chunk==chunks-1 || new_count==0 ? new_count : std::lower_bound(incoming,incoming+new_count,old_keys[end])-incoming;
    const auto offset=begin+first_new;
    std::int64_t i=begin,j=first_new,out=offset;
    while (i<end || j<end_new) {
        const auto edge = j==end_new ? old_keys[i] : i==end ? incoming[j]
                         : std::min(old_keys[i],incoming[j]);
        const auto target=post[edge];
        const bool behavior=pathway[edge]==2;
        float value=0.f,trace=0.f,count=0.f;
        if (i<end && old_keys[i]==edge) {
            value=old_values[i];
            if (causal && !behavior) {
                proposals[edge] += errors[target]*value;
            }
            value *= causal && !behavior ? current_decay[target] : decay;
            trace=old_traces[i]*pair_decay;
            ++i;
        }
        while (j<end_new && incoming[j]==edge) {count+=1.f;++j;}
        trace += count;
        if (behavior) {
            const float positive=float(spikes[target])*trace;
            const float negative=count*post_trace[target];
            value += positive-negative;
        } else value += count*signs[edge];
        if (std::abs(value)>epsilon || trace>epsilon) {
            float proposal=0.f;
            if (behavior) proposal=(eta_reward*reward)*value;
            else if (!causal) proposal=errors[target]*value;
            proposals[edge] += proposal;
            keys[out]=edge;values[out]=value;traces[out]=trace;++out;
        }
    }
    offsets[chunk]=offset;counts[chunk]=out-offset;
    }
    std::int64_t total=0;
    for (std::int64_t chunk=0;chunk<chunks;++chunk) {
        const auto offset=offsets[chunk],count=counts[chunk];
        if (count && total!=offset) {
            std::memmove(keys+total,keys+offset,count*sizeof(*keys));
            std::memmove(values+total,values+offset,count*sizeof(*values));
            std::memmove(traces+total,traces+offset,count*sizeof(*traces));
        }
        total+=count;
    }
    return total;
}

extern "C" EXPORT void neural_step(std::int64_t n, std::int64_t arrivals,
    std::int64_t classes, std::int64_t refractory_steps, std::int64_t sensory_filter, std::int64_t threads,
    float current_leak, float membrane_leak, float membrane_gain, float sensory_leak,
    float adaptation_leak, float threshold, float jump,
    const std::int64_t* edges, const std::int32_t* post, const std::uint8_t* pathway,
    const float* weights, const std::int8_t* signs, const float* current_decay, const float* membrane_decay,
    const float* rest, const float* injection, const bool* silenced,
    float* ff, float* pred, float* behavior, float* sensory, float* adaptation,
    float* voltage, std::int64_t* refractory, float* observed, float* predicted,
    float* increments, bool* spikes) {
    #pragma omp parallel for num_threads(threads) if(n>=16384 && threads>1)
    for (std::int64_t i=0;i<n;++i) {
        const float leak=classes ? current_decay[i] : current_leak;
        ff[i]*=leak;pred[i]*=leak;behavior[i]*=leak;
        if (increments) increments[i]=0.f;
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

extern "C" EXPORT void prepare_observation(std::int64_t n, std::int64_t increments,
    std::int64_t threads, float threshold, float sensory_gain, float pair_decay, float low,
    const float* observed, const float* arrivals, const float* sensory, const bool* mask,
    float* post_trace, float* target) {
    #pragma omp parallel for num_threads(threads) if(n>=16384 && threads>1)
    for (std::int64_t i=0;i<n;++i) {
        const bool input=increments && mask[i];
        const float raw=increments ? (input ? sensory[i] : arrivals[i]) : observed[i];
        const float divisor=input ? sensory_gain : threshold;
        float value=divisor==1.f ? raw : raw/divisor;
        if (value<low) value=low;
        if (value>1.f) value=1.f;
        target[i]=value;
        post_trace[i]*=pair_decay;
    }
}

extern "C" EXPORT int synchronize_weights(std::int64_t edges, std::int64_t threads,
    float maximum, float* weights, float* proposals, float* exponents) {
    int active=0;
    #pragma omp parallel for num_threads(threads) reduction(|:active) if(edges>=16384 && threads>1)
    for (std::int64_t i=0;i<edges;++i) active|=exponents[i]!=0.f;
    // Leave all state untouched for the reference exponential implementation.
    if (active) return 0;
    #pragma omp parallel for num_threads(threads) if(edges>=16384 && threads>1)
    for (std::int64_t i=0;i<edges;++i) {
        float value=weights[i]+proposals[i];
        if (value<0.f) value=0.f;
        if (value>maximum) value=maximum;
        weights[i]=value;
        proposals[i]=0.f;
        exponents[i]=0.f;
    }
    return 1;
}

extern "C" EXPORT void finish_observation(std::int64_t n, std::int64_t threads,
    float threshold, float low, float maximum_rate,
    const float* predicted, const bool* spikes, float* expected, float* post_trace,
    float* rates, float* overload) {
    #pragma omp parallel for num_threads(threads) if(n>=16384 && threads>1)
    for (std::int64_t i=0;i<n;++i) {
        float prediction=threshold==1.f ? predicted[i] : predicted[i]/threshold;
        if (prediction<low) prediction=low;
        if (prediction>1.f) prediction=1.f;
        expected[i]=prediction;
        const float spike=float(spikes[i]);
        post_trace[i]+=spike;
        overload[i]=std::max(0.f,rates[i]-maximum_rate);
    }
}

extern "C" EXPORT std::int64_t spike_arrivals(std::int64_t n, std::int64_t slots,
    std::int64_t step, const bool* history, const std::int64_t* starts,
    const std::int64_t* delays, std::int64_t* output) {
    std::int64_t count=0;
    for (std::int64_t slot=0;slot<slots;++slot) {
        const auto age=((step-slot)%slots+slots)%slots;
        if (!age) continue;
        for (std::int64_t neuron=0;neuron<n;++neuron) {
            if (!history[slot*n+neuron]) continue;
            for (auto edge=starts[neuron];edge<starts[neuron+1];++edge) {
                if (delays[edge]!=age) continue;
                if (output) output[count]=edge;
                ++count;
            }
        }
    }
    return count;
}

// Experimental fallback for deferred intervals: fuse memory passes while retaining
// every float32 tick, clipping-derived local error, and pruning decision.
extern "C" EXPORT int deferred_abi_version() { return 2; }
extern "C" EXPORT void deferred_error(std::int64_t n, std::int64_t threads, float eta,
    const float* observed, const float* expected, float* errors) {
    #pragma omp parallel for num_threads(threads) if(n>=16384 && threads>1)
    for (std::int64_t i=0;i<n;++i) errors[i]=eta*(observed[i]-expected[i]);
}

extern "C" EXPORT std::int64_t deferred_visual(
    std::int64_t old_count, std::int64_t new_count, std::int64_t steps,
    std::int64_t n, std::int64_t threads, std::int64_t aggregate, float pair_decay, float epsilon,
    const std::int64_t* old_keys, const float* old_values, const float* old_traces,
    std::int64_t* incoming, const std::int32_t* post, const std::int8_t* signs,
    const float* decay, const float* errors, float* proposals,
    std::int64_t* keys, float* values, float* traces) {
    if (new_count>1) std::sort(incoming,incoming+new_count);
    std::vector<double> coefficients(aggregate ? n : 0);
    if (aggregate) {
        #pragma omp parallel for num_threads(threads) if(n>=16384 && threads>1)
        for (std::int64_t neuron=0;neuron<n;++neuron) {
            double power=1.,sum=0.;
            for (std::int64_t tick=0;tick<steps;++tick) {
                sum+=power*errors[neuron*steps+tick];
                power*=decay[neuron];
            }
            coefficients[neuron]=sum;
        }
    }
    const auto chunks=std::min(threads,old_count/4096+1);
    std::vector<std::int64_t> offsets(chunks),counts(chunks);
    #pragma omp parallel for num_threads(chunks) if(chunks>1)
    for (std::int64_t chunk=0;chunk<chunks;++chunk) {
        const auto begin=old_count*chunk/chunks,end=old_count*(chunk+1)/chunks;
        const auto first_new=chunk==0 || !new_count ? 0 : std::lower_bound(incoming,incoming+new_count,old_keys[begin]*steps)-incoming;
        const auto end_new=chunk==chunks-1 || !new_count ? new_count : std::lower_bound(incoming,incoming+new_count,old_keys[end]*steps)-incoming;
        const auto offset=begin+first_new;
        auto i=begin,j=first_new,out=offset;
        while (i<end || j<end_new) {
            const auto edge=j==end_new ? old_keys[i] : i==end ? incoming[j]/steps
                : std::min(old_keys[i],incoming[j]/steps);
            const auto target=post[edge];
            bool alive=i<end && old_keys[i]==edge;
            float value=alive ? old_values[i] : 0.f;
            float trace=alive ? old_traces[i] : 0.f;
            if (alive) ++i;
            float proposal=proposals[edge];
            if (aggregate && alive && (j==end_new || incoming[j]/steps!=edge)) {
                float end_value=value,end_trace=trace;
                for (std::int64_t tick=0;tick<steps;++tick) {
                    end_value*=decay[target];end_value+=0.f*signs[edge];
                    end_trace*=pair_decay;end_trace+=0.f;
                }
                // Without arrivals both magnitudes are monotone. Surviving the
                // end proves no pruning crossing; otherwise replay below.
                if (std::abs(end_value)>epsilon || end_trace>epsilon) {
                    proposals[edge]+=float(double(value)*coefficients[target]);
                    keys[out]=edge;values[out]=end_value;traces[out]=end_trace;++out;
                    continue;
                }
            }
            for (std::int64_t tick=0;tick<steps;++tick) {
                if (alive) {
                    proposal+=errors[target*steps+tick]*value;
                    value*=decay[target];
                    trace*=pair_decay;
                }
                float count=0.f;
                while (j<end_new && incoming[j]==edge*steps+tick) {count+=1.f;++j;}
                trace+=count;
                value+=count*signs[edge];
                alive=std::abs(value)>epsilon || trace>epsilon;
                if (alive) proposal+=0.f;
                else {value=0.f;trace=0.f;}
            }
            proposals[edge]=proposal;
            if (alive) {keys[out]=edge;values[out]=value;traces[out]=trace;++out;}
        }
        offsets[chunk]=offset;counts[chunk]=out-offset;
    }
    std::int64_t total=0;
    for (std::int64_t chunk=0;chunk<chunks;++chunk) {
        const auto offset=offsets[chunk],count=counts[chunk];
        if (count && total!=offset) {
            std::memmove(keys+total,keys+offset,count*sizeof(*keys));
            std::memmove(values+total,values+offset,count*sizeof(*values));
            std::memmove(traces+total,traces+offset,count*sizeof(*traces));
        }
        total+=count;
    }
    return total;
}
