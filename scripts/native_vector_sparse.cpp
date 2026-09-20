// Isolated exact staged-sparse experiment; retain the production ABI and kernels.
#define sparse_observe reference_sparse_observe
#include "../src/fly_connectome/native_cpu.cpp"
#undef sparse_observe

namespace {
struct Lanes {
    std::vector<std::int64_t> slots;
    std::vector<float> value,trace,count,proposal,error,leak,sign,post_trace,spike;
    std::vector<std::uint8_t> keep;
    void clear() {
        slots.clear();value.clear();trace.clear();count.clear();proposal.clear();
        error.clear();leak.clear();sign.clear();post_trace.clear();spike.clear();keep.clear();
    }
};

template<bool Behavior, bool Old, bool Causal>
void update(Lanes& lane, float pair_decay, float epsilon) {
    const auto size=static_cast<std::int64_t>(lane.slots.size());
    float* __restrict value=lane.value.data();
    float* __restrict trace=lane.trace.data();
    float* __restrict proposal=lane.proposal.data();
    const float* __restrict count=lane.count.data();
    const float* __restrict error=lane.error.data();
    const float* __restrict leak=lane.leak.data();
    const float* __restrict sign=lane.sign.data();
    const float* __restrict post_trace=lane.post_trace.data();
    const float* __restrict spike=lane.spike.data();
    std::uint8_t* __restrict keep=lane.keep.data();
    // Each lane owns one unique edge. No reductions or reordered tick updates.
    #pragma omp simd
    for (std::int64_t i=0;i<size;++i) {
        float v=value[i],t=trace[i],p=proposal[i];
        if (Old) {
            if (Causal && !Behavior) p+=error[i]*v;
            v*=leak[i];
            t*=pair_decay;
        }
        t+=count[i];
        if (Behavior) {
            const float positive=spike[i]*t;
            const float negative=count[i]*post_trace[i];
            v+=positive-negative;
        } else v+=count[i]*sign[i];
        const bool alive=(std::abs(v)>epsilon) | (t>epsilon);
        value[i]=v;trace[i]=t;proposal[i]=p;keep[i]=alive;
    }
}

template<bool Causal>
void update_groups(Lanes* lanes, float pair_decay, float epsilon) {
    update<false,false,Causal>(lanes[0],pair_decay,epsilon);
    update<false,true,Causal>(lanes[1],pair_decay,epsilon);
    update<true,false,Causal>(lanes[2],pair_decay,epsilon);
    update<true,true,Causal>(lanes[3],pair_decay,epsilon);
}
}

extern "C" EXPORT std::int64_t sparse_observe(
    std::int64_t old_count, std::int64_t new_count, std::int64_t causal, std::int64_t threads, std::int64_t n,
    float eta, float eta_reward, float decay, float pair_decay, float epsilon, float reward,
    const std::int64_t* old_keys, const float* old_values, const float* old_traces,
    std::int64_t* incoming, const std::int32_t* post, const std::uint8_t* pathway,
    const std::int8_t* signs, const float* current_decay, const float* observed,
    const float* expected, const float* post_trace, const bool* spikes, float* proposals,
    std::int64_t* keys, float* values, float* traces) {
    if (!old_count && !new_count) return 0;
    if (new_count>1) std::sort(incoming,incoming+new_count);
    // Private to the calling OS thread; reuse allocation without shared caches.
    thread_local std::vector<float> errors;
    errors.resize(n);
    float* error_data=errors.data();
    #pragma omp parallel for num_threads(threads) if(n>=16384 && threads>1)
    for (std::int64_t neuron=0;neuron<n;++neuron)
        error_data[neuron]=eta*(observed[neuron]-expected[neuron]);
    const auto chunks=std::min(threads,old_count/4096+1);
    std::vector<std::int64_t> offsets(chunks),counts(chunks);
    #pragma omp parallel for num_threads(chunks) if(chunks>1)
    for (std::int64_t chunk=0;chunk<chunks;++chunk) {
        thread_local Lanes lanes[4];
        thread_local std::vector<std::uint8_t> keep;
        for (auto& lane:lanes) lane.clear();
        const auto begin=old_count*chunk/chunks,end=old_count*(chunk+1)/chunks;
        const auto first_new=chunk==0 || new_count==0 ? 0
            : std::lower_bound(incoming,incoming+new_count,old_keys[begin])-incoming;
        const auto end_new=chunk==chunks-1 || new_count==0 ? new_count
            : std::lower_bound(incoming,incoming+new_count,old_keys[end])-incoming;
        const auto offset=begin+first_new;
        std::int64_t candidates=0;
        auto stage=[&](std::int64_t edge,std::int64_t old,float count) {
            const auto target=post[edge];
            const bool behavior=pathway[edge]==2;
            auto& lane=lanes[2*int(behavior)+int(old>=0)];
            keys[offset+candidates]=edge;
            lane.slots.push_back(candidates++);
            lane.value.push_back(old>=0 ? old_values[old] : 0.f);
            lane.trace.push_back(old>=0 ? old_traces[old] : 0.f);
            lane.count.push_back(count);
            lane.proposal.push_back(proposals[edge]);
            lane.error.push_back(error_data[target]);
            lane.leak.push_back(causal && !behavior ? current_decay[target] : decay);
            lane.sign.push_back(float(signs[edge]));
            lane.post_trace.push_back(post_trace[target]);
            lane.spike.push_back(float(spikes[target]));
            lane.keep.push_back(0);
        };
        if (first_new==end_new) {
            // No merge or duplicate-count branches on a quiet sparse partition.
            for (auto i=begin;i<end;++i) stage(old_keys[i],i,0.f);
        } else {
            auto i=begin,j=first_new;
            while (i<end || j<end_new) {
                const auto edge=j==end_new ? old_keys[i] : i==end ? incoming[j]
                    : std::min(old_keys[i],incoming[j]);
                std::int64_t old=-1;
                if (i<end && old_keys[i]==edge) old=i++;
                float count=0.f;
                while (j<end_new && incoming[j]==edge) {count+=1.f;++j;}
                stage(edge,old,count);
            }
        }
        keep.resize(candidates);
        if (causal) update_groups<true>(lanes,pair_decay,epsilon);
        else update_groups<false>(lanes,pair_decay,epsilon);
        for (int group=0;group<4;++group) {
            auto& lane=lanes[group];
            for (std::size_t i=0;i<lane.slots.size();++i) {
                const auto slot=lane.slots[i];
                if (lane.keep[i]) {
                    float delta=0.f;
                    if (group>=2) delta=(eta_reward*reward)*lane.value[i];
                    else if (!causal) delta=lane.error[i]*lane.value[i];
                    lane.proposal[i]+=delta;
                }
                proposals[keys[offset+slot]]=lane.proposal[i];
                values[offset+slot]=lane.value[i];
                traces[offset+slot]=lane.trace[i];
                keep[slot]=lane.keep[i];
            }
        }
        auto out=offset;
        for (std::int64_t i=0;i<candidates;++i) {
            if (!keep[i]) continue;
            keys[out]=keys[offset+i];values[out]=values[offset+i];traces[out]=traces[offset+i];
            ++out;
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
