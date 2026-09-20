// Experimental CPU merge. Inputs and outputs are distinct contiguous buffers.
#include <algorithm>
#include <cstdint>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" EXPORT std::int64_t merge_arrivals(
    std::int64_t old_count, std::int64_t new_count,
    const std::int64_t* old_keys, const float* old_values, const float* old_traces,
    std::int64_t* incoming, std::int64_t* keys, float* values, float* traces, float* arrivals) {
    if (new_count > 1) std::sort(incoming, incoming + new_count);
    std::int64_t i = 0, j = 0, out = 0;
    while (i < old_count || j < new_count) {
        const auto key = j == new_count ? old_keys[i]
                       : i == old_count ? incoming[j] : std::min(old_keys[i], incoming[j]);
        float value = 0.0f, trace = 0.0f, count = 0.0f;
        if (i < old_count && old_keys[i] == key) {
            value = old_values[i];
            trace = old_traces[i];
            ++i;
        }
        while (j < new_count && incoming[j] == key) {
            count += 1.0f;
            ++j;
        }
        keys[out] = key;
        values[out] = value;
        traces[out] = trace + count;
        arrivals[out] = count;
        ++out;
    }
    return out;
}
