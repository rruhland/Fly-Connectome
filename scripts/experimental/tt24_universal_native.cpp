// Feature replacement inserted into the unchanged donor physics and referee.
constexpr int kChannels = 11;
double universal_center[kChannels] = {};
double universal_scale[kChannels] = {};
double universal_matrix[kChannels*kChannels] = {};

void features(const double* state, double* output) {
    double normalized[kChannels], mixed[kChannels];
    for (int i = 0; i < kChannels; ++i) {
        normalized[i] = std::clamp(std::asinh((state[i] - universal_center[i]) * universal_scale[i]), -3.4, 3.4);
    }
    output[0] = 1.;
    for (int i = 0; i < kChannels; ++i) {
        double value = 0.;
        for (int j = 0; j < kChannels; ++j) value += universal_matrix[kChannels*i + j] * normalized[j];
        mixed[i] = std::clamp(value, -3., 3.);
        output[i + 1] = mixed[i];
    }
    int at = 1 + kChannels;
    for (int i = 0; i < kChannels; ++i) {
        for (int j = i; j < kChannels; ++j) {
            output[at++] = std::clamp(.75 * mixed[i] * mixed[j], -3., 3.);
        }
    }
}
