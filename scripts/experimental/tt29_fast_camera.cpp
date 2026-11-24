// Preserve the RGB mask, 8-connected components and scoring of ball_pixels.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" bool ball_component(const uint8_t* frame, int height, int width,
                               ptrdiff_t row, ptrdiff_t col, ptrdiff_t channel,
                               const double* prediction, double* result) {
    std::vector<uint8_t> mask(static_cast<size_t>(height) * width, 0);
    std::vector<int> colored;
    for (int y = 0; y < height; ++y) {
        const uint8_t* pixel = frame + y * row;
        for (int x = 0; x < width; ++x, pixel += col) {
            const int r = pixel[0], g = pixel[channel], b = pixel[2 * channel];
            const int chroma = r - b, green = 100 * (g - b);
            if (chroma >= 64 && green > 25 * chroma && green < 62 * chroma && 100 * b < 24 * r) {
                const int index = y * width + x;
                mask[index] = 1;
                colored.push_back(index);
            }
        }
    }
    const int span = std::min(width, height) / 3;
    std::vector<int> connected;
    double best = std::numeric_limits<double>::infinity();
    for (const int index : colored) {
        if (!mask[index]) continue;
        connected.clear();
        connected.push_back(index);
        mask[index] = 0;
        int x_min = width, x_max = 0, y_min = height, y_max = 0;
        int64_t x_sum = 0, y_sum = 0;
        for (size_t i = 0; i < connected.size(); ++i) {
            const int y = connected[i] / width, x = connected[i] % width;
            x_min = std::min(x_min, x); x_max = std::max(x_max, x);
            y_min = std::min(y_min, y); y_max = std::max(y_max, y);
            x_sum += x; y_sum += y;
            for (int yy = std::max(0, y - 1); yy <= std::min(height - 1, y + 1); ++yy) {
                for (int xx = std::max(0, x - 1); xx <= std::min(width - 1, x + 1); ++xx) {
                    const int neighbor = yy * width + xx;
                    if (mask[neighbor]) {
                        mask[neighbor] = 0;
                        connected.push_back(neighbor);
                    }
                }
            }
        }
        const int box_w = x_max - x_min + 1, box_h = y_max - y_min + 1;
        if (std::max(box_w, box_h) > span) continue;
        const double area = static_cast<double>(connected.size());
        const double x = x_min + static_cast<double>(x_sum - x_min * static_cast<int64_t>(area)) / area;
        const double y = y_min + static_cast<double>(y_sum - y_min * static_cast<int64_t>(area)) / area;
        constexpr double pi = 3.14159265358979323846;
        const double radius = std::sqrt(area / pi);
        const double compact = static_cast<double>(std::min(box_w, box_h)) / std::max(box_w, box_h)
                               * area / (box_w * box_h);
        double cost = -std::min(radius, 9.) - 3. * compact;
        if (prediction) {
            const double dx = x - prediction[0], dy = y - prediction[1];
            cost += std::min(25., std::sqrt(dx * dx + dy * dy) * .3);
        }
        if (cost < best) {
            best = cost;
            result[0] = x; result[1] = y; result[2] = radius; result[4] = area;
            result[3] = x_min == 0 || y_min == 0 || x_max == width - 1 || y_max == height - 1 || compact < .48;
        }
    }
    return std::isfinite(best);
}
