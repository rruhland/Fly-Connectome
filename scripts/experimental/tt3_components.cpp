// Four-connected components of constant appearance in a small RGB camera frame.
#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>

struct Run {
    int y, left, right, color, parent;
};

static int root(std::vector<Run>& runs, int i) {
    while (runs[i].parent != i) {
        runs[i].parent = runs[runs[i].parent].parent;
        i = runs[i].parent;
    }
    return i;
}

struct Box {
    int color, first, left, right, top, bottom, area;
};

extern "C" int tt3_components(const uint8_t* frame, int height, int width, int* out) {
    std::vector<Run> runs;
    runs.reserve(height * 8);
    std::map<int, int> totals;
    int prev = 0, begin = 0;
    for (int y = 0; y < height; ++y) {
        int prior = prev;
        for (int x = 0; x < width;) {
            const int start = x++;
            const auto* rgb = frame + 3 * (y * width + start);
            const int color = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2];
            while (x < width) {
                const auto* test = frame + 3 * (y * width + x);
                if (test[0] != rgb[0] || test[1] != rgb[1] || test[2] != rgb[2]) break;
                ++x;
            }
            const int index = static_cast<int>(runs.size());
            runs.push_back({y, start, x, color, index});
            totals[color] += x - start;
            while (prior < begin && runs[prior].right <= start) ++prior;
            for (int i = prior; i < begin && runs[i].left < x; ++i) {
                if (runs[i].color == color) {
                    const int a = root(runs, i), b = root(runs, index);
                    runs[std::max(a, b)].parent = std::min(a, b);
                }
            }
        }
        prev = begin;
        begin = static_cast<int>(runs.size());
    }
    std::vector<Box> boxes(runs.size());
    for (int i = 0; i < static_cast<int>(runs.size()); ++i) {
        const auto& r = runs[i];
        const int id = root(runs, i);
        auto& b = boxes[id];
        if (i == id) b = {r.color, i, r.left, r.right, r.y, r.y + 1, 0};
        b.left = std::min(b.left, r.left);
        b.right = std::max(b.right, r.right);
        b.bottom = r.y + 1;
        b.area += r.right - r.left;
    }
    std::vector<Box> kept;
    for (const auto& b : boxes) {
        if (b.area < 2 || b.area > .10 * (height * width) ||
            totals[b.color] > .45 * (height * width) || b.right - b.left == width ||
            b.bottom - b.top == height) continue;
        kept.push_back(b);
    }
    std::sort(kept.begin(), kept.end(), [](const Box& a, const Box& b) {
        return a.color < b.color || (a.color == b.color && a.first < b.first);
    });
    for (const auto& b : kept) {
        *out++ = b.left + b.right;
        *out++ = b.top + b.bottom;
        *out++ = b.color;
        *out++ = b.area;
    }
    return static_cast<int>(kept.size());
}
