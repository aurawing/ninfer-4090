#include "targets/qwen3_6/impl/vision/cpu_vision_bridge.h"

#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace ninfer::targets::qwen3_6;

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

template<class F> void rejects(F&& f) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid visual input was accepted");
}

// Independent coordinate oracle: each channel/frame/pixel has a distinct exact float value.
float pixel(int frame, int channel, int y, int x) {
    return static_cast<float>(frame * 1000000 + channel * 100000 + y * 1000 + x);
}

std::vector<float> prepared(int groups, int gh, int gw, bool image) {
    std::vector<float> patches;
    for (int group = 0; group < groups; ++group) {
        for (int by = 0; by < gh; by += 2) {
            for (int bx = 0; bx < gw; bx += 2) {
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        for (int channel = 0; channel < 3; ++channel) {
                            for (int t = 0; t < 2; ++t) {
                                for (int y = 0; y < 16; ++y) {
                                    for (int x = 0; x < 16; ++x) {
                                        patches.push_back(pixel(image ? 0 : group * 2 + t,
                                            channel, (by + dy) * 16 + y, (bx + dx) * 16 + x));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return patches;
}

void check_video() {
    constexpr int gh = 2, gw = 6, groups = 3;
    const auto patches = prepared(groups, gh, gw, false);
    const CpuVisionInput input{patches, groups, gh, gw, true};
    const auto shape = validate_cpu_vision_input(input);
    require(shape.merged_tokens == groups * gh * gw / 4, "incorrect merged count");
    for (int group = 0; group < groups; ++group) {
        const auto frames = unpack_cpu_vision_group(input, group);
        require(frames.size() == 2, "video temporal group needs two frames");
        for (int t = 0; t < 2; ++t) {
            require(frames[t].width == gw * 16 && frames[t].height == gh * 16,
                    "frame geometry changed");
            for (int y = 0; y < gh * 16; ++y) {
                for (int x = 0; x < gw * 16; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        const auto index = static_cast<std::size_t>((y * gw * 16 + x) * 3 + c);
                        require(frames[t].rgb[index] == pixel(group * 2 + t, c, y, x),
                                "patch-to-RGB channel/frame/spatial mapping is wrong");
                    }
                }
            }
        }
    }
}

void check_image_and_invalid() {
    auto patches = prepared(1, 4, 2, true);
    const CpuVisionInput image{patches, 1, 4, 2, false};
    const auto frames = unpack_cpu_vision_group(image, 0);
    require(frames.size() == 1, "still image should expose one repeated frame");
    require(frames[0].rgb[0] == pixel(0, 0, 0, 0), "image value changed");
    require(frames[0].rgb.back() == pixel(0, 2, 63, 31), "image final value changed");
    rejects([&] { unpack_cpu_vision_group(image, 1); });
    rejects([&] { validate_cpu_vision_input({patches, 0, 4, 2, false}); });
    rejects([&] { validate_cpu_vision_input({patches, 1, 3, 2, false}); });
    rejects([&] { validate_cpu_vision_input({patches, 2, 4, 2, false}); });
    rejects([&] { validate_cpu_vision_input({std::span<const float>(patches).first(1),
                                            1, 4, 2, false}); });
    // The temporal replicas of an image are required to agree; do not silently discard one.
    patches[256] += 1.0F;
    rejects([&] { unpack_cpu_vision_group(image, 0); });
}

} // namespace

int main() {
    try { check_video(); check_image_and_invalid(); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
    std::cout << "CPU vision input bridge checks passed\n";
    return 0;
}
