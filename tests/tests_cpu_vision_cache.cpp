#include "targets/qwen3_6/impl/vision/cpu_vision_encoder.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace ninfer::targets::qwen3_6;
constexpr std::size_t one_embedding_bytes = 5120 * sizeof(std::uint16_t);

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

template<class Exception, class F> void rejects(F&& function, const char* message) {
    bool rejected = false;
    try { function(); } catch (const Exception&) { rejected = true; }
    require(rejected, message);
}

class FakeEncoder final : public CpuVisionEncoder {
public:
    std::atomic<int> calls = 0;
    mutable std::atomic<int> validations = 0;
    bool reject_budget = false;
    bool fail_next = false;
    bool malformed_next = false;
    bool nonfinite_next = false;
    std::uint64_t identity = 1234;
    std::function<void()> on_encode;

    void validate(const CpuVisionInput& input) const override {
        ++validations;
        (void)validate_cpu_vision_input(input);
        if (reject_budget) { throw std::runtime_error("fake host-memory budget exceeded"); }
    }
    std::vector<std::uint16_t> encode(const CpuVisionInput& input,
                                    const std::function<bool()>& cancelled) override {
        validate(input);
        if (cancelled && cancelled()) { throw CpuVisionCancelled(); }
        ++calls;
        if (on_encode) { on_encode(); }
        if (fail_next) { fail_next = false; throw std::runtime_error("fake encode failure"); }
        const auto shape = validate_cpu_vision_input(input);
        std::vector<std::uint16_t> result(shape.merged_tokens * 5120, 0x3f80);
        if (malformed_next) { malformed_next = false; result.pop_back(); }
        if (nonfinite_next) { nonfinite_next = false; result.front() = 0x7f80; }
        return result;
    }
    std::uint64_t identity_hash() const noexcept override { return identity; }
    std::size_t weight_bytes() const noexcept override { return 5678; }
};

struct Input {
    std::vector<float> patches;
    CpuVisionInput view;
    explicit Input(std::uint32_t fingerprint, int height = 2, int width = 2, bool video = false)
        : patches(static_cast<std::size_t>(height * width) * 1536, 0.0F),
          view{patches, 1, height, width, video, std::array<std::uint8_t, 32>{}} {
        for (int i = 0; i < 4; ++i) {
            (*view.processed_fingerprint)[i] = static_cast<std::uint8_t>(fingerprint >> (i * 8));
        }
    }
};

void check_reuse_and_keys() {
    const auto inner = std::make_shared<FakeEncoder>();
    const auto cached = make_cached_cpu_vision_encoder(inner, one_embedding_bytes * 8);
    require(cached->weight_bytes() == 5678 && cached->identity_hash() == 1234 &&
            cached->cache_capacity_bytes() == one_embedding_bytes * 8, "decorator metadata changed");
    Input first(1);
    const auto initial = cached->encode_with_status(first.view);
    require(inner->validations == 1, "cache miss validated the whole image more than once");
    const auto repeated = cached->encode_with_status(first.view);
    require(inner->validations == 2, "cache hit skipped or repeated backend validation");
    require(!initial.cache_hit && repeated.cache_hit && initial.embeddings == repeated.embeddings &&
            inner->calls == 1, "identical processed image did not reuse exact BF16 embeddings");
    auto owned_copy = repeated.embeddings;
    owned_copy.front() = 0;
    require(cached->encode(first.view).front() == 0x3f80, "caller mutated cache-owned embeddings");
    Input changed(2);
    require(!cached->encode_with_status(changed.view).cache_hit, "changed fingerprint hit");
    inner->identity = 4321;
    require(!cached->encode_with_status(first.view).cache_hit, "changed encoder identity hit");
    inner->identity = 1234;
    require(cached->encode_with_status(first.view).cache_hit, "encoder identities did not remain isolated");
    Input tall(3, 4, 2), wide(3, 2, 4);
    require(!cached->encode_with_status(tall.view).cache_hit &&
            !cached->encode_with_status(wide.view).cache_hit, "changed geometry hit");
    Input video(1, 2, 2, true);
    require(!cached->encode_with_status(video.view).cache_hit &&
            !cached->encode_with_status(video.view).cache_hit, "video was cached");
    require(cached->encode_with_status(first.view).cache_hit, "video bypass replaced image entry");
    Input unknown(4);
    unknown.view.processed_fingerprint.reset();
    require(!cached->encode_with_status(unknown.view).cache_hit &&
            !cached->encode_with_status(unknown.view).cache_hit, "missing fingerprint was cached");
    const auto separate = make_cached_cpu_vision_encoder(std::make_shared<FakeEncoder>(),
                                                        one_embedding_bytes * 8);
    require(!separate->encode_with_status(first.view).cache_hit, "model instances shared cache state");
}

void check_lru_limits() {
    const auto inner = std::make_shared<FakeEncoder>();
    const auto cached = make_cached_cpu_vision_encoder(inner, 2 * one_embedding_bytes);
    Input a(1), b(2), c(3);
    (void)cached->encode(a.view); (void)cached->encode(b.view);
    require(cached->encode_with_status(a.view).cache_hit, "LRU read missed");
    (void)cached->encode(c.view);
    require(cached->encode_with_status(a.view).cache_hit &&
            !cached->encode_with_status(b.view).cache_hit, "byte bound did not evict least recent entry");
    const auto bounded = make_cached_cpu_vision_encoder(std::make_shared<FakeEncoder>(),
                                                       300 * one_embedding_bytes);
    for (std::uint32_t key = 0; key < 257; ++key) {
        Input item(key); (void)bounded->encode(item.view);
    }
    Input oldest(0), newest(256);
    require(bounded->encode_with_status(newest.view).cache_hit &&
            !bounded->encode_with_status(oldest.view).cache_hit, "256-entry bound was exceeded");
}

void check_disabled_and_oversized() {
    Input input(1);
    for (const auto cap : {std::size_t{0}, one_embedding_bytes - 1}) {
        const auto inner = std::make_shared<FakeEncoder>();
        const auto cached = make_cached_cpu_vision_encoder(inner, cap);
        require(!cached->encode_with_status(input.view).cache_hit &&
                !cached->encode_with_status(input.view).cache_hit && inner->calls == 2,
                "disabled or oversized output was retained");
    }
    rejects<std::invalid_argument>([] { (void)make_cached_cpu_vision_encoder({}, 1); },
                                   "null encoder accepted");
}

void check_validation_failures_and_cancellation() {
    const auto inner = std::make_shared<FakeEncoder>();
    const auto cached = make_cached_cpu_vision_encoder(inner, 4 * one_embedding_bytes);
    Input input(1);
    (void)cached->encode(input.view);
    inner->reject_budget = true;
    rejects<std::runtime_error>([&] { (void)cached->encode(input.view); }, "hit skipped memory admission");
    inner->reject_budget = false;
    input.patches.front() = 1.0F;
    rejects<std::invalid_argument>([&] { (void)cached->encode(input.view); }, "hit skipped input validation");
    input.patches.front() = 0.0F;
    rejects<CpuVisionCancelled>([&] { (void)cached->encode(input.view, [] { return true; }); },
                               "cancelled hit returned output");
    int checks = 0;
    rejects<CpuVisionCancelled>([&] { (void)cached->encode(input.view, [&] { return ++checks == 2; }); },
                               "hit ignored cancellation after lookup/copy");
    require(inner->calls == 1, "invalid/cancelled hit reencoded");
    Input retry(2);
    inner->fail_next = true;
    rejects<std::runtime_error>([&] { (void)cached->encode(retry.view); }, "encode failure returned output");
    require(!cached->encode_with_status(retry.view).cache_hit &&
            cached->encode_with_status(retry.view).cache_hit, "failed encode poisoned retry");
    Input cancelled(3);
    bool abort = false;
    inner->on_encode = [&] { abort = true; };
    rejects<CpuVisionCancelled>([&] { (void)cached->encode(cancelled.view, [&] { return abort; }); },
                               "post-encode cancellation returned output");
    inner->on_encode = {};
    require(!cached->encode_with_status(cancelled.view).cache_hit, "cancelled encode was inserted");
    Input malformed(4);
    inner->malformed_next = true;
    rejects<std::runtime_error>([&] { (void)cached->encode(malformed.view); }, "malformed shape accepted");
    inner->nonfinite_next = true;
    rejects<std::runtime_error>([&] { (void)cached->encode(malformed.view); }, "nonfinite BF16 accepted");
    require(!cached->encode_with_status(malformed.view).cache_hit, "malformed output poisoned retry");
}

void check_concurrent_reuse() {
    const auto inner = std::make_shared<FakeEncoder>();
    const auto cached = make_cached_cpu_vision_encoder(inner, one_embedding_bytes);
    Input input(1);
    constexpr int workers = 8;
    std::barrier start(workers);
    std::atomic<int> hits = 0, failures = 0;
    std::vector<std::thread> threads;
    for (int i = 0; i < workers; ++i) {
        threads.emplace_back([&] {
            start.arrive_and_wait();
            try {
                const auto result = cached->encode_with_status(input.view);
                if (result.cache_hit) { ++hits; }
                if (result.embeddings.size() != 5120 || result.embeddings.front() != 0x3f80) {
                    ++failures;
                }
            } catch (...) { ++failures; }
        });
    }
    for (auto& thread : threads) { thread.join(); }
    require(inner->calls == 1 && hits == workers - 1 && failures == 0,
            "concurrent identical requests executed more than one encode");
}
} // namespace

int main() {
    try {
        check_reuse_and_keys(); check_lru_limits(); check_disabled_and_oversized();
        check_validation_failures_and_cancellation(); check_concurrent_reuse();
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    std::cout << "CPU vision image cache checks passed\n";
    return 0;
}
