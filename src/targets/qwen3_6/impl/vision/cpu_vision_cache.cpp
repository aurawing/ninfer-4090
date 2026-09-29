#include "cpu_vision_encoder.h"

#include <algorithm>
#include <list>
#include <mutex>
#include <utility>

namespace ninfer::targets::qwen3_6 {
namespace {

constexpr std::size_t maximum_entries = 256;
constexpr std::size_t output_hidden = 5120;

void check_cancelled(const std::function<bool()>& cancelled) {
    if (cancelled && cancelled()) { throw CpuVisionCancelled(); }
}

struct CacheKey {
    std::array<std::uint8_t, 32> fingerprint;
    std::uint64_t encoder_identity;
    std::int32_t temporal;
    std::int32_t height;
    std::int32_t width;
    bool video;
    bool operator==(const CacheKey&) const = default;
};

struct CacheEntry {
    CacheKey key;
    // A sized array avoids retaining spare vector capacity outside the byte limit.
    std::unique_ptr<std::uint16_t[]> embeddings;
    std::size_t elements;
};

static_assert(maximum_entries * (sizeof(CacheEntry) + 2 * sizeof(void*)) + 1024 <
              cpu_vision_cache_metadata_bytes);

class CachedCpuVisionEncoder final : public CpuVisionEncoder {
public:
    CachedCpuVisionEncoder(std::shared_ptr<CpuVisionEncoder> inner, std::size_t capacity)
        : inner_(std::move(inner)), capacity_(capacity) {}

    void validate(const CpuVisionInput& input) const override {
        std::lock_guard lock(mutex_);
        inner_->validate(input);
    }

    std::vector<std::uint16_t> encode(const CpuVisionInput& input,
                                     const std::function<bool()>& cancelled) override {
        return encode_with_status(input, cancelled).embeddings;
    }

    CpuVisionEncodeResult encode_with_status(const CpuVisionInput& input,
                                             const std::function<bool()>& cancelled) override {
        // The underlying CPU encoder uses shared scratch. Keep validation, lookup and encoding
        // under the same lock so concurrent identical images execute the encoder only once.
        std::lock_guard lock(mutex_);
        check_cancelled(cancelled);
        inner_->validate(input); // Includes host-memory admission even on a hit.
        const auto shape = validate_cpu_vision_input(input);
        const bool eligible = capacity_ != 0 && !input.video && input.processed_fingerprint;
        const CacheKey key{input.processed_fingerprint.value_or(std::array<std::uint8_t, 32>{}),
                           inner_->identity_hash(), input.temporal, input.height, input.width,
                           input.video};
        if (eligible) {
            const auto found = std::find_if(entries_.begin(), entries_.end(),
                [&](const CacheEntry& entry) { return entry.key == key; });
            if (found != entries_.end()) {
                std::vector<std::uint16_t> result(found->embeddings.get(),
                                                  found->embeddings.get() + found->elements);
                check_cancelled(cancelled);
                entries_.splice(entries_.begin(), entries_, found);
                return {std::move(result), true};
            }
        }

        check_cancelled(cancelled);
        auto result = inner_->encode(input, cancelled);
        check_cancelled(cancelled);
        if (result.size() / output_hidden != shape.merged_tokens ||
            result.size() % output_hidden != 0) {
            throw std::runtime_error("CPU vision output shape does not match the input grid");
        }
        if (std::any_of(result.begin(), result.end(), [](std::uint16_t value) {
                return (value & 0x7f80U) == 0x7f80U;
            })) {
            throw std::runtime_error("CPU vision output contains a nonfinite BF16 value");
        }
        // Division avoids overflow when comparing the allocation extent against the byte cap.
        if (eligible && result.size() <= capacity_ / sizeof(std::uint16_t)) {
            check_cancelled(cancelled);
            const auto bytes = result.size() * sizeof(std::uint16_t);
            while (!entries_.empty() &&
                   (entries_.size() >= maximum_entries || bytes > capacity_ - used_bytes_)) {
                used_bytes_ -= entries_.back().elements * sizeof(std::uint16_t);
                entries_.pop_back();
            }
            auto owned = std::make_unique<std::uint16_t[]>(result.size());
            std::copy(result.begin(), result.end(), owned.get());
            check_cancelled(cancelled);
            entries_.push_front({key, std::move(owned), result.size()});
            used_bytes_ += bytes;
        }
        return {std::move(result), false};
    }

    std::size_t cache_capacity_bytes() const noexcept override { return capacity_; }
    std::uint64_t identity_hash() const noexcept override { return inner_->identity_hash(); }
    std::size_t weight_bytes() const noexcept override { return inner_->weight_bytes(); }

private:
    std::shared_ptr<CpuVisionEncoder> inner_;
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::list<CacheEntry> entries_;
    std::size_t used_bytes_ = 0;
};

} // namespace

std::shared_ptr<CpuVisionEncoder> make_cached_cpu_vision_encoder(
    std::shared_ptr<CpuVisionEncoder> inner, std::size_t cache_capacity_bytes) {
    if (!inner) { throw std::invalid_argument("CPU vision cache requires an encoder"); }
    if (cache_capacity_bytes == 0) { return inner; }
    return std::make_shared<CachedCpuVisionEncoder>(std::move(inner), cache_capacity_bytes);
}

} // namespace ninfer::targets::qwen3_6
