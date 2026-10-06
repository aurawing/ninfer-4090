#pragma once

#include <ninfer/types.h>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {
inline KvMode parse_kv_mode(std::string_view value) {
    if (value == "dense") return KvMode::Dense;
    if (value == "tiered-exact") return KvMode::TieredExact;
    if (value == "kvmem") return KvMode::KVMem;
    throw std::invalid_argument("invalid kv-mode: " + std::string(value));
}
inline void validate_kvmem_prefill(std::string_view value) {
    if (value != "exact") throw std::invalid_argument("--kvmem-prefill accepts only exact; window prefill and query replay are unavailable");
}
inline void validate_kvmem_options(KvMode mode, const TieredKVOptions& options,
                                 std::uint32_t concurrency) {
    if (mode != KvMode::Dense && concurrency != 1)
        throw std::invalid_argument("tiered-exact/kvmem requires --max-concurrency 1 (C=1)");
    if (mode == KvMode::KVMem && (!options.query_tokens || options.query_tokens > 16 ||
        options.recent_tokens % 64 || options.gen_reserve_tokens % 64))
        throw std::invalid_argument("kvmem requires query-tokens 1..16 and 64-aligned recent/gen-reserve tokens");
    if (options.lock_archive && (mode == KvMode::Dense || options.host_archive == HostKVArchiveMode::Pinned))
        throw std::invalid_argument("--kvmem-lock-archive requires tiered-exact/kvmem with auto/pageable archive; CUDA pinned archive already has residency protection");
}

inline HostKVArchiveMode parse_host_archive_mode(std::string_view value) {
    if (value == "auto") return HostKVArchiveMode::Auto;
    if (value == "pinned") return HostKVArchiveMode::Pinned;
    if (value == "pageable") return HostKVArchiveMode::Pageable;
    throw std::invalid_argument("invalid kvmem-host-archive: " + std::string(value));
}
} // namespace ninfer::product
