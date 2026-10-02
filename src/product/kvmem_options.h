#pragma once

#include <ninfer/types.h>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {
inline KvMode parse_kv_mode(std::string_view value) {
    if (value == "dense") return KvMode::Dense;
    if (value == "tiered-exact") return KvMode::TieredExact;
    if (value == "kvmem") throw std::invalid_argument("--kv-mode kvmem requires stage 4 sparse decode (Mean-K scoring and page selection); use tiered-exact for exact validation");
    throw std::invalid_argument("invalid kv-mode: " + std::string(value));
}
inline void validate_kvmem_prefill(std::string_view value) {
    if (value != "exact") throw std::invalid_argument("--kvmem-prefill accepts only exact; window prefill and query replay are unavailable");
}
inline void validate_kvmem_options(KvMode mode, const TieredKVOptions& options,
                                 std::uint32_t concurrency) {
    if (mode == KvMode::KVMem)
        throw std::invalid_argument("--kv-mode kvmem requires stage 4 sparse decode; use tiered-exact");
    if (mode == KvMode::TieredExact && concurrency != 1)
        throw std::invalid_argument("tiered-exact requires --max-concurrency 1 (C=1)");
    if (options.lock_archive && (mode != KvMode::TieredExact || options.host_archive == HostKVArchiveMode::Pinned))
        throw std::invalid_argument("--kvmem-lock-archive requires tiered-exact with auto/pageable archive; CUDA pinned archive already has residency protection");
}

inline HostKVArchiveMode parse_host_archive_mode(std::string_view value) {
    if (value == "auto") return HostKVArchiveMode::Auto;
    if (value == "pinned") return HostKVArchiveMode::Pinned;
    if (value == "pageable") return HostKVArchiveMode::Pageable;
    throw std::invalid_argument("invalid kvmem-host-archive: " + std::string(value));
}
} // namespace ninfer::product
