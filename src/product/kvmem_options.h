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

inline HostKVArchiveMode parse_host_archive_mode(std::string_view value) {
    if (value == "auto") return HostKVArchiveMode::Auto;
    if (value == "pinned") return HostKVArchiveMode::Pinned;
    if (value == "pageable") return HostKVArchiveMode::Pageable;
    throw std::invalid_argument("invalid kvmem-host-archive: " + std::string(value));
}
} // namespace ninfer::product
