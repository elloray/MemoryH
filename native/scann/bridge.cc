#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>

#include "scann/scann_ops/cc/scann.h"
#include "scann_core/config_builder.h"

using research_scann::ScannInterface;

namespace {
void error_message(char* out, size_t capacity, const std::string& message) {
  std::snprintf(out, capacity, "%s", message.c_str());
}
}  // namespace

// Fixed ABI: row-major, unit-normalized float32 vectors; callers own the output buffers.
extern "C" __attribute__((visibility("default"))) void* memoryh_scann_build_v1(
    const float* rows, uint32_t count, uint32_t dim, char* error,
    size_t capacity) noexcept {
  try {
    const int k = std::min<uint32_t>(9, count);
    scann_core::ConfigBuilder builder(k, scann_core::DistanceMeasure::kDotProduct, dim);
    scann_core::TreeOptions tree;
    tree.num_leaves = count == 1 ? 1 : std::max(2, static_cast<int>(std::sqrt(count)));
    // For small stores all nine requested candidates may span every partition.
    tree.num_leaves_to_search = count < 18 ? tree.num_leaves : (tree.num_leaves + 1) / 2;
    tree.min_partition_size = 1;
    tree.random_init = false;
    builder.Tree(tree);
    auto config = builder.ScoreBruteForce().BuildText(count);
    if (!config.ok()) {
      error_message(error, capacity, config.status().ToString());
      return nullptr;
    }
    auto index = std::make_unique<ScannInterface>();
    auto status = index->Initialize(absl::Span<const float>(rows, size_t(count) * dim),
                                    count, *config, /*training_threads=*/1);
    if (!status.ok()) {
      error_message(error, capacity, status.ToString());
      return nullptr;
    }
    return index.release();
  } catch (const std::exception& ex) {
    error_message(error, capacity, ex.what());
    return nullptr;
  } catch (...) {
    error_message(error, capacity, "ScaNN build failed");
    return nullptr;
  }
}

extern "C" __attribute__((visibility("default"))) int memoryh_scann_search_v1(
    const void* handle, const float* query, uint32_t dim, int k,
    uint32_t* ids, char* error, size_t capacity) noexcept {
  try {
    const auto* index = static_cast<const ScannInterface*>(handle);
    research_scann::DatapointPtr<float> point(nullptr, query, dim, dim);
    research_scann::NNResultsVector hits;
    auto status = index->Search(point, &hits, k, -1, -1);
    if (!status.ok() || hits.size() > static_cast<size_t>(k)) {
      error_message(error, capacity, status.ok() ? "invalid ScaNN result size" : status.ToString());
      return -1;
    }
    for (size_t i = 0; i < hits.size(); ++i) ids[i] = hits[i].first;
    return static_cast<int>(hits.size());
  } catch (const std::exception& ex) {
    error_message(error, capacity, ex.what());
    return -1;
  } catch (...) {
    error_message(error, capacity, "ScaNN search failed");
    return -1;
  }
}

extern "C" __attribute__((visibility("default"))) void memoryh_scann_drop_v1(void* handle) noexcept {
  delete static_cast<ScannInterface*>(handle);
}
