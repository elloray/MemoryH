// Native M4 scoring worker. The Rust tokenizer sends [rows, width], IDs and masks
// in little-endian order. Replies are FP32 [no, yes] logits for each row.
#include <torch/script.h>
#include <ATen/Context.h>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 2 || !at::hasMPS()) throw std::runtime_error("expected a model and Metal GPU");
    c10::InferenceMode inference;
    const torch::Device gpu(torch::kMPS);
    // Keep traced scalar constants in their original dtype and device.
    auto model = torch::jit::load(argv[1]);
    model.to(gpu);
    model.eval();
    for (const auto& p : model.parameters()) {
      if (!p.is_mps()) throw std::runtime_error("model parameter outside MPS");
    }
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.write("MPS1", 4).flush();
    std::uint32_t shape[2];
    while (std::cin.read(reinterpret_cast<char*>(shape), sizeof(shape))) {
      const auto rows = shape[0], width = shape[1];
      if (!rows || rows > 8 || !width || width > 512) throw std::runtime_error("invalid batch shape");
      auto input = [&]() {
        auto t = torch::empty({rows, width}, torch::TensorOptions().dtype(torch::kLong));
        if (!std::cin.read(reinterpret_cast<char*>(t.data_ptr<int64_t>()), t.numel() * 8))
          throw std::runtime_error("incomplete tensor");
        return t.to(gpu);
      };
      auto ids = input();
      auto mask = input();
      auto logits = model.forward({ids, mask}).toTensor();
      if (!logits.is_mps() || logits.scalar_type() != torch::kFloat32 ||
          logits.dim() != 2 || logits.size(0) != rows || logits.size(1) != 2)
        throw std::runtime_error("expected MPS float32 [rows, 2] output");
      auto result = logits.cpu().contiguous();  // Synchronous readback included in timing.
      std::cout.write(reinterpret_cast<const char*>(result.const_data_ptr<float>()), rows * 8).flush();
      if (!std::cout) return 1;
    }
    return std::cin.eof() ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "Yuanbao MPS: " << e.what() << '\n';
    return 1;
  }
}
