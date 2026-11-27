# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.8.0", "transformers==4.53.3", "onnx==1.19.1", "onnxruntime==1.23.2", "numpy<2.4"]
# ///
"""Build a native Metal scoring worker and trace the same FP16 head as the ONNX export."""

import argparse
import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path

# Unsupported model ops must fail visibly.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def prepare(source_dir, output):
    spec = importlib.util.spec_from_file_location(
        "yuanbao_export", Path(__file__).with_name("export-yuanbao.py")
    )
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal device unavailable")
    output.mkdir(parents=True, exist_ok=True)
    torch_dir = Path(torch.__file__).resolve().parent
    lib = torch_dir / "lib"
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        stage = Path(tmp)
        source = stage / "source"
        source.mkdir()
        exporter.prepare_source(source, source_dir)
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=torch.float16,
            attn_implementation="eager",
            local_files_only=True,
        ).eval()
        labels = [tokenizer.encode(s, add_special_tokens=False) for s in ("no", "yes")]
        if any(len(label) != 1 for label in labels):
            raise RuntimeError("expected single-token decision labels")
        head = (
            exporter.DecisionHead(model, [label[0] for label in labels])
            .eval()
            .to("mps")
        )
        del model

        def batch(texts):
            sample = tokenizer(texts, padding=True, return_tensors="pt")
            return sample.input_ids.to("mps"), sample.attention_mask.to("mps")

        with torch.inference_mode():
            traced = torch.jit.trace(
                head, batch(["once", "second example"]), check_trace=False
            )
            for texts in [
                ["one"],
                ["short", "a different and longer text", "third sentence"],
            ]:
                inputs = batch(texts)
                torch.testing.assert_close(
                    traced(*inputs), head(*inputs), atol=0.02, rtol=0.01
                )
            traced.save(str(stage / "yuanbao-mps.pt"))
        subprocess.run(
            [
                "xcrun",
                "clang++",
                "-std=c++17",
                "-O2",
                "-I",
                str(torch_dir / "include"),
                "-I",
                str(torch_dir / "include/torch/csrc/api/include"),
                str(Path(__file__).with_name("yuanbao-mps.cpp")),
                "-L",
                str(lib),
                "-ltorch",
                "-ltorch_cpu",
                "-lc10",
                "-Wl,-rpath," + str(lib),
                "-o",
                str(stage / "yuanbao-mps"),
            ],
            check=True,
        )
        for name in ("yuanbao-mps.pt", "yuanbao-mps"):
            (stage / name).replace(output / name)
    print(
        f"Prepared {exporter.MODEL}@{exporter.REVISION} for Metal in {output}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
