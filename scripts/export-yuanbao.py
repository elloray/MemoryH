# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.8.0", "transformers==4.53.3", "onnx==1.19.1", "onnxruntime==1.23.2", "numpy<2.4"]
# ///
"""Prepare the pinned Yuanbao weights for the local CPU runtime. No Python at query time."""

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Yuanbao-Search/Yuanbao-Reranker-0.6B"
REVISION = "0101299a7937a016848d8d55b092f6e31cfa312d"

SOURCE_HASHES = {
    "config.json": "48d60f3a8b44c23dc5fc76a247ccfc3876cf011472eda340bc5d1588620c5938",
    "tokenizer_config.json": "397bdca5697f44ab70a567f196203e396fc904b1891bb79fc107e304fb3d8f2c",
    "tokenizer.json": "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506",
    "model.safetensors": "5518577ec411e854485f0424a67dd93729a17c1f9f500200f4ce3438ee3477e1",
}


def matches(path, digest):
    if not path.is_file():
        return False
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest() == digest


def prepare_source(stage, preferred):
    for name, digest in SOURCE_HASHES.items():
        local = preferred / name if preferred else None
        if local is None or not matches(local, digest):
            cached = try_to_load_from_cache(MODEL, name, revision=REVISION)
            if isinstance(cached, str) and matches(Path(cached), digest):
                local = Path(cached)
            else:
                local = Path(
                    hf_hub_download(
                        MODEL,
                        name,
                        revision=REVISION,
                        force_download=isinstance(cached, str),
                    )
                )
                if not matches(local, digest):
                    raise ValueError(f"{name} does not match {MODEL}@{REVISION}")
        (stage / name).symlink_to(local.resolve())


class DecisionHead(torch.nn.Module):
    def __init__(self, model, decision_ids):
        super().__init__()
        self.backbone = model.model
        self.register_buffer(
            "decision_weights",
            model.get_output_embeddings().weight[decision_ids].detach(),
        )

    def forward(self, input_ids, attention_mask):
        # Right padding, per-row RoPE positions and the last non-pad token.
        length = input_ids.shape[1]
        positions = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        steps = torch.arange(length, device=input_ids.device)
        allowed = (steps[:, None] >= steps[None, :]) & attention_mask[:, None, :].bool()
        bias = (~allowed[:, None]).to(self.decision_weights.dtype) * torch.finfo(
            self.decision_weights.dtype
        ).min
        hidden = self.backbone(
            input_ids,
            attention_mask={"full_attention": bias},
            position_ids=positions,
            use_cache=False,
        ).last_hidden_state
        last = (attention_mask.sum(-1) - 1)[:, None, None].expand(
            -1, 1, hidden.shape[-1]
        )
        return torch.nn.functional.linear(
            hidden.gather(1, last).squeeze(1), self.decision_weights
        ).float()


def export(source, output):
    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokens = [
        tokenizer.encode(word, add_special_tokens=False) for word in ("no", "yes")
    ]
    if any(len(item) != 1 for item in tokens):
        raise ValueError("expected single-token no/yes labels")
    decision_ids = [item[0] for item in tokens]
    head = DecisionHead(model, decision_ids).eval()
    sample = tokenizer(
        ["一次校验", "长一点的第二条原文"], padding=True, return_tensors="pt"
    )
    args = (sample.input_ids, sample.attention_mask)
    graph_path = output / "yuanbao.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            head,
            args,
            str(graph_path),
            export_params=False,
            do_constant_folding=False,
            dynamo=False,
            opset_version=17,
            input_names=["input_ids", "attention_mask"],
            output_names=["decision_logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "decision_logits": {0: "batch"},
            },
        )
    # Keep the protobuf small and stream the traced parameters to an external weight file.
    graph = onnx.load(graph_path)
    weights = head.state_dict()
    weight_name = "yuanbao-fp16.data"
    with (output / weight_name).open("wb") as file:
        for value in list(graph.graph.input):
            if value.name in ("input_ids", "attention_mask"):
                continue
            data = weights[value.name].detach().contiguous().numpy()
            tensor = onnx.TensorProto(
                name=value.name,
                dims=data.shape,
                data_type=onnx.helper.np_dtype_to_tensor_dtype(data.dtype),
                raw_data=b"x",
            )
            onnx.external_data_helper.set_external_data(
                tensor, location=weight_name, offset=file.tell(), length=data.nbytes
            )
            tensor.ClearField("raw_data")
            file.write(memoryview(data).cast("B"))
            graph.graph.initializer.append(tensor)
            graph.graph.input.remove(value)
    onnx.save_model(graph, graph_path)
    onnx.checker.check_model(str(graph_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(graph_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    for texts in (
        ["单句验证"],
        ["hi", "这条候选记忆比前一条长，用来检验补齐后最后一个有效位置。"],
    ):
        batch = tokenizer(texts, padding=True, return_tensors="pt")
        mask = batch.attention_mask
        with torch.inference_mode():
            logits = model(
                **batch,
                position_ids=(mask.cumsum(-1) - 1).clamp(min=0),
                use_cache=False,
            ).logits
            reference = (
                logits[torch.arange(len(texts)), mask.sum(-1) - 1][:, decision_ids]
                .float()
                .numpy()
            )
        actual = session.run(
            None, {"input_ids": batch.input_ids.numpy(), "attention_mask": mask.numpy()}
        )[0]
        np.testing.assert_allclose(actual, reference, atol=0.15, rtol=0.02)
        print(
            f"Verified batch={len(texts)}, max logit difference={np.abs(actual - reference).max():.4f}",
            flush=True,
        )
    shutil.copyfile(source / "tokenizer.json", output / "tokenizer.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--source",
        type=Path,
        help="reuse matching files from this local source before checking the Hub cache",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output) as tmp:
        stage = Path(tmp)
        source = stage / "source"
        source.mkdir()
        prepare_source(source, args.source)
        export(source, stage)
        assets = ("yuanbao-fp16.data", "tokenizer.json", "yuanbao.onnx")
        with (stage / "checksums.sha256").open("w") as manifest:
            for name in assets:
                with (stage / name).open("rb") as file:
                    manifest.write(
                        f"{hashlib.file_digest(file, 'sha256').hexdigest()}  {name}\n"
                    )
        for name in (*assets, "checksums.sha256"):
            (stage / name).replace(args.output / name)
