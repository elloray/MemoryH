#!/usr/bin/env bash
# Prepare the local Qwen encoder and the selected reranker.
set -euo pipefail
repo_dir=$(cd "$(dirname "$0")/.." && pwd)
canonical="$repo_dir/target/memoryh-models"
root=${1:-"$canonical"}
ranker=${2:-yuanbao}
case "$ranker" in
    yuanbao|minilm) ;;
    *) echo "Usage: $0 [MODEL_ROOT] [yuanbao|minilm]" >&2; exit 2 ;;
esac
mkdir -p "$root/qwen"
fetch() {
    local url=$1 digest=$2 output=$3
    if [[ -f "$output" ]] && [[ "$(shasum -a 256 "$output" | cut -d' ' -f1)" == "$digest" ]]; then
        return
    fi
    local tmp
    tmp=$(mktemp "$output.XXXXXX")
    local hub repo rest cached
    hub=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
    repo=${url#https://huggingface.co/}
    rest=${repo#*/resolve/}
    repo=${repo%%/resolve/*}
    for cached in "$canonical/${output#"$root"/}" "$hub/models--${repo//\//--}/snapshots/$rest"; do
        if [[ -f "$cached" ]] && [[ "$(shasum -a 256 "$cached" | cut -d' ' -f1)" == "$digest" ]]; then
            cp "$cached" "$tmp" && mv "$tmp" "$output"
            return
        fi
    done
    if curl --fail --location --silent --show-error --retry 3 "$url" --output "$tmp" && [[ "$(shasum -a 256 "$tmp" | cut -d' ' -f1)" == "$digest" ]]; then
        mv "$tmp" "$output"
    else
        rm -f "$tmp"
        echo "Failed to download or verify $output" >&2
        return 1
    fi
}
hf=https://huggingface.co
fetch "$hf/onnx-community/Qwen3-Embedding-0.6B-ONNX/resolve/c25a394dd583836952667c12f008335071b3f43d/onnx/model_int8.onnx" 6d0ea863f78b4a84afa3c7fcba1ec341572b5e28121aef77b7092b1dfdf679c7 "$root/qwen/model_int8.onnx"
fetch "$hf/Qwen/Qwen3-Embedding-0.6B/resolve/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/tokenizer.json" def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a "$root/qwen/tokenizer.json"
yuanbao_ready() (
    cd "$1" 2>/dev/null || return 1
    [[ -s yuanbao.onnx && -s yuanbao-fp16.data && -s tokenizer.json ]] || return 1
    if [[ -f checksums.sha256 ]]; then
        shasum -a 256 --check --status checksums.sha256
    else
        # Earlier exports of the pinned graph predate the checksum manifest.
        shasum -a 256 --check --status <<'DIGESTS'
25e46ca5110353d90a8f3d30a61d42d8cc60da4cf8038b70a6f5c9d49a8c7205  yuanbao-fp16.data
be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506  tokenizer.json
f590b383701c13d48f0d7e3a9c1dd59e6ce166b4275cfc3a5504776e08e1f582  yuanbao.onnx
DIGESTS
    fi
)
if [[ "$ranker" == yuanbao ]]; then
    if ! yuanbao_ready "$root/yuanbao"; then
        if [[ "$root" != "$canonical" ]] && yuanbao_ready "$canonical/yuanbao"; then
            mkdir -p "$root/yuanbao"
            for asset in yuanbao-fp16.data tokenizer.json yuanbao.onnx; do
                tmp=$(mktemp "$root/yuanbao/$asset.XXXXXX")
                cp "$canonical/yuanbao/$asset" "$tmp" && mv "$tmp" "$root/yuanbao/$asset"
            done
            (cd "$root/yuanbao" && manifest=$(mktemp "checksums.XXXXXX") &&
                shasum -a 256 yuanbao-fp16.data tokenizer.json yuanbao.onnx > "$manifest" &&
                mv "$manifest" checksums.sha256)
            exit 0
        fi
        exporter="$(dirname "$0")/export-yuanbao.py"
        source_dir=${MEMORYH_YUANBAO_SOURCE:-"$(dirname "$0")/../target/yuanbao/source"}
        if [[ -d "$source_dir" ]]; then
            uv run --script "$exporter" "$root/yuanbao" --source "$source_dir"
        else
            uv run --script "$exporter" "$root/yuanbao"
        fi
    fi
    exit 0
fi
mkdir -p "$root/reranker"
reranker=$hf/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/resolve/1427fd652930e4ba29e8149678df786c240d8825
fetch "$reranker/onnx/model_qint8_arm64.onnx" 1825907d6c1a9001ff78124780bbde20a614a8c3df3b63409cf3c72c6fe5c8b4 "$root/reranker/model.onnx"
fetch "$reranker/tokenizer.json" 62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626 "$root/reranker/tokenizer.json"
