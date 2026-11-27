# 模型目录

本目录保留三个模型的配置、分词器和模型卡，权重文件不纳入仓库。目录中的三个链接指向 `.shared/` 下的同名目录，以匹配 `config/casp_mem.yaml`。检出后需要联网获取权重，模型总大小约 17 GB。

| 本地目录 | 上游模型 |
| --- | --- |
| `model/Qwen3-8B` | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| `model/Qwen3-Embedding-0.6B` | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| `model/mDeBERTa-v3-base-mnli-xnli` | [MoritzLaurer/mDeBERTa-v3-base-mnli-xnli](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli) |

从项目根目录安装 `huggingface_hub` 后执行：

```bash
python -m pip install huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download

for repo_id, directory in [
    ("Qwen/Qwen3-8B", "model/.shared/Qwen3-8B"),
    ("Qwen/Qwen3-Embedding-0.6B", "model/.shared/Qwen3-Embedding-0.6B"),
    ("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", "model/.shared/mDeBERTa-v3-base-mnli-xnli"),
]:
    snapshot_download(repo_id=repo_id, local_dir=directory)
PY
```

命令获取上游当前版本，可能覆盖仓库中保存的模型配置；如需严格复现实验，应记录并使用上游具体 revision。Qwen3-8B 还需要单独启动兼容的推理服务，下载权重不会自动启动服务。
