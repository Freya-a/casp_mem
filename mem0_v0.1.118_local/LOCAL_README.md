# Mem0 0.1.118 本地模型实验版

本目录基于 `../mem0_v0.1.118` 的 `v0.1.118`（Git commit `ed5a1e9fc68fcd7671a70c33a540d24573755449`）复制。`mem0/` 中的记忆提取、更新、检索、提示词和原始操作逻辑未改动。本地用法以 `local_memory.create_memory()` 为入口：它显式选择本地模型和 Qdrant、关闭 Mem0 遥测并启用 Hugging Face 离线模式。上游裸 `Memory()` 的默认配置没有改写；请使用本入口，勿依赖上游默认配置来保证本地运行。

## 模型和后端

- 语言模型：`../model/Qwen3-8B`，Hugging Face `Qwen/Qwen3-8B`，既有下载元数据修订 `b968826d9c46dd6066d109eabc6255188de91218`；原始 BF16 权重，单张 RTX 3090 Ti GPU，Transformers 推理。`enable_thinking=False`，temperature=0，top_p=1，max_tokens=1024。该实验服务的输入与输出合计上限为 8192 tokens，每次串行处理一个请求；长上下文和并发应用需另行评估。
- 向量模型：`../model/Qwen3-Embedding-0.6B`，ModelScope 官方 `Qwen/Qwen3-Embedding-0.6B`，commit `dd3e37eb506b3addbd930037796e25d349ae9010`；CPU BF16，1024 维，SentenceTransformers。模型权重 SHA-256：`0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`。下载文件的 13 个文件大小与固定版本的 ModelScope 清单一致；上述 SHA-256 是本机计算值。
- 向量数据库：进程内的 Qdrant local，cosine，持久化在使用者指定的数据目录；历史记录存 SQLite。单个 Qdrant 目录同一时间只能由一个进程打开。
- 环境：`.venv` 使用本机 `/home/shu1004/anaconda3/envs/agent-memory` 的系统包；Python 3.10、PyTorch 2.7.1+cu118、Transformers 5.12.1、sentence-transformers 5.6.0、qdrant-client 1.20.0、OpenAI SDK 1.109.1。该环境依赖此基础环境，不能当成独立可搬运的 venv。

## 启动

在一个终端中运行，保持其运行：

```bash
cd /home/shu1004/ld/system/mem0_halumem_diagnosis/mem0_v0.1.118_local
.venv/bin/python local_llm_server.py
```

出现 `Ready at http://127.0.0.1:18888/v1` 后，在另一个终端运行下面示例。服务仅绑定本机，使用一个简化的 OpenAI 兼容 chat 接口，不提供完整 OpenAI API 或鉴权；本机其他用户能访问此端口。首次加载需数十秒并占用约 16 GiB 显存。

```bash
cd /home/shu1004/ld/system/mem0_halumem_diagnosis/mem0_v0.1.118_local
.venv/bin/python - <<'PY'
from local_memory import create_memory

memory = create_memory()  # 默认数据目录：本目录/local_data
print(memory.add('我住在北京，喜欢喝茶。', user_id='demo'))
print(memory.search('我住在哪里？', user_id='demo'))
print(memory.add('我已经搬到上海，现在住在上海。', user_id='demo'))
print(memory.search('我现在住在哪里？', user_id='demo'))
PY
```

`create_memory('/绝对路径/实验数据目录')` 可指定独立数据目录；创建不同实验时使用不同目录。必须在导入 `mem0` 之前调用该入口；同一进程中要更换数据目录，应启动新进程。上例会持久化真实演示数据，实验时请更换 `user_id` 和目录。

## 验证

2026-11-25 的端到端临时数据测试记录在 `local_validation/latest_result.json`，测试脚本为 `local_validation/smoke.py`。输入巴黎住所与茶饮，得到两条 ADD，检索返回巴黎；随后输入搬往伦敦，原记录产生 UPDATE（同一 ID，从 `Lives in Paris` 变为 `Moved from Paris to London`），模型还额外 ADD `Now live in London`；再次检索首条为伦敦。该模型输出并不保证所有表达都能正确更新。

验证进程拦截 DNS 和 socket 连接，仅允许 loopback；记录了 4 次 `127.0.0.1:18888`，没有发现被拦截的外部连接。语言模型服务使用本机文件并处于离线模式；向量模型在 CPU 同进程运行，Qdrant 是本地文件。临时数据测试后已删除。本地安全配置及 Qdrant 重开持久化另有 `local_tests/` 检查。

复现：先启动语言模型服务，然后执行：

```bash
MEM0_TELEMETRY=false MEM0_DIR=/tmp/mem0-local-tests OMP_NUM_THREADS=8 .venv/bin/python -m pytest -q local_tests
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python local_validation/smoke.py
```

第二条命令使用新的临时数据目录并覆盖最新验证记录；不会触及默认 `local_data`。验证完成后可在服务终端按 Ctrl-C 释放 GPU。
