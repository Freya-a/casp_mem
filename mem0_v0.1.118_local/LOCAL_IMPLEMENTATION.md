# 本地版实施记录

目标：在独立副本中使用 Qwen3-8B、独立向量模型和本地向量数据库，保留 Mem0 0.1.118 的核心提取、检索、更新逻辑及提示词。

方案：以 `local_memory.py` 构造显式本机配置；以本机 loopback 单模型服务对接上游 vLLM/OpenAI 兼容客户端；在 CPU 上通过现有 HuggingFace embedder 加载本地模型；Qdrant 使用 on_disk 持久化。相比将两个模型都放 GPU，CPU embedding 为 24 GiB 显卡保留生成上下文空间；相比改写 Mem0 算法，这一方案保留原版决策与提示词。

实施检查：复制源目录（不复制其 Git 元数据）；独立 venv；下载并核对 embedding 模型清单；添加 loopback 限制与本地配置测试；验证持久化、离线写入、检索和模型推断更新；记录实际输出与本地 socket 目标。GPU 服务为实验用途的简化串行 HTTP 实现，非通用生产服务。
