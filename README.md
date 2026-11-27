# CASP-Mem

独立的实验代码；基线是从 `../mem0_halumem_diagnosis/mem0_v0.1.118_local/` 复制的、已适配 Qwen3-8B 的 Mem0 0.1.118，**并非未经修改的官方原版**。CASP-Mem 没有修改复制版 `mem0/`；来源和逐文件 SHA-256 见 `docs/mem0-copy-sha256.txt`。本轮没有启动模型服务或使用既有持久化库；模型和真实服务尚未联调。

## 内容和资源

- `casp_mem/`：CASP 方法源码，包括模型客户端、分段、范围定位、补丁、审计和适配层。
- `mem0_v0.1.118_local/`：本地基线源码、许可和配置例；详见 `docs/mem0-copy-sha256.txt`。
- `model/`：三个模型的目录链接、配置、分词器及模型卡。权重未纳入仓库，下载来源和命令见 [model/README.md](model/README.md)。
- `dataset/`：HaluMem-Medium、LoCoMo 和 LongMemEval 数据及来源说明。LongMemEval 完整源文件以 gzip 保存，使用前按 [数据说明](dataset/LongMemEval/README.md) 解压。
- `config/`：完整方法、单开关消融及基线配置；模型路径和服务地址分别配置。
- `scripts/`、`tests/`：运行入口和自动化测试。
- `results/`：现有运行记录；当前仅有一次失败记录，详见 [results/README.md](results/README.md)。为不同实验指定唯一 `experiment`，同名实验会复用同一套存储。数据库和缓存由 `.gitignore` 排除。

## 算法和边界

1. 原文 SHA-256、偏移覆盖、完整标注及标注模型/接口、提示版本、切分规则和前提开关均匹配才复用缓存；不满足时重新标注。更新后的候选文本重新标注后才缓存，标注失败则清空缓存。按标点和条件连接词过切分，Python 字符偏移为 `[start,end)`，严格验证连续性及拼接还原。Qwen 非思考模式标注断言、条件、作用域、修饰关系和前提；结构异常直接拒绝。前提开关关闭时不请求也不使用前提。
2. CPU NLI 对证据与各跨度/前提打分；矛盾达到 `contradiction_threshold` 的是种子，条件跨度与证据的蕴含达到阈值也作为条件触发种子。无种子拒绝。对非种子，反事实与原上下文都排除当前跨度，再取 `max(0, 原蕴含 - 反事实蕴含)`。不以代词过滤任何跨度。
3. `a_i = modification_cost`；`b_i = miss_cost * max(该跨度及前提最大矛盾分, ΔP_i)`；对于标注 `modifies_id=j` 的跨度 i 建立 `j→i`，`w_ji=dependency_weight`。`propagation` 为 λ。精确枚举所有自由二元变量，强制种子为 1，按字典序打破同成本平局；为控制资源，超过 `max_spans`（默认 22）直接拒绝。反事实关闭时 ΔP=0 且不调用相应 NLI。思考模式复核只接受规定标签与理由；选中项不能改为锁定，`UNCERTAIN` 拒绝。
4. 非思考模式仅获得影响跨度和相邻跨度。每个影响跨度恰好一条局部补丁，检查失败最多重生成一次。条件变化必须使用条件分叉；旧、新分支分别须是原文、证据和补丁中可核对的原样子串；锚点必须是原文或证据与补丁均存在的原样子串，且包含保守的时间/条件/地点提示，无法识别即拒绝。该字符规则与模型/NLI 复核不能证明锚点语义一定正确，需要人工与真实模型评估。关闭分叉遇到此类变化拒绝。
5. 原文锁定跨度按原偏移逐字符拼接，接缝不修改它们。审计锁定跨度邻接窗口的蕴含与矛盾，并双向检查候选整体与证据的矛盾。**局部窗口包含目标跨度时蕴含可能偏高**；字符硬锁定只保证被判为锁定的文字不变，不能证明语义正确或影响范围定位准确。NLI 的阈值是可配置的实验默认值，未经真实数据校准；输入超过模型最大长度会被 tokenizer 截断。

消融 `hard_lock=false` 以逐跨度压缩空白、再用单空格合并代替硬锁定，不增加 LLM 调用；普通更新时它可以改变未受影响的字符，仍做最终 NLI 审计；裁剪始终保留锁定字符。`patch_check=false` 跳过补丁级双向 NLI 及其失败触发的一次修复，最终审计仍在。`multi_anchor=false` 不索引或检索分支锚点。候选文本成功审计后会尝试额外标注一次以建立可复用缓存；开启补丁检查只有在失败时才有一次额外修复调用。

## Mem0 接入及恢复

使用 `Mem0Adapter.add(raw_evidence, user_id=...)` 而不是直接调用基线 `Memory.add`。原生 `add()` 在线程池进入 `_add_to_vector_store`，随后模型决策的 `UPDATE` 文本已是整段改写，并且动作处理会捕获异常。适配层复用原生事实抽取及动作提示，自行执行动作，调用 `update(..., raw_evidence=...)` 显式传递本次原文，绝不使用全局证据变量或把模型整段改写当证据。返回的 `memory` 为实际写回文本，并包括补丁与锚点。适配层同步方法在同一实例内用可重入锁串行化；不提供跨进程并发写入或原生接口绕过适配器的保护，实验应单写入者运行。当前仅支持 `user_id` 作用域，图存储被拒绝。

Mem0 推断的 `DELETE` 先由独立 `delete_guard` 用原始证据判断整条、局部或不确定；局部范围交给四步流程重新审查并审计，范围不明或审计拒绝时返回 `RETAIN`。显式调用适配器 `delete` 表示用户要求删除整条，并清理该用户的关联锚点；推断整条删除依赖模型明确确认。模型判断不是形式化证明，真实接入须验证。`PRUNE_OR_ADAPT` 的 `op_mode=PRUNE` 可在已审查影响范围中输出空文本；裁剪时锁定字符始终保持原样。

写回使用复制版 `_update_memory`，传入合并后的原有元数据及版本、跨度缓存、期望锚点，保留 SQLite 历史。独立锚点集合保存父 ID、用户、版本及嵌入空间；检索丢弃缺失父记录、用户不符及与各自父记录版本不符的锚点。入口使用同一嵌入器并检查两侧均为余弦距离后，对同一父记录取主记录与有效锚点最高分，再去重排序；无法确认可比较时主记录优先、锚点独立排序。旧嵌入空间锚点不会参与新空间检索；旧主记录须先通过适配器更新以写入新空间标识，之后可用 `recover` 重建对应锚点；切换模型或原地替换模型权重后也应更换嵌入空间标识并重建。主记录与锚点并非原子事务：`PartialSyncError` 表示主记录已写、锚点可能未同步。期望锚点已保存在主元数据，可用下面的 `recover` 幂等重建。`PartialWriteError` 表示主向量可能已写但历史写入失败，必须检查 SQLite 历史；恢复锚点不自动修复历史。删除失败可能发生在锚点或主记录操作之后，若主记录仍存在可 `recover`，否则需检查历史与锚点。多动作批次部分执行后失败会抛 `PartialBatchError`，携带已完成动作。请勿把这些状态当作原子回滚。

## 运行

先放入三个本地模型，配置合适的 Qwen OpenAI-compatible 服务地址与 served model name。按 `pyproject.toml` 安装依赖到独立 Python 3.10+ 环境，并可在新环境安装复制的 `mem0_v0.1.118_local` 以提供基线依赖及 `mem0ai` 包元数据（本机现有旧项目 `.venv` 可用于离线测试；它不会被复制或修改）。实际服务需支持 `response_format=json_object` 及 `chat_template_kwargs.enable_thinking`，并返回可识别的思考内容；否则调用会报错，不会静默伪造标注。旧项目现有 `local_llm_server.py` 固定关闭思考模式、忽略该参数，不能直接承担完整方法的语义终审；需另行提供支持思考模式的兼容服务后联调。

```bash
cd ~/ld/system/casp_mem_project
PY=../mem0_halumem_diagnosis/mem0_v0.1.118_local/.venv/bin/python
$PY scripts/run_casp_update.py --check-import
$PY -m pytest tests -q
# 下列命令会加载本地 CPU 模型、调用已配置的 Qwen 服务并写入本项目独立存储：
$PY scripts/run_casp_update.py --user-id example --operation add --evidence '原始新证据'
$PY scripts/run_casp_update.py --user-id example --operation update --memory-id ID --evidence '原始新证据'
$PY scripts/run_casp_update.py --user-id example --operation search --evidence '检索文本'
$PY scripts/run_casp_update.py --user-id example --operation recover --memory-id ID
# 消融或基线：
$PY scripts/run_casp_update.py --config config/ablations/no_counterfactual.yaml --user-id example --evidence '原始新证据'
$PY scripts/run_casp_update.py --config config/mem0_baseline.yaml --user-id example --evidence '原始新证据'
```

每次正常进入执行阶段后将配置、模型标识、随机种子、状态及错误或结果写入该实验结果目录。日志可能包含记忆文本，请妥善管理。`--check-import` 只验证源码导入位置，不加载模型；正常运行创建新的实验数据库。基线 CLI 支持 add/search；CASP CLI 还支持 update/search/delete/recover。要恢复部分同步，使用原实验的同一配置与 user ID。

服务器恢复后应先用独立实验目录验证 Qwen 服务的思考开关与 JSON 解析、NLI 标签和阈值、嵌入维度及 Qdrant API、原生动作决策与实际数据上的端到端写入/恢复，再进行数据集实验。本项目离线 mock 样例不是正式实验数据，不能据此声称方法有效或真实服务已联调。
