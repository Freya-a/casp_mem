# LongMemEval-S cleaned

- Source: [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned), revision `98d7416c24c778c2fee6e6f3006e7a073259d48f`.
- Original file: [longmemeval_s_cleaned.json](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json), downloaded without modification. 500 samples; 277,383,467 bytes.
- Original SHA-256: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`, verified against Hugging Face's LFS metadata.
- `longmemeval_s_knowledge_update.json` is a derived JSON array of the 78 samples with `question_type == "knowledge-update"`, in original order, preserving every original field and nested conversation order. The output is reserialized JSON, not a byte-for-byte extract.
- Subset SHA-256: `1dcb1253b357b1fd9609ef5a17b228bcd14f8cee9e8317a7cd4c23d5c0a5c60d`.
- License: MIT, as specified by the Hugging Face dataset card; the upstream [LongMemEval MIT license](https://github.com/xiaowu0162/LongMemEval/blob/main/LICENSE) is reproduced as `LICENSE` (copyright 2024 Di Wu).
- Prepared and validated: 2026-11-26. Every selected record was re-parsed and compared against the corresponding complete source object.

The full source file is stored in this repository as `longmemeval_s_cleaned.json.gz`. Restore it from the project root with:

```bash
gzip -dk dataset/LongMemEval/longmemeval_s_cleaned.json.gz
sha256sum dataset/LongMemEval/longmemeval_s_cleaned.json
```

The expected SHA-256 is the original hash above. The decompressed file is ignored by Git.
