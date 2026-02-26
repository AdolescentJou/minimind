建议学习次序（按“先能跑通理解，再深入细节”）
第一遍（最重要）：MiniMindForCausalLM → MiniMindModel → MiniMindBlock
第二遍（注意力主线）：Attention → apply_rotary_pos_emb / precompute_freqs_cis → repeat_kv → KV cache 逻辑
第三遍（MLP 主线）：FeedForward
第四遍（可选进阶：MoE）：MoEGate → MOEFeedForward（训练路径 vs 推理 moe_infer）→ aux_loss 是怎么汇总回去的

MiniMindConfig
   │  (提供超参数：hidden_size/层数/头数/RoPE/MoE等)
   ▼
MiniMindForCausalLM  (训练/推理入口；HF generate 的封装)
   ├─ lm_head: Linear(H → V)
   └─ model: MiniMindModel  (Transformer 主干 backbone)
         ├─ embed_tokens: Embedding(V → H)
         ├─ RoPE表: precompute_freqs_cis(...) → freqs_cos/freqs_sin
         ├─ layers: [MiniMindBlock] * num_hidden_layers
         │      ├─ input_layernorm: RMSNorm
         │      ├─ self_attn: Attention
         │      │      ├─ q/k/v/o_proj (Linear)
         │      │      ├─ apply_rotary_pos_emb(q,k,cos,sin)  (RoPE)
         │      │      ├─ repeat_kv(...)  (GQA/MQA)
         │      │      └─ (可选) past_key_value / use_cache  (KV cache)
         │      ├─ post_attention_layernorm: RMSNorm
         │      └─ mlp:
         │           ├─ FeedForward                 (use_moe=False)
         │           └─ MOEFeedForward              (use_moe=True)
         │                ├─ gate: MoEGate → topk_idx/topk_weight/aux_loss
         │                ├─ experts: [FeedForward] * n_routed_experts
         │                └─ shared_experts: [FeedForward] * n_shared_experts (可选)
         └─ norm: RMSNorm



下次想启动大模型，输入问题，是怎么个流程/走哪个文件？
你仓库里已经有推理入口了，主要有 3 个选择：
命令行对话/推理：eval_llm.py
流程是：加载 tokenizer → 加载权重 → 组织 prompt（apply_chat_template；如果是 pretrain 权重则用更简单的拼接）→ tokenizer(...) → model.generate(...) → decode 输出。
本地网页聊天（Streamlit）：scripts/web_demo.py
流程类似，但把生成改成流式显示（TextIteratorStreamer），可选“本地模型”或“API”。
OpenAI API 兼容服务（FastAPI）：scripts/serve_openai_api.py
启动后你可以用 OpenAI SDK/任何兼容客户端请求 /v1/chat/completions，内部还是：messages → chat template → tokenize → generate → 返回（支持 stream）。
如果你告诉我你训练出的权重前缀（例如 pretrain）和 hidden_size（例如 512），我可以把对应的启动命令也按你当前目录结构直接写出来。


下一次需要关注的东西
1. 训练全流程 <-> 关联
2. 完整训练一次，然后看一下启动流程，接口流程等