// Prints sizeof/offsetof for the llama.h structs the llikert adapter passes by value
// or pointer, as JSON, so ctypes layouts can be checked against the pinned header.
#include <stddef.h>
#include <stdio.h>
#include "llama.h"

#define F(S, M) printf("%s\"%s\": %zu", first ? "" : ", ", #M, offsetof(struct S, M)), first = 0
#define BEGIN(S) do { int first = 1; printf("%s\"%s\": {\"sizeof\": %zu, \"fields\": {", nstruct++ ? ", " : "", #S, sizeof(struct S))
#define END() printf("}}"); } while (0)

int main(void) {
    int nstruct = 0;
    printf("{");
    BEGIN(llama_model_params);
    F(llama_model_params, devices); F(llama_model_params, tensor_buft_overrides);
    F(llama_model_params, n_gpu_layers); F(llama_model_params, split_mode);
    F(llama_model_params, load_mode); F(llama_model_params, lazy_mode);
    F(llama_model_params, main_gpu); F(llama_model_params, tensor_split);
    F(llama_model_params, progress_callback); F(llama_model_params, progress_callback_user_data);
    F(llama_model_params, kv_overrides); F(llama_model_params, vocab_only);
    F(llama_model_params, check_tensors); F(llama_model_params, use_extra_bufts);
    F(llama_model_params, no_host); F(llama_model_params, no_alloc); F(llama_model_params, load_mtp);
    END();
    BEGIN(llama_context_params);
    F(llama_context_params, n_ctx); F(llama_context_params, n_batch); F(llama_context_params, n_ubatch);
    F(llama_context_params, n_seq_max); F(llama_context_params, n_rs_seq);
    F(llama_context_params, n_outputs_max); F(llama_context_params, n_outputs_max_per_seq);
    F(llama_context_params, n_threads); F(llama_context_params, n_threads_batch);
    F(llama_context_params, ctx_type); F(llama_context_params, rope_scaling_type);
    F(llama_context_params, pooling_type); F(llama_context_params, attention_type);
    F(llama_context_params, flash_attn_type); F(llama_context_params, rope_freq_base);
    F(llama_context_params, rope_freq_scale); F(llama_context_params, yarn_ext_factor);
    F(llama_context_params, yarn_attn_factor); F(llama_context_params, yarn_beta_fast);
    F(llama_context_params, yarn_beta_slow); F(llama_context_params, yarn_orig_ctx);
    F(llama_context_params, defrag_thold); F(llama_context_params, cb_eval);
    F(llama_context_params, cb_eval_user_data); F(llama_context_params, type_k);
    F(llama_context_params, type_v); F(llama_context_params, abort_callback);
    F(llama_context_params, abort_callback_data); F(llama_context_params, embeddings);
    F(llama_context_params, offload_kqv); F(llama_context_params, no_perf);
    F(llama_context_params, op_offload); F(llama_context_params, swa_full);
    F(llama_context_params, kv_unified); F(llama_context_params, samplers);
    F(llama_context_params, n_samplers); F(llama_context_params, ctx_other);
    END();
    BEGIN(llama_batch);
    F(llama_batch, n_tokens); F(llama_batch, token); F(llama_batch, embd); F(llama_batch, pos);
    F(llama_batch, n_seq_id); F(llama_batch, seq_id); F(llama_batch, logits);
    END();
    printf("}\n");
    return 0;
}
