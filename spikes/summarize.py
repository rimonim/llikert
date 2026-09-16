import json, sys
r = json.load(open(sys.argv[1]))
for k in ['gguf_sha256','hash_seconds','load_seconds','n_vocab','n_ctx_train','add_bos','general.name','tokenizer.ggml.pre','template_sha256','template_matches_official','vectorized_vs_scalar_max_abs','repeat_same_ctx_max_abs_logit','after_other_prompt_max_abs_logit','logits_ptr_after_next_decode','batch_size_vs_512','context_limit','tail_segment']:
    print(k, '=>', json.dumps(r[k]))
print('render_checks'); [print('  ', k, v) for k, v in r['render_checks'].items()]
print('tail tokens', [(t['id'], t['piece'], t['attr'], t['control']) for t in r['tail_tokens']])
print('boundary'); [print('  ', repr(k), v['ok'], v.get('suffix'), (v.get('token') or v.get('first_token') or {}).get('piece')) for k, v in r['boundary'].items()]
for name, t in r['results'].items():
    print(name, t['token_ids'])
    for tid, it in t['items'].items():
        print('  ', tid, [round(x, 3) for x in it['probabilities']], 'cov=%.3g' % it['coverage'], 'logcov=%.3f' % it['log_coverage'], 'EV=', it.get('expected_value') and round(it['expected_value'], 3), 'argmax=', repr(it['argmax_vocab']['piece']), it['n_prompt_tokens'])
