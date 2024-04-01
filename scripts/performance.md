# Performance report

## The sql to query the nsys

```sql
CREATE TEMPORARY TABLE IF NOT EXISTS TEMP_KERN_INFOS AS
SELECT R.start AS API_START, R.end AS API_END,
       K.start AS KERN_START, K.end AS KERN_END,
       R.start AS T_START,
       MAX(R.end, K.end) AS T_END,
       KNAME.value AS KERN_NAME
    FROM 
        CUPTI_ACTIVITY_KIND_KERNEL AS K 
    JOIN
        CUPTI_ACTIVITY_KIND_RUNTIME AS R 
        ON K.correlationId == R.correlationId
    LEFT JOIN
        StringIds AS KNAME
        ON KNAME.id == K.demangledName;

CREATE INDEX IF NOT EXISTS TEMPINDEX ON TEMP_KERN_INFOS(T_START);

SELECT
    E_NAME AS ENVET_NAME,
    API_START AS KERN_API_START,
    API_END AS KERN_API_END,
    KERN_START AS KERN_START,
    KERN_END AS KERN_END,
    KERN_NAME AS KERN_NAME
FROM
    (SELECT start AS E_START, end AS E_END, text AS E_NAME FROM NVTX_EVENTS)
LEFT JOIN 
    TEMP_KERN_INFOS
ON
    E_START <= T_START AND E_END >= T_START;
```

## The performace report about mlora's nvtx range

### mlora metirc group

1. f_embedding
2. f_attention_norm
3. f_linear = f_linear_wq + f_linear_wk +_ f_linear_wv + f_linear_wo
4. f_lora_{adapter} = f_dropout_{adapter} + f_lora_a_{adapter} + f_lora_b_{adapter} + f_scaling_{adapter}
5. f_lora_add_{adapter}
6. f_rotray_emb
7. f_attention
8. f_o_add
9. f_ffn_norm
10. f_mlp = f_w1 + f_w3 + f_silu_w2 + f_w1_m_w3 + f_w2 + f_w2_add
11. f_mlp_add = f_w2_add
12. f_rmsnorm
13. f_output
14. f_calc_loss
15. b_loss
16. b_output
17. b_rmsnorm
18. b_mlp = b_linear_w2 + b_w1_m_w3 + b_silu_w2 + b_linear_w3 + b_linear_w1
19. b_ffn_norm
20. b_lora_{adapter} = b_scaling_{adapter} + b_lora_b_{adapter} + b_lora_a_{adapter} + b_dropout_{adapter} + b_lora_split_{adapter}
21. b_lora_add_{adapter}
22. b_linear
23. b_attention
24. b_k_rope
25. b_q_rope
26. b_attention_norm

### mlora timeline

1. f_embedding
2. f_attention_norm_{layer_id}
3. f_q_{layer_id}
4. f_linear_{name}
5. f_dropout_{adapter}
6. f_lora_a_{adapter}
7. f_lora_b_{adapter}
8. f_scaling_{adapter}
9. f_lora_add_({adapter})
10. f_k_{layer_id}
11. f_v_{layer_id}
12. f_rotray_emb_{layer_id}
13. f_attention_{layer_id}
14. f_o_{layer_id}
15. f_o_add_{layer_id}
16. f_ffn_norm_{layer_id}
17. f_w1_{layer_id}
18. f_w3_{layer_id}
19. f_silu_w2_{layer_id}
20. f_w1_m_w3_{layer_id}
21. f_w2_{layer_id}
22. f_w2_add
23. f_rmsnorm
24. f_output
25. f_calc_loss
26. b_loss
27. b_output
28. b_rmsnorm
29. b_checkpoint
30. b_linear_w2_
31. b_w1_m_w3_{layer_id}
32. b_silu_w2_{layer_id}
33. b_linear_w3_
34. b_linear_w1_
35. b_ffn_norm_{layer_id}
36. b_lora_add_{adapter}
37. b_scaling_{adapter}
38. b_lora_b_{adapter}
39. b_lora_a_{adapter}
40. b_dropout_{adapter}
41. b_lora_split_{adapter}
42. b_linear_wo_
43. b_attention_{layer_id}
44. b_k_rope_{layer_id}
45. b_q_rope_{layer_id}
46. b_linear_wv_
47. b_linear_wk_
48. b_linear_wq_
49. b_attention_norm_{layer_id}

## The performace report about peft's nvtx range

### peft metric group

1. f_embedding
2. f_attention_norm
3. f_linear
4. f_lora_{adapter}
5. f_lora_add_{adapter}
6. f_get_cos_sin
7. f_rotray_emb
8. f_attention
9. f_o_add
10. f_ffn_norm
11. f_mlp
12. f_mlp_add
13. f_rmsnorm
14. f_output
15. f_calc_loss
16. b_loss
17. b_output
18. b_rmsnorm
19. b_mlp
20. b_ffn_norm
21. b_lora_{adapter}
22. b_lora_add_{adpter}
23. b_linear
24. b_attention
25. b_k_rope
26. b_q_rope
27. b_attention_norm

### peft timeline

1. f_embedding
2. f_attention_norm
3. f_q (f_linear + f_lora_{adapter} + f_lora_add_{adapter})
4. f_k
5. f_v
6. f_get_cos_sin
7. f_rotray_emb
8. f_attention
9. f_o
10. f_o_add
11. f_ffn_norm
12. f_mlp
13. f_mlp_add
14. f_rmsnorm
15. f_output
16. f_calc_loss
17. b_loss
18. b_output
19. b_rmsnorm
20. b_checkpoint <- same with transformer block forward
21. b_mlp
22. b_ffn_norm
23. b_lora_{adapter}
24. b_linear
25. b_k_rope
26. b_q_rope
27. b_attention_norm
