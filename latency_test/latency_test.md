## How to test the model train latency
### test llama-7b
* test aspen
```bash
export CUDA_VISIBLE_DEVICES=0
python mlora.py --base_model=<llama-7b-model-path> --device "cuda:0" --config latency_test/latency_test.json --load_8bit
```
* test alpaca-lora@seq
```bash
export CUDA_VISIBLE_DEVICES=0
python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=1e-4 --group_by_length && python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=5e-4 --group_by_length && python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=1e-3 --group_by_length && python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=5e-3 --group_by_length
```
* test alpaca-lora@sync
```bash
# run the 2 process simultaneously
python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=1e-4 --group_by_length && python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=5e-4 --group_by_length
# run below command simultaneously and in same gpu
python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=1e-3 --group_by_length && python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=5e-3 --group_by_length
```

### test llama2-7b
* test aspen
```bash
export CUDA_VISIBLE_DEVICES=0
python mlora.py --base_model=<llama2-7b-model-path> --device "cuda:0" --config latency_test/latency_test_llama2.json --load_8bit
```
* test alpaca-lora@seq
```bash
export CUDA_VISIBLE_DEVICES=0
python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=1e-4 --group_by_length && python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=5e-4 --group_by_length && python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=1e-3 --group_by_length && python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=5e-3 --group_by_length
```
* test alpaca-lora@sync
```bash
# run the 2 process simultaneously
python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=1e-4 --group_by_length && python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=5e-4 --group_by_length
# run below command simultaneously and in same gpu
python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=1e-3 --group_by_length && python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=5e-3 --group_by_length
```

### test chatglm2-6b
* test aspen
```bash
export CUDA_VISIBLE_DEVICES=0
python mlora.py --base_model=<chatglm-model> --device "cuda:0" --config latency_test_chatglm.json --model_type "chatglm" --load_8bit
```
* test aspen@seq
```bash
export CUDA_VISIBLE_DEVICES=0
python finetune_chatglm.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=4 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["dense","dense_4h_to_h","dense_h_to_4h","query_key_value"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=1e-4 --group_by_length && python finetune_chatglm.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=4 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["dense","dense_4h_to_h","dense_h_to_4h","query_key_value"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=5e-4 --group_by_length && python finetune_chatglm.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=4 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["dense","dense_4h_to_h","dense_h_to_4h","query_key_value"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=1e-3 --group_by_length && python finetune_chatglm.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=4 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["dense","dense_4h_to_h","dense_h_to_4h","query_key_value"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=5e-3 --group_by_length
```
* test aspen@sync
```bash
# run the 2 process simultaneously
export CUDA_VISIBLE_DEVICES=0
python finetune_chatglm.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=4 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["dense","dense_4h_to_h","dense_h_to_4h","query_key_value"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=1e-4 --group_by_length && python finetune_chatglm.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=4 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["dense","dense_4h_to_h","dense_h_to_4h","query_key_value"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=5e-4 --group_by_length
# run below command simultaneously and in same gpu
python finetune_chatglm.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=4 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["dense","dense_4h_to_h","dense_h_to_4h","query_key_value"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=1e-3 --group_by_length && python finetune_chatglm.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=4 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["dense","dense_4h_to_h","dense_h_to_4h","query_key_value"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=5e-3 --group_by_length
```

### test qlora
* test aspen
```bash
export CUDA_VISIBLE_DEVICES=0
python mlora.py --base_model=<llama-7b-model-path> --device "cuda:0" --config latency_test/latency_test.json --load_4bit
```
* test alpaca-lora@seq
```bash
export CUDA_VISIBLE_DEVICES=0
python finetune_qlora.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=1e-4 --group_by_length && python finetune_qlora.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=5e-4 --group_by_length && python finetune_qlora.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=1e-3 --group_by_length && python finetune_qlora.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=5e-3 --group_by_length
```
* test alpaca-lora@sync
```bash
# run the 2 process simultaneously
python finetune_qlora.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=1e-4 --group_by_length && python finetune_qlora.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=5e-4 --group_by_length
# run below command simultaneously and in same gpu
python finetune_qlora.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=1e-3 --group_by_length && python finetune_qlora.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=1024 --prompt_template_name=sql --learning_rate=5e-3 --group_by_length
```