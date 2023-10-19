## How to test
run aspen with this config in different gpu.
* aspen
```bash
export CUDA_VISIBLE_DEVICES=0
python mlora.py --base_model=<llama-7b-model-path> --device "cuda:0" --config ablation/ablation.json --load_8bit
```
* test alpaca-lora@seq
```bash
export CUDA_VISIBLE_DEVICES=0
python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=4 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=1e-4 --group_by_length && python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=4 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=5e-4 --group_by_length && python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=4 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=1e-3 --group_by_length && python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=4 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=5e-3 --group_by_length
```
* test alpaca-lora@sync
```bash
# run the 2 process simultaneously
python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=4 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=1e-4 --group_by_length && python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=4 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=5e-4 --group_by_length
python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=4 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=1e-3 --group_by_length && python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=4 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=5e-3 --group_by_length
```