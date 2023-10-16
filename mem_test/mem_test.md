## How to the peak memory usage
```bash
export CUDA_VISIBLE_DEVICES=0
python mlora.py --base_model=<llama-7b-model-path> --device "cuda:0" --config <config.file> --load_8bit
```
batch size 4
```bash
export CUDA_VISIBLE_DEVICES=0
# in alpaca-lora repo
python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=4 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=1e-4 --group_by_length
# ... <- multi process to run multi model
```
batch size 6
```bash
export CUDA_VISIBLE_DEVICES=0
python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=60 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=1e-4 --group_by_length
```
batch size 8
```bash
export CUDA_VISIBLE_DEVICES=0
python finetune.py --base_model= --data_path="data/data_set_3.json" --batch_size=64 --micro_batch_size=4 --num_epochs=8 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=1e-4 --group_by_length
```