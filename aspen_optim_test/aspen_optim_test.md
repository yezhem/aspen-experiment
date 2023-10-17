## How to test
* aspen
```bash
export CUDA_VISIBLE_DEVICES=0
python mlora.py --base_model=<llama-7b-model-path> --device "cuda:0" --config aspen_optim_test/aspen_m1.json --load_8bit
python mlora.py --base_model=<llama-7b-model-path> --device "cuda:0" --config aspen_optim_test/aspen_m2.json --load_8bit
python mlora.py --base_model=<llama-7b-model-path> --device "cuda:0" --config aspen_optim_test/aspen_m3.json --load_8bit
```
* baseline
```bash
export CUDA_VISIBLE_DEVICES=0
python finetune.py --base_model=<llama-7b-model-path> --data_path="data/data_set_1.json" --batch_size=48 --micro_batch_size=6 --num_epochs=1 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=3e-4 --group_by_length && python finetune.py --base_model=<llama-7b-model-path> --data_path="data/data_set_1.json" --batch_size=48 --micro_batch_size=6 --num_epochs=1 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=3e-3 --group_by_length && python finetune.py --base_model=<llama-7b-model-path> --data_path="data/data_set_2.json" --batch_size=48 --micro_batch_size=6 --num_epochs=2 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=3e-4 --group_by_length && python finetune.py --base_model=<llama-7b-model-path> --data_path="data/data_set_2.json" --batch_size=48 --micro_batch_size=6 --num_epochs=2 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=3e-3 --group_by_length && python finetune.py --base_model=<llama-7b-model-path> --data_path="data/data_set_3.json" --batch_size=48 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=3e-4 --group_by_length && python finetune.py --base_model=<llama-7b-model-path> --data_path="data/data_set_3.json" --batch_size=48 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=3e-3 --group_by_length &&  python finetune.py --base_model=<llama-7b-model-path> --data_path="data/data_set_4.json" --batch_size=48 --micro_batch_size=6 --num_epochs=8 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=3e-4 --group_by_length &&  python finetune.py --base_model=<llama-7b-model-path> --data_path="data/data_set_4.json" --batch_size=48 --micro_batch_size=6 --num_epochs=4 --val_set_size=-1 --lora_target_modules=["q_proj","k_proj","v_proj","o_proj"] --cutoff_len=512 --prompt_template_name=sql --learning_rate=3e-3 --group_by_length
```