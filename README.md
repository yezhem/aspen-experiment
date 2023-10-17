## How to test

- step1. install aspen from [aspen](https://github.com/TUDB-Labs/multi-lora-fine-tune) and install alpaca-lora from [alpaca-lora](https://github.com/tloen/alpaca-lora)
- step2. change aspen branch to `main_performance_analyze` for performance analyze
- step3. install extra requirements by `pip install -r requirements.txt`
- step4. use patch to add code to alpaca-lora for performance analyze, use this patch, you need to find transformers and peft lib in your python env. 
    * `patch transformers/trainer.py < patch/trainer.patch`
    * `patch transformers/models/llama/modeling_llama.py < patch/modeling_llama.patch`
    * `patch peft/tuners/lora.py < patch/lora.patch`
- step5. add patch to alpaca-lora.
    * `git apply patch/finetune.patch`
    * `cp -r alpaca_template templates`
- step7. run alpaca-lora and aspen will produce the `logs.log` file, those fill will record the performance data, use script to analyze.
    * `python analyze.py logs.log`