## How to test
different methods can trigger the early stop in different step, for equality, we use the same step to stop the process.

the baseline code need to add patch to simulation the early stop.
- step1. add below code to `transformers/trainer.py:1767`
```python
early_stop_step_cnt = 0
```
- step2. add below code to `transformers/trainer.py:1911`
```python
early_stop_step_cnt = early_stop_step_cnt + 1
stop_step_from_os = int(os.environ["EA_STEP"])
if early_stop_step_cnt == stop_step_from_os:
    exit(0)
```

to test aspen, you shoud use new branch `ea_performance_analyze`, change the branch:
`git checkout ea_performance_analyze`
```python
cnt = (self.epoch_cnt_ - 1) * len(self.train_token_data_) + \
    self.next_train_data_start_idx_
if self.adapter_name_ == "lora_0" and cnt >= 250 * 60:
    return True
if self.adapter_name_ == "lora_1" and cnt >= 280 * 60:
    return True
if self.adapter_name_ == "lora_2" and cnt >= 300 * 60:
    return True
if self.adapter_name_ == "lora_3" and cnt >= 150 * 60:
    return True
```

the test just same as latency_test, you should found the test command in `latency_test/latency_test.md`