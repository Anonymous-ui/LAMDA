#LAMDA

#Model Slicing
The input file is `input.json` and the output file is `output.json`.
```bash
python ./NAR-predictor/dp2.py input.json output.json
```bash

LLM Fine-tuning：
```bash
git clone https://github.com/QwenLM/Qwen.git
cd Qwen
bash ./finetune/finetune_qlora_single_gpu.sh 
```bash


LLM Model Generation：
```bash
python generator.py
```bash

Model decoder：
```bash
python decode_V2.py
```bash

Accuracy Testing Module：
```bash
python python ./Training_module/trainer.py
```bash


