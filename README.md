# LAMDA

## Model Slicing
The input file is `input.json` and the output file is `output.json`.

```bash
python ./NAR-predictor/dp2.py input.json output.json

##LLM Fine-tuning
git clone https://github.com/QwenLM/Qwen.git
cd Qwen
bash ./finetune/finetune_qlora_single_gpu.sh


##LLM Model Generation
python generator.py


##Model decoder
python decode_V2.py


##Accuracy Testing Module
python python ./Training_module/trainer.py



