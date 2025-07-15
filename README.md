# LAMDA

模型切割模块：
其中input.json为输入  output.json为输出
python ./NAR-predictor/dp2.py input.json output.json 

LLM微调：
git clone https://github.com/QwenLM/Qwen.git
cd Qwen
bash ./finetune/finetune_qlora_single_gpu.sh 

LLM模型生成：
python generator.py

模型decoder：
python decode_V2.py

精度测试模块：
python python ./Training_module/trainer.py



