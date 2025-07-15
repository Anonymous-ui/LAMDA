# LAMDA

## Data preparation
Download the `unseen_structure` part of [NNLQP](https://github.com/ModelTC/NNLQP) and put it in `dataset/`. Download the `dataset/unseen_structure/gt_stage.txt` and put it in `dataset/unseen_structure/`.

# Model Slicing
The input file is `input.json` and the output file is `output.json`.

```bash
python ./NAR-predictor/dp2.py input.json output.json
```

# LLM Fine-tuning
```bash
git clone https://github.com/QwenLM/Qwen.git
cd Qwen
bash ./finetune/finetune_qlora_single_gpu.sh
```

# LLM Model Generation
```bash
python generator.py
```

# Model decoder
```bash
python decode_V2.py
```

# Latency Predictor

### Train NAR-Former V2
You can directly download the `experiments/latency_prediction/in_domain/checkpoints/ckpt_best.pth` or train from scratch following the steps below:
Change the `BASE_DIR` in `experiments/latency_prediction/in_domain/train.sh` to the absolute path of our codes and run:

```
cd experiments/latency_prediction/in_domain/
bash train.sh
```

### Use NAR-Former V2 as latency predictor
```
cd experiments/latency_prediction/in_domain/
bash test.sh
```

# Accuracy Testing Module
```bash
python python ./Training_module/trainer.py
```



