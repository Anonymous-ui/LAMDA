# LAMDA
## Environmental requirements
sudo apt-get install build-essential linux-generic libmpich-dev libopenmpi-dev
conda create -n Qwen python=3.10
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirement.txt
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.32.0 accelerate tiktoken einops scipy transformers_stream_generator==0.0.4 peft deepspeed mpi4py
pip install auto-gptq optimum
conda install mpi4py
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
sudo apt-get install protobuf-compiler libprotoc-dev cmake
pip install scipy torch_geometric torch_scatter onnx==1.8.1 networkx
pip install protobuf==3.20.0 pyproject numpy==1.20.0

## Model Slicing
The input file is `input.json` and the output file is `output.json`.

```bash
python ./NAR-predictor/dp2.py input.json output.json
```

## LLM Fine-tuning Dataset
```bash
unzip ./train001.zip
```

## LLM Fine-tuning
```bash
git clone https://github.com/QwenLM/Qwen.git
cd Qwen
bash ./finetune/finetune_qlora_single_gpu.sh
```

## LLM Model Generation
```bash
python generator.py
```

## Model decoder
```bash
python decode_V2.py
```

## Latency Predictor

### Data preparation
Download the `unseen_structure` part of [NNLQP](https://github.com/ModelTC/NNLQP) and put it in `dataset/`. Download the `dataset/unseen_structure/gt_stage.txt` and put it in `dataset/unseen_structure/`.

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

## Accuracy Testing Module
```bash
python python ./Training_module/trainer.py
```



