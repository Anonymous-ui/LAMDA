import os
import time
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM


model_dir = 'Qwen-main/14Bsq3-2000'
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoPeftModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    trust_remote_code=True
).eval()


responses = {}


latency_values = [0.5648, 0.5564, 0.7851, 1.0812, 1.0751, 1.0614, 1.0543, 1.0451, 1.4663, 1.6753, 2.7849, 2.8821, 3.8741]
model_names = ["squeezenet"]


num_parts = 5  


output_dir = 'data8'
os.makedirs(output_dir, exist_ok=True)


for latency in latency_values:
    for model_name in model_names:
        history = []
        print(f"Processing for model: {model_name}, latency: {latency}s...")
        
        output_file_path = os.path.join(output_dir, f"{model_name}.txt")
        
        for part in range(num_parts):
          
            if part == 0:
               
                input_text = f"Can you generate a PyTorch {model_name} model whose latency is {latency}s? Please ensure the input and output dimensions conform to the rules of the neural network (part {part + 1})."
            else:
                input_text = f"Can you generate a PyTorch {model_name} model whose latency is {latency}s? Please ensure the input and output dimensions conform to the rules of the neural network (part {part + 1})."
            
       
            start_time = time.time()
         
           
            response, history = model.chat(tokenizer, input_text, history=history)
            
     
            end_time = time.time()
            response_time = end_time - start_time
            
         
            responses[input_text] = (response, response_time)
          
            with open(output_file_path, 'a', encoding='utf-8') as file:
                file.write(f"Input: {input_text}\nResponse: {response}\nResponse Time: {response_time:.4f} seconds\n\n")
            
            previous_input = input_text  
            history = []  
