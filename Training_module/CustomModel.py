import torch
import torch.nn as nn

def read_file(path):
  string = ''
  with open(path,'r') as f:
    lines =  f.readlines()
    for line in lines:
      string += line
  return string

class CustomModel(nn.Module):
    def __init__(self, layers_info):
        super(CustomModel, self).__init__()
        self.layers = nn.ModuleList()
        self.layers_info = layers_info
        self.layers_id = {}
        self.nn_layers = ['Conv','Relu','MaxPool','AveragePool','Flatten','Gemm','GlobalAveragePool','Sigmoid']
        
        for layer_info in layers_info:
            op_type = layer_info['op_type']
            input_dim = layer_info['input_dim']
            output_dim = layer_info['output_dim']

            if op_type == 'Conv':
              if layer_info['BatchNorm'] == 1: #BN层
                layer = nn.ModuleList([
                        nn.Conv2d(input_dim, output_dim, kernel_size=layer_info['kernel_shape'],
                                  padding=layer_info['pads'], stride=layer_info['strides']),
                        nn.BatchNorm2d(output_dim)])
              else: #TODO:BN
                layer = nn.Conv2d(input_dim, output_dim, kernel_size=layer_info['kernel_shape'],
                                  padding=layer_info['pads'], stride=layer_info['strides'])
            elif op_type == 'Relu':
                layer = nn.ReLU()
            elif op_type == 'MaxPool':
                layer = nn.MaxPool2d(kernel_size=layer_info['kernel_shape'], padding=layer_info['pads'], stride=layer_info['strides'])
            elif op_type == 'AveragePool':
                layer = nn.AvgPool2d(kernel_size=layer_info['kernel_shape'], stride=layer_info['strides'],count_include_pad=False, padding=0)
            elif op_type == 'Flatten':
                layer = nn.Flatten(start_dim=1)
            elif op_type == 'Gemm':
                layer = nn.Linear(input_dim, output_dim)
            elif op_type == 'ReduceMean':
                layer = None
            elif op_type == 'GlobalAveragePool':
                layer = nn.AdaptiveAvgPool2d(1)
            elif op_type == 'Clip':
                layer = None
            elif op_type == 'Add':
                layer = None
            elif op_type == 'Concat':
                layer = None
            elif op_type == 'Sigmoid':
                layer == nn.Sigmoid()
            elif op_type == 'Mul':
                layer == None
            else:
                raise ValueError(f"Unsupported op_type: {op_type}")
            self.layers.append(layer)
    
    def forward(self, x):
        Index = 0
        self.layers_id.update({'data': x})
        
        for layer in self.layers:
          if self.layers_info[Index]['op_type'] in self.nn_layers:
            input_id = self.layers_info[Index]['input_id']
            output_id = self.layers_info[Index]['output_id']
            if isinstance(layer,type(self.layers)):
              x = layer[0](self.layers_id[input_id])
              x = layer[1](x)
            else:
              x = layer(self.layers_id[input_id])
            self.layers_id.update({f'{output_id}': x})
            Index +=1
            
          elif layer == None:
            if self.layers_info[Index]['op_type'] == 'Add': #Add
              input_ids = self.layers_info[Index]['input_id'].split(' and ')
              output_id = self.layers_info[Index]['output_id']
              x = self.layers_id[input_ids[0]] + self.layers_id[input_ids[1]]
              self.layers_id.update({f'{output_id}': x})
              
            elif self.layers_info[Index]['op_type'] == 'Clip': #Clip
              input_id = self.layers_info[Index]['input_id']
              output_id = self.layers_info[Index]['output_id']
              x = torch.clip(self.layers_id[input_id], min=self.layers_info[Index]['min'],max=self.layers_info[Index]['max'])
              self.layers_id.update({f'{output_id}': x})
              
            elif self.layers_info[Index]['op_type'] == 'ReduceMean': #Reduce Mean
              input_id = self.layers_info[Index]['input_id']
              output_id = self.layers_info[Index]['output_id']
              x = torch.mean(self.layers_id[input_id],dim=self.layers_info[Index]['axes'],keepdim=bool(self.layers_info[Index]['keepdims']))
              self.layers_id.update({f'{output_id}': x})
              
            elif self.layers_info[Index]['op_type'] == 'Concat': #Concat 
              input_ids = self.layers_info[Index]['input_id'].split(' and ')
              output_id = self.layers_info[Index]['output_id']
              
              x = torch.cat([self.layers_id[id] for id in input_ids],dim = self.layers_info[Index]['axis'])
              self.layers_id.update({f'{output_id}': x})
              
            elif self.layers_info[Index]['op_type'] == 'Mul': #Mul
              input_ids = self.layers_info[Index]['input_id'].split(' and ')
              output_id = self.layers_info[Index]['output_id']
              x = torch.mul([self.layers_id[input_ids[0]], self.layers_id[input_ids[1]]], dim = self.layers_info[Index]['axis'])
              self.layers_id.update({f'{output_id}': x})
            Index += 1
        return x
      
      
class ModelWarpper(nn.Module):
  def __init__(self, model, config):
    super(ModelWarpper,self).__init__()
    self.model = model
    
    output = self.model(torch.ones(config.batch_size,3,config.resolution,config.resolution))
    self.linear = nn.Linear(output.shape[1],config.class_number)
    
  def forward(self,x):
    x = self.model(x)
    x = self.linear(x)
    return x
    