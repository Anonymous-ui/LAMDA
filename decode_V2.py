import torch
import torch.nn as nn
import torch_pruning as tp

def read_file(path):
    string = ''
    with open(path,'r') as f:
        lines = f.readlines()
        for line in lines:
            string += line
    return string


class CustomModel(nn.Module):
    def __init__(self,layers_info):
        super(CustomModel,self).__init__()
        self.layers = nn.ModuleList()
        self.layers_info = layers_info
        self.layers_id = {}
        self.nn_layers = ['Conv','Relu','MaxPool','AveragePool','Flatten','Gemm','GlobalAveragePool','Sigmoid','Relu6','Clip','SiLU']

        for layer_info in layers_info:
            op_type = layer_info['op_type']
            input_dim = layer_info['input_dim']
            output_dim = layer_info['output_dim']

            if op_type == 'Conv':
                if layer_info['BatchNorm'] == 1:
                    layer = nn.ModuleList([
                        nn.Conv2d(input_dim,output_dim,kernel_size=layer_info['kernel_shape'],
                                  padding=layer_info['pads'],stride=layer_info['strides']),
                        nn.BatchNorm2d(output_dim)])
                else: #TODO:BN
                    layer = nn.Conv2d(input_dim,output_dim,kernel_size=layer_info['kernel_shape'],
                                      padding=layer_info['pads'],stride=layer_info['strides'])
            elif op_type == 'Relu':
                layer = nn.ReLU()
            elif op_type == 'SiLU':
                layer = nn.SiLU()
            elif op_type == 'Relu6':
                layer = nn.ReLU6()
            elif op_type == 'Clip':
                layer = nn.ReLU6()
            elif op_type == 'MaxPool':
                layer = nn.MaxPool2d(kernel_size=layer_info['kernel_shape'],padding=layer_info['pads'],
                                     stride=layer_info['strides'])
            elif op_type == 'AveragePool':
                layer = nn.AvgPool2d(kernel_size=layer_info['kernel_shape'],stride=layer_info['strides'],
                                     count_include_pad=False,padding=0)
            elif op_type == 'Flatten':
                layer = nn.Flatten(start_dim=1)
            elif op_type == 'Gemm':
                layer = nn.Linear(input_dim,output_dim)
            elif op_type == 'ReduceMean':
                layer = None
            elif op_type == 'GlobalAveragePool':
                layer = nn.AdaptiveAvgPool2d(1)
           #elif op_type == 'Clip':
           #    layer = None
            elif op_type == 'Add':
                layer = None
            elif op_type == 'Concat':
                layer = None
            elif op_type == 'Sigmoid':
                layer = nn.Sigmoid()
            elif op_type == 'Mul':
                layer = None
            else:
                raise ValueError(f"Unsupported op_type: {op_type}")
            self.layers.append(layer)

    def forward(self,x):
        Index = 0
        self.layers_id.update({'data': x})

        for layer in self.layers:
            if layers_info[Index]['op_type'] in self.nn_layers:
                input_id = self.layers_info[Index]['input_id']
                output_id = self.layers_info[Index]['output_id']
                if isinstance(layer,type(self.layers)):
                    x = layer[0](self.layers_id[input_id])
                    x = layer[1](x)
                else:
                    x = layer(self.layers_id[input_id])
                self.layers_id.update({f'{output_id}': x})
                Index += 1

            elif layer == None:
                if layers_info[Index]['op_type'] == 'Add': #Add
                    input_ids = self.layers_info[Index]['input_id'].split(' and ')
                    output_id = self.layers_info[Index]['output_id']
                    x = self.layers_id[input_ids[0]] + self.layers_id[input_ids[1]]
                    self.layers_id.update({f'{output_id}': x})

                #elif layers_info[Index]['op_type'] == 'Clip': #Clip
                #    input_id = self.layers_info[Index]['input_id']
                #    output_id = self.layers_info[Index]['output_id']
                #    self.layers_id.update({f'{output_id}': x})

                elif layers_info[Index]['op_type'] == 'ReduceMean': #Reduce Mean
                    input_id = self.layers_info[Index]['input_id']
                    output_id = self.layers_info[Index]['output_id']
                    x = torch.mean(self.layers_id[input_id],dim=layers_info[Index]['axes'],
                                   keepdim=bool(layers_info[Index]['keepdims']))
                    self.layers_id.update({f'{output_id}': x})

                elif layers_info[Index]['op_type'] == 'Concat': #Concat
                    input_ids = self.layers_info[Index]['input_id'].split(' and ')
                    output_id = self.layers_info[Index]['output_id']
                    x = torch.cat([self.layers_id[id] for id in input_ids],dim=layers_info[Index]['axis'])
                    self.layers_id.update({f'{output_id}': x})



                elif layers_info[Index]['op_type'] == 'Mul':

                    input_ids = self.layers_info[Index]['input_id'].split(' and ')
                    output_id = self.layers_info[Index]['output_id']
                    print(f"Tensor A shape: {self.layers_id[input_ids[0]].shape}")
                    print(f"Tensor B shape: {self.layers_id[input_ids[1]].shape}")
                    self.layers_id[input_ids[1]] = self.layers_id[input_ids[1]].unsqueeze(-1).unsqueeze(-1)
                    x = torch.mul(self.layers_id[input_ids[0]],self.layers_id[input_ids[1]])
                    self.layers_id.update({f'{output_id}': x})

                Index += 1

        return x


class decoder():
    def decode(input):
        layers = input.split('\n')
        layers_info = []

        for layer in layers:
            layer_info = {}

            if '#' in layer:
                layer = layer.split('#')
                Type_inout = layer[0].split(',')
                op_type = Type_inout[0].split(':')[1]


                input_dim = Type_inout[1].split(':')[1]
                input_id = Type_inout[2].split(';')[0].split(':')[1]

                output_dim = Type_inout[2].split(';')[1].split(':')[1]
                output_id = Type_inout[3].split(':')[1]

                if input_dim == 'data':
                    input_dim = 3
                if output_dim == 'output1':
                    output_dim = input_dim

                layer_info.update({'op_type': op_type})
                layer_info.update({'input_dim': int(input_dim)})
                layer_info.update({'output_dim': int(output_dim)})
                layer_info.update({'input_id': input_id})
                layer_info.update({'output_id': output_id})

                Attributes = layer[1].split(';')
                for Attribute in Attributes:
                    Key_V = Attribute.split(':')
                    Key = Key_V[0]
                    Value = Key_V[1]
                    try: #int
                        layer_info.update({f'{Key}': int(Value)})
                    except Exception:
                        try: #Float
                            layer_info.update({f'{Key}': float(Value)})
                        except Exception:
                            dims = Value.strip('[]').split(',')
                            layer_info.update({f'{Key}': tuple([int(x) for x in dims])})
                layers_info.append(layer_info)

            else: #
                Type_inout = layer.split(',')
                op_type = Type_inout[0].split(':')[1]

                if op_type == 'Add':
                    Type_inout = layer.split(',')
                    op_type = Type_inout[0].split(':')[1]
                    input_dim = Type_inout[1].split(':')[1]

                    input_id = Type_inout[2].split(';')[0].split(':')[1]

                    output_dim = Type_inout[2].split(';')[1].split(':')[1]
                    output_id = Type_inout[3].split(':')[1]

                    if input_dim == 'data':
                        input_dim = 3
                    if output_dim == 'output1':
                        output_dim = input_dim
                    layer_info.update({'op_type': op_type})
                    layer_info.update({'input_dim': int(input_dim)})
                    layer_info.update({'output_dim': int(output_dim)})
                    layer_info.update({'input_id': input_id})
                    layer_info.update({'output_id': output_id})
                    layers_info.append(layer_info)

                else:
                    input_dim = Type_inout[1].split(':')[1]

                    input_id = Type_inout[2].split(';')[0].split(':')[1]
                    output_dim = Type_inout[2].split(';')[1].split(':')[1]
                    output_id = Type_inout[3].split(':')[1]

                    if input_dim == 'data':
                        input_dim = 3
                    if output_dim == 'output1':
                        output_dim = input_dim
                    layer_info.update({'op_type': op_type})
                    layer_info.update({'input_dim': int(input_dim)})
                    layer_info.update({'output_dim': int(output_dim)})
                    layer_info.update({'input_id': input_id})
                    layer_info.update({'output_id': output_id})
                    layers_info.append(layer_info)
        return layers_info


if __name__ == '__main__':
    input = '''Op_type:Conv,input_dim:data,input_id:data;output_dim:43,output_id:192#dilations:1;group:1;kernel_shape:3;pads:1;strides:2;BatchNorm:1
Op_type:Relu,input_dim:43,input_id:192;output_dim:43,output_id:125
Op_type:MaxPool,input_dim:43,input_id:125;output_dim:43,output_id:126#kernel_shape:3;pads:1;strides:2
Op_type:Conv,input_dim:43,input_id:126;output_dim:109,output_id:195#dilations:1;group:1;kernel_shape:3;pads:1;strides:1;BatchNorm:1
Op_type:Relu,input_dim:109,input_id:195;output_dim:109,output_id:129
Op_type:Conv,input_dim:109,input_id:129;output_dim:43,output_id:198#dilations:1;group:1;kernel_shape:3;pads:1;strides:1;BatchNorm:1
Op_type:Add,input_dim:43,input_id:198 and 126;output_dim:43,output_id:132
Op_type:Relu,input_dim:43,input_id:132;output_dim:43,output_id:133
Op_type:Conv,input_dim:43,input_id:133;output_dim:49,output_id:201#dilations:1;group:1;kernel_shape:3;pads:1;strides:1;BatchNorm:1
Op_type:Relu,input_dim:49,input_id:201;output_dim:49,output_id:136
Op_type:Conv,input_dim:49,input_id:136;output_dim:43,output_id:204#dilations:1;group:1;kernel_shape:1;pads:0;strides:1;BatchNorm:1
Op_type:Add,input_dim:43,input_id:204 and 133;output_dim:43,output_id:139
Op_type:Relu,input_dim:43,input_id:139;output_dim:43,output_id:140
Op_type:Conv,input_dim:43,input_id:140;output_dim:191,output_id:207#dilations:1;group:1;kernel_shape:3;pads:1;strides:2;BatchNorm:1
Op_type:Relu,input_dim:191,input_id:207;output_dim:191,output_id:143
Op_type:Conv,input_dim:191,input_id:143;output_dim:390,output_id:210#dilations:1;group:1;kernel_shape:3;pads:1;strides:1;BatchNorm:1
Op_type:Conv,input_dim:43,input_id:140;output_dim:390,output_id:213#dilations:1;group:1;kernel_shape:3;pads:1;strides:2;BatchNorm:1
Op_type:Add,input_dim:390,input_id:210 and 213;output_dim:390,output_id:148
Op_type:Relu,input_dim:390,input_id:148;output_dim:390,output_id:149
Op_type:Conv,input_dim:390,input_id:149;output_dim:149,output_id:216#dilations:1;group:1;kernel_shape:1;pads:0;strides:1;BatchNorm:1
Op_type:Relu,input_dim:149,input_id:216;output_dim:149,output_id:152
Op_type:Conv,input_dim:149,input_id:152;output_dim:390,output_id:219#dilations:1;group:1;kernel_shape:3;pads:1;strides:1;BatchNorm:1
Op_type:Add,input_dim:390,input_id:219 and 149;output_dim:390,output_id:155
Op_type:Relu,input_dim:390,input_id:155;output_dim:390,output_id:156
Op_type:Conv,input_dim:390,input_id:156;output_dim:558,output_id:222#dilations:1;group:1;kernel_shape:3;pads:1;strides:1;BatchNorm:1
Op_type:Relu,input_dim:558,input_id:222;output_dim:558,output_id:159
Op_type:Conv,input_dim:558,input_id:159;output_dim:390,output_id:225#dilations:1;group:1;kernel_shape:3;pads:1;strides:1;BatchNorm:1
Op_type:Add,input_dim:390,input_id:225 and 156;output_dim:390,output_id:164
Op_type:Relu,input_dim:390,input_id:164;output_dim:390,output_id:165
Op_type:GlobalAveragePool,input_dim:390,input_id:165;output_dim:390,output_id:166
Op_type:Flatten,input_dim:390,input_id:166;output_dim:390,output_id:167
Op_type:Gemm,input_dim:390,input_id:167;output_dim:390,output_id:output1#alpha:1.0;beta:1.0;transB:1'''
    Decoder = decoder
    layers_info = decoder.decode(input)
    model = CustomModel(layers_info)
    input = torch.ones(1,3,224,224)
    model.eval()
    torch.save(model,           '')
    print(f"Model saved to ")
    print(model)
    dummy_input = torch.randn(1,3,224,224)
    onnx_path = ''
    torch.onnx.export(model,dummy_input,onnx_path,verbose=True)
    print(f"Model saved to '{onnx_path}'")



