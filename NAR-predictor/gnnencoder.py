import os
import torch
import json
from onnx.shape_inference import infer_shapes
import numpy as np
from torch_geometric.data import Data, Dataset
from predictor.feature.graph_feature import extract_graph_feature


class Latency_Encoder():
    def __init__(self, onnx_path, latency_path, embed_type='trans') -> None:  # 初始化参照文件
        self.onnx_dir = onnx_path
        self.latency_file = latency_path
        self.embed_type = embed_type

    def get_data(self, onnx_file, batch_size, cost_time, embed_type):  # 取数据
        adjacent, node_features, static_features, topo_features = extract_graph_feature(onnx_file, batch_size,
                                                                                        embed_type)
       # print("batch_size",batch_size,' embed_type',embed_type)
       # print(" adjacent", adjacent.shape,'static_features',static_features,'topo_features',topo_features)
        edge_index = torch.from_numpy(np.array(np.where(adjacent > 0))).type(torch.long)  # 边集特征
        node_features = np.array(node_features, dtype=np.float32)
        node_features = torch.from_numpy(node_features).type(torch.float)
        x = node_features  # 特征
        sf = torch.from_numpy(static_features).type(torch.float)

        y = torch.FloatTensor([cost_time])  # 延迟
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
        )

        return data, sf

    def encode(self):
        with open(self.latency_file) as f:
            for line in f.readlines():
                # ---------读取文件信息------------
                line = line.rstrip()
                items = line.split(" ")
                graph_id = str(items[1])
                batch_size = int(items[2])
                cost_time = float(items[3])
                # --------------------------------
                onnx_file = os.path.join(self.onnx_dir, graph_id)
               # print('1',onnx_file, '2',batch_size, '3',cost_time,'4', embed_type)
                data, sf = self.get_data(onnx_file, batch_size, cost_time, embed_type)
                #print(data, sf)  # TODO:改成合适的格式输入方便LLM finetune


# For test
if __name__ == '__main__':
    latency_path = 'D:\\NAR\\NAR-Former-V2-main\\NAR-Former-V2-main\\dataset\\unseen_structure\\gt_stage.txt'
    embed_type = 'trans'
    onnx_path = 'D:\\NAR\\NAR-Former-V2-main\\NAR-Former-V2-main\\dataset\\unseen_structure'


    encoder = Latency_Encoder(onnx_path=onnx_path, latency_path=latency_path, embed_type=embed_type)  # TODO:改成yaml格式
    encoder.encode()


