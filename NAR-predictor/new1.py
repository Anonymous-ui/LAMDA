import os

# 要保存的文件路径
output_file = 'D:\\NAR\\NAR-Former-V2-main\\NAR-Former-V2-main\\dataset\\unseen_structure\\help.txt'

# onnx 目录的路径
onnx_dir = 'D:/NAR/NAR-Former-V2-main/NAR-Former-V2-main/dataset/unseen_structure/onnx/help'

# 初始化一个空列表来存储信息
file_info_list = []

# 获取目录中的所有 .onnx 文件
for root, dirs, files in os.walk(onnx_dir):
    for file in files:
        if file.endswith('.onnx'):
            # 获取文件的相对路径
            relative_path = os.path.relpath(os.path.join(root, file), onnx_dir)
            # 将路径分隔符统一替换为正斜杠
            relative_path = relative_path.replace(os.sep, '/')
            # 提取文件的名称部分（不含扩展名）
            file_name = os.path.splitext(file)[0]
            # 获取文件序号，这里简单的使用列表长度加1
            file_index = len(file_info_list) + 4  # 假设从 00004 开始
            # 将信息格式化
            file_info = f"{file_index:05d} onnx/help/{relative_path} 1 {file_name.split('/')[-1]} help 1 test"
            file_info_list.append(file_info)

# 将信息写入到 txt 文件中
with open(output_file, 'w') as f:
    for info in file_info_list:
        f.write(info + '\n')

print(f"信息已保存到 {output_file}")
