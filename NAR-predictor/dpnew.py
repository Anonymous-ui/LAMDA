import json
from transformers import AutoTokenizer
import re

# 初始化 Qwen 的 tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("", trust_remote_code=True)
except Exception as e:
    exit(1)


def read_json(file_path):
    print(f"{file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(file_path, data):
    print(f" {file_path}")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def count_tokens(text):
    """计算文本的 token 数量"""
    tokens = tokenizer.tokenize(text)
    return len(tokens)


def calculate_weights(lines):

    weights = [0] * len(lines)


    input_ids = {}
    output_ids = {}

    for i, line in enumerate(lines):
        matches_input = re.findall(r'input_id:(\d+(?: and \d+)*)', line)
        matches_output = re.search(r'output_id:(\d+)', line)

        if matches_input:
            ids = set(int(id.strip()) for id in matches_input[0].split('and'))
            for id in ids:
                input_ids.setdefault(id, []).append(i)

        if matches_output:
            output_id = int(matches_output.group(1))
            output_ids[i] = output_id


    for i in range(len(lines)):
        current_output_ids = set()
        for j in range(i + 1):
            if j in output_ids:
                current_output_ids.add(output_ids[j])


        for output_id in current_output_ids:
            if output_id in input_ids:
                for idx in input_ids[output_id]:
                    if idx > i:
                        weights[i] += 1

    return weights



def dpprocess_conversation(item, item_index, total_items, max_token_limit=1000):

    print(f"{item_index + 1}/{total_items}")
    assistant_value = item['conversations'][1]['value']
    lines = assistant_value.split('\n')

    # 计算权重
    weights = calculate_weights(lines)

    print("所有行的最终权重:")
    for idx, weight in enumerate(weights):
        print(f"行 {idx + 1}: {weight}")

    # 动态规划算法初始化
    dp = [float('inf')] * len(lines)
    cut_point = [-1] * len(lines)
    dp[0] = count_tokens(lines[0])
    cut_point[0] = -1

    for i in range(1, len(lines)):
        for j in range(i + 1):
            segment_text = '\n'.join(lines[j:i + 1])
            segment_tokens = count_tokens(segment_text)
            if segment_tokens <= max_token_limit:
                new_cost = dp[j - 1] + weights[i] if j > 0 else weights[i]
                if new_cost < dp[i]:
                    dp[i] = new_cost
                    cut_point[i] = j

    print("最终的 dp[] 数组:")
    for idx, value in enumerate(dp):
        print(f"dp[{idx}] = {value}")


    cut_points = []
    i = len(lines) - 1
    while i >= 0:
        start = cut_point[i]
        if start != -1:
            cut_points.append((start, i))
            print(f"回溯: 当前切割位置从行 {start + 1} 到行 {i + 1}")
        i = start - 1

    cut_points.reverse()


    segments = []
    prev_end = 0
    for start, end in cut_points:
        segment_text = '\n'.join(lines[prev_end:end + 1])
        segments.append(segment_text)
        prev_end = end + 1

    # 处理最后一段
    if prev_end < len(lines):
        segment_text = '\n'.join(lines[prev_end:])
        segments.append(segment_text)

    print(f"对话项 {item_index + 1}/{total_items} 分割完成，共分割为 {len(segments)} 部分")
    return segments


def create_convo(new_id, user_question, assistant_response, part_number):

    return {
        "id": new_id,
        "conversations": [
            {
                "from": "user",
                "value": f"{user_question} (part {part_number})"
            },
            {
                "from": "assistant",
                "value": assistant_response
            }
        ]
    }


def main(input_file, output_file):
    try:
        data = read_json(input_file)
        new_data = []
        new_id = 1
        total_items = len(data)

        # 遍历原始数据并进行拆分
        for item_index, item in enumerate(data):
            segments = dpprocess_conversation(item, item_index, total_items)
            for part_number, segment in enumerate(segments, start=1):
                new_conversation = create_convo(new_id, item['conversations'][0]['value'], segment, part_number)
                new_data.append(new_conversation)
                new_id += 1
            print(f" {item_index + 1}/{total_items} s")

        write_json(output_file, new_data)


    except Exception as e:
        print(f"error: {e}")


if __name__ == "__main__":
    input_file = ''
    output_file = ''
    main(input_file, output_file)