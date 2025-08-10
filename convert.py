import json
import re


def ultra_robust_converter(input_file, output_file):
    # 读取原始内容（强制UTF-8编码）
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 预处理步骤
    content = re.sub(r'\n\s*\n', '\n', content)  # 清理多余空行
    blocks = re.split(r'(?<=\})\s*?(?=\{)', content)  # 智能分割对象

    # 解析JSON对象
    json_array = []
    error_count = 0
    for idx, block in enumerate(blocks, 1):
        try:
            # 基础修复
            block = block.strip()
            if not block.startswith('{'):
                block = '{' + block.split('{', 1)[-1]
            if not block.endswith('}'):
                block = block.rsplit('}', 1)[0] + '}'

            # 解析并验证
            obj = json.loads(block)
            json_array.append(obj)
        except Exception as e:
            error_count += 1
            print(f"错误 题目{idx}: {str(e)}")
            print("问题内容片段:", block[:100] + ('...' if len(block) > 100 else ''))

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_array, f, indent=2, ensure_ascii=False)

    print(f"转换完成，成功 {len(json_array)} 题，失败 {error_count} 题")


if __name__ == "__main__":
    ultra_robust_converter(
        input_file="benchmarck2.0.md",
        output_file="benchmarck2.0.json"
    )