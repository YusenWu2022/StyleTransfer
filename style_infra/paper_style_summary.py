import json
import os
import json
import os
from multiprocessing import Pool
import math



# from multiple patterns to overall style
prompt1_ch = '''下面是学术论文的若干逻辑表达基本结构，每个部分有若干项
请对每一部分中的基本结构逐个进行整理，如果语义不重复则直接沿用原来的功能描述和表达模板，如果一部分中存在语义逻辑重复的项，则进一步抽象出它们的表达模板作为结果;在表达模板中{}中代表针对不同任务可以修改的内容，{}外面是抽象出来的与具体任务无关的格式语句，这些语句代表了论文的写作风格。
最后的结果应该类似如下["xxx","xxx","xxx"]（包括两侧的方括号）
下面是需要处理的风格格式项'''


def init_ds_model():
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    deepseek_14b_path = '/root/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
    tokenizer = AutoTokenizer.from_pretrained(deepseek_14b_path)
    model = AutoModelForCausalLM.from_pretrained(deepseek_14b_path, device_map="cuda:0")
    return model, tokenizer

def load_pdf(path):
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(path)
    docs = loader.load()
    text = ''
    for doc in docs:
        text += doc.page_content
    text = text[:40000]
    return text

def get_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def ds_14b_chat(input, model, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=10000,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def extract_json_block(text):
    start_marker = "```json"
    end_marker = "```"
    start_index = text.find(start_marker)
    end_index = text.find(end_marker, start_index + len(start_marker))
    if start_index == -1 or end_index == -1:
        return text
    code_block = text[start_index + len(start_marker):end_index].strip()
    return code_block

def load_json(path):
    data = []
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def save_to_json(data, filename: str) -> None:
    """
    将数据保存到JSON文件
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []
    if not isinstance(existing_data, list):
        existing_data = []
    existing_data.extend(data)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

# 提取样式函数，修改为接受设备ID和子任务
def style_summary(original_style_path, output_sub_path):
    
    model, tokenizer = init_ds_model()
    style_collection = []
    styles = load_json(original_style_path)
    input_query = prompt1_ch + str(styles)
    ans = ds_14b_chat(input_query, model, tokenizer)
    ans = extract_json_block(ans)
    results = [s.strip('"') for s in ans.strip('[]').split(', ')]
    style_collection.extend(results)
    with open(output_sub_path, 'w', encoding='utf-8') as f:
        json.dump(style_collection, f, ensure_ascii=False, indent=4)
 


if __name__ == "__main__":
    # 配置参数
    output_style_path = "/root/pku/yusen/research_assistant/style_infra/style/summaried_style.json"
    style_path = "/root/pku/yusen/research_assistant/style_infra/style/style.json"
    style_summary(style_path, output_style_path)
    d = load_json(output_style_path)
    s = []
    for i in range(len(d)):
        s.append({"pattern": d[i], "id":i})
    save_to_json(s, "/root/pku/yusen/research_assistant/style_infra/style/summaried_style_with_id.json")
  


