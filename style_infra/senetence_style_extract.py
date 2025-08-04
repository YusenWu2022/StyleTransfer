
import json
import os
import json
import os
from multiprocessing import Pool
import math


# apply style to new paper
# （模板中的{}中的内容为特定领域内容，而{}外面的内容为骨架模板）
prompt_transfer_ch = '''根据下面给出的风格模板字典，为后面给定的语段中的句子进行风格模板匹配。风格模板字典包括多项，每项都为一个底朝天，其中pattern字段为句子的模板，每项的id字段为编号。
要求对给定语段逐句进行匹配，每一句都在风格模板字典项中找到风格模板逻辑格式最相似的项（找到句式逻辑关系最为相似的id即可，不需要内容的相似，且无论如何必须输出一个最相似的id）。
输出应该为一个list，里面的元素为一个字典，字典包括两个键：语段句子原文，以及其匹配到的最相似的风格模板逻辑格式编号。
下面是给出的风格模板字典'''


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


def process_article(article_path, model, tokenizer, device_id, format_path):
    try:
        # 处理单个文章
        text = load_pdf(article_path)
        # 分片处理（每10000字符）
        # chunk_size = 2000
        # chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        chunks = text.split('  ')
        # 处理每个分片
        for idx, chunk in enumerate(chunks):
            # 生成临时文件名
            base_name = os.path.basename(article_path).replace(".", "")
            temp_file = f"style/extract_code_{base_name}_{idx}.json"
            # 如果临时文件已存在则跳过
            if os.path.exists(temp_file):
                continue
            # 处理分片内容
            format = load_json(format_path)
            input_query = prompt_transfer_ch + str(format) + '''
            下面是给定的需要逐句匹配模板的语段
            '''+chunk
            ans = ds_14b_chat(input_query, model, tokenizer)
            # 保存分片结果到临时文件
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump([ans], f, ensure_ascii=False)
        # 合并当前文章的所有分片结果
        merged = []
        for idx in range(len(chunks)):
            temp_file = f"style/extract_code_{base_name}_{idx}.json"
            try:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    merged.extend(json.load(f))
                os.remove(temp_file)  # 删除临时文件
            except Exception as e:
                print(f"加载临时文件 {temp_file} 失败: {str(e)}")
        # 保存最终合并结果【
        final_file = f"style/{base_name}_extract_code_final.json"
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=4)
        return final_file
    
    except Exception as e:
        print(f"处理文章 {article_path} 时发生错误: {str(e)}")
        return None

def style_transfer(args):
    articles_subset, device_id, format_path = args
    # 设置当前进程使用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    # 初始化模型
    from transformers import AutoTokenizer, AutoModelForCausalLM
    deepseek_14b_path = '/root/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
    tokenizer = AutoTokenizer.from_pretrained(deepseek_14b_path)
    model = AutoModelForCausalLM.from_pretrained(deepseek_14b_path, device_map=f"cuda:0")
    results = []
    for article in articles_subset:
        result = process_article(article, model, tokenizer, device_id, format_path)
        if result:
            results.append(result)
    return results

if __name__ == "__main__":
    # 配置参数
    paper2transfer_path = "/root/pku/yusen/research_assistant/simple_search_paper/downloaded_pdfs"
    format_path = "/root/pku/yusen/research_assistant/style_infra/style/summaried_style_with_id.json"
    articles = get_file_paths(paper2transfer_path)
    
    # GPU配置
    num_gpus = 2
    devices = [1,2, 3]
    
    # 分割任务
    chunk_size = math.ceil(len(articles) / num_gpus)
    tasks = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = (i+1) * chunk_size
        subset = articles[start:end]
        tasks.append( (subset, devices[i], format_path) )
    
    # 创建进程池并行处理
    with Pool(processes=num_gpus) as pool:
        all_results = pool.map(style_transfer, tasks)
    
    