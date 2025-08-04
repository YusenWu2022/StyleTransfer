
import json
import os
import json
import os
from multiprocessing import Pool
import math


# from multiple patterns to overall style
prompt1_ch = '''下面是学术论文的若干逻辑表达基本结构，每个部分有若干项
请对每一部分中的基本结构逐个进行整理，如果语义不重复则直接沿用原来的功能描述和表达模板，如果一部分中存在语义逻辑重复的项，则进一步抽象出它们的表达模板作为结果;在表达模板中{}中代表针对不同任务可以修改的内容，{}外面是抽象出来的与具体任务无关的格式语句，这些语句代表了论文的写作风格。
最后的结果要求写为用json可解析的dict，类似
{
“abstract”:
{"xxx":"xxxx"...“xxx”:"xxxx"}
"function":
{"xxx":"xxxx",..."xxx":"xxxx"}
...
}

下面是需要处理的风格格式项'''


# apply style to new paper
prompt_transfer_ch = '''根据下面给出的风格模板，修改后面给定的语段。
要求逐句进行修改，每一句都在相应部分的风格模板格式项中进行匹配，找到含义表征最接近的项，参考该项内容对原句进行改写，使得维持文章具体内容和相关信息不变的情况下，让语段骨架表达风格转换为匹配到的模板格式项的风格（把文章具体内容填入模板格式项的{}中，达到替换衔接用词、语气词...等的目标）。
逐句替换后最终输出应该和输入的语段保持段数和段长接近,用一段英语自然语言表达。下面是给出的风格模板'''

# 匹配抽象出的风格表达项并修改原文对应句子
prompt3_ch = '''
现有论文写作中不同部分中不同句式模板骨架，但不包含风格（风格即这些骨架具体表达时所用的衔接词和格式等，与{}中由文章具体研究内容决定的具体专有用词无关），
现在请遍历下面给定的新论文片段中的每个句子，在模板骨架中寻找与新论文句子含义相近的骨架，将新论文中的具体研究内容填充到匹配的的句式骨架框架里面，然后用填充组合好的句子替换新论文中对应的句子构成最终输出，要求用英文。
'''

# 总结每个section的核心逻辑思路，提取共同模板

# 筛选出主要的idea确定不同section的任务，包括method部分中分为几部分，用一句话描述每部分的主题思路

# 根据总的section的主题思路，对每个section展开文段生成，生成每句话的时候都要使用模板



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
        chunk_size = 10000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        # 处理每个分片
        for idx, chunk in enumerate(chunks):
            # 生成临时文件名
            base_name = os.path.basename(article_path).replace(".", "")
            temp_file = f"style/temp_{base_name}_{idx}.json"
            # 如果临时文件已存在则跳过
            if os.path.exists(temp_file):
                continue
            # 处理分片内容
            format = load_json(format_path)
            input_query = prompt_transfer_ch + str(format) + '''
            下面是需要修改的文段部分
            '''+chunk
            ans = ds_14b_chat(input_query, model, tokenizer)
            # 保存分片结果到临时文件
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump([ans], f, ensure_ascii=False)
        # 合并当前文章的所有分片结果
        merged = []
        for idx in range(len(chunks)):
            temp_file = f"style/temp_{base_name}_{idx}.json"
            try:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    merged.extend(json.load(f))
                os.remove(temp_file)  # 删除临时文件
            except Exception as e:
                print(f"加载临时文件 {temp_file} 失败: {str(e)}")
        # 保存最终合并结果【
        final_file = f"style/{base_name}_final.json"
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
    format_path = "/root/pku/yusen/research_assistant/style_infra/style/style.json"
    articles = get_file_paths(paper2transfer_path)
    
    # GPU配置
    num_gpus = 2
    devices = [4, 5, 6, 7]
    
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
    
    