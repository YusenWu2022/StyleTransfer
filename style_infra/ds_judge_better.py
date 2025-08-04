import json
from utils import ds_14b_chat, init_ds_model

prompt = '''
你将收到以下内容：
1. **初始文本**：一段原始文本，包含特定的研究内容和表达主题。
2. **风格文段**：若干段具有特定风格的文本，用于参考风格转换的目标风格。
3. **风格转换文本**：两段经过风格转换的文本，旨在将初始文本转换为与风格文段相近的风格。这两段文本的语义相近，但风格不同。

请根据以下标准对两段风格转换后的文本进行评分：
1. **风格一致性**：转换后的文本是否与风格文段的表达风格一致，包括语言风格、语气、词汇选择等方面。
2. **内容与主题保持**：转换后的文本是否保持了初始文本的研究内容、表达主题和行文思路。
3. **表达规范性**：转换后的文本是否遵循普遍公认的写作基准，例如逻辑性、真实性和可读性。

你对每段文本的打分应为一个三元向量 [x, y, z]，其中：
- x 表示风格一致性得分；
- y 表示内容与主题保持得分；
- z 表示表达规范性得分。
分数范围为 0 到 10 分，分数越高表示效果越好。

请为每段风格转换后的文本提供一个评分向量，并给出简要的评分理由。

初始文本：
{init_text}

风格文段：
{style_text}

风格转换文本1：
{transformed}

'''

# 示例输入
initial_text = "这是一篇关于人工智能在医疗领域应用的研究报告。"
style_segments = [
    "这是一篇充满激情的演讲，激励人们追求梦想。",
    "这是一篇幽默风趣的散文，讲述生活中的点滴趣事。"
]
converted_text_1 = "这篇报告以幽默风趣的方式探讨了人工智能在医疗领域的应用。"
converted_text_2 = "这篇报告以激昂的语气阐述了人工智能在医疗领域的巨大潜力。"

params = {
    "init_text": initial_text,
    "style_text": style_segments,
    "transformed": converted_text_1
}


prompt_filled = prompt.format(**params)
print(prompt_filled)


with open("/root/pku/yusen/research_assistant/style_infra/style/direct_rewritten.json", "r", encoding="utf-8") as file:
    direct_transfer = json.load(file)

with open("/root/pku/yusen/research_assistant/style_infra/style/rewritten.json", "r", encoding="utf-8") as file:
    structured_transfer = json.load(file)

with open("/root/pku/yusen/research_assistant/style_infra/style/original_paras.json", "r", encoding="utf-8") as file:
    origin_paras = json.load(file) 
    
with open("/root/pku/yusen/research_assistant/style_infra/style/summaried_style_with_id.json", "r", encoding="utf-8") as file:
    style_paras = json.load(file) 
    
model, tokenizer = init_ds_model(0)
# for i in range(len(direct_transfer)):
#     params = {
#     "init_text": origin_paras[i],
#     "style_text": style_paras[i],
#     "transformed": structured_transfer[i]
#     }
#     prompt_filled = prompt.format(**params)
#     response = ds_14b_chat(prompt_filled, model, tokenizer)
    
#     try:
#         with open("/root/pku/yusen/research_assistant/style_infra/style/structured_score.json", 'r', encoding='utf-8') as json_file:
#             data = json.load(json_file)
#             if not isinstance(data, list):
#                 raise ValueError("JSON 文件内容不是一个列表")
#     except (FileNotFoundError, json.JSONDecodeError):
#         data = []
#     data.append(response)
#     with open("/root/pku/yusen/research_assistant/style_infra/style/structured_score.json", 'w', encoding='utf-8') as json_file:
#         json.dump(data, json_file, ensure_ascii=False, indent=4)
        
for i in range(len(direct_transfer)):
    params = {
    "init_text": origin_paras[i],
    "style_text": style_paras[i],
    "transformed": direct_transfer[i]
    }
    prompt_filled = prompt.format(**params)
    response = ds_14b_chat(prompt_filled, model, tokenizer)
    
    try:
        with open("/root/pku/yusen/research_assistant/style_infra/style/direct_transfer_score.json", 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            if not isinstance(data, list):
                raise ValueError("JSON 文件内容不是一个列表")
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    data.append(response)
    with open("/root/pku/yusen/research_assistant/style_infra/style/direct_transfer_score.json", 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)