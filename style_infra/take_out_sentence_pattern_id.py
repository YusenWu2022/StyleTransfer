import json
def extract_json_block(text):
    start_marker = "```json"
    end_marker = "```"
    start_index = text.find(start_marker)
    end_index = text.find(end_marker, start_index + len(start_marker))
    if start_index == -1 or end_index == -1:
        return text
    code_block = text[start_index + len(start_marker):end_index].strip()
    return code_block
def extract_kuohao_block(text):
    start_marker = "["
    end_marker = "]"
    start_index = text.find(start_marker)
    end_index = text.rfind(end_marker)+1
    if start_index == -1 or end_index == -1:
        return None
    code_block = text[start_index:end_index+1].strip()
    return code_block
response_path = "/root/pku/yusen/research_assistant/style_infra/style/250405206v1pdf_extract_code_final.json"
json_responses = []
with open(response_path, "r", encoding="utf-8") as file:
    paragraphs = json.load(file)
    for paragraph in paragraphs:
        json_response = extract_json_block(paragraph)
        json_response = extract_kuohao_block(json_response)
        if json_response is not None:
            json_responses.append(json_response)
print(len(json_responses))
sentences_with_ids = []
for json_response in json_responses:
    try:
        sentences_with_id =  json.loads(json_response)
        sentences_with_ids.append(sentences_with_id)
    except:
        # print(json_response)
        continue
print(len(sentences_with_ids))

paragraph_sentence_pattern = []
for sentences_with_id in sentences_with_ids:
    result_dict = {}
    for d in sentences_with_id:
        for key, value in d.items():
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(value)
    paragraph_sentence_pattern.append(result_dict)
print(len(paragraph_sentence_pattern))
# for sentence_pattern in paragraph_sentence_pattern:
#     print(sentence_pattern.keys)

def trans(input):
    return [int(i) for i in input]
rename_paras = []
for d in paragraph_sentence_pattern:
    keys = list(d.keys())
    new_dict = {
        "原文": d[keys[0]],  
        "匹配模板编号": trans(d[keys[1]])  
    }
    rename_paras.append(new_dict)
print(len(rename_paras))
# for i in rename_paras:  
#     print(i['匹配模板编号'])

para_patterns = []
original_paras = []
for i in rename_paras:  
    para_patterns.append(i['匹配模板编号'])
    original_paras.append(i['原文'])
para_pattern_file = "/root/pku/yusen/research_assistant/style_infra/style/para_patterns.json"
original_paras_file = "/root/pku/yusen/research_assistant/style_infra/style/original_paras.json"
with open(para_pattern_file, "w", encoding="utf-8") as file:
    json.dump(para_patterns, file, ensure_ascii=False, indent=4)
with open(original_paras_file, "w", encoding="utf-8") as file:
    json.dump(original_paras, file, ensure_ascii=False, indent=4)