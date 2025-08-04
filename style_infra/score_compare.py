import json
import ast

def extract_and_parse_list(input_string):
    stack = [] 
    result = []  
    for i, char in enumerate(input_string):
        if char == '[':
            stack.append(i)  
        elif char == ']':
            if stack:  
                start_index = stack.pop()
                substring = input_string[start_index:i + 1]  
                if substring.count(',') == 2:
                    result.append(substring)
            else:
                continue
    if result:
        try:
            return  ast.literal_eval(result[0])
        except:
            return None
    else:
        return None


structured_score_path = "/root/pku/yusen/research_assistant/style_infra/style/structured_score.json"
direct_transfer_score_path = "/root/pku/yusen/research_assistant/style_infra/style/direct_transfer_score.json"
with open(structured_score_path, 'r', encoding='utf-8') as json_file:
    structured_score = json.load(json_file)
with open(direct_transfer_score_path, 'r', encoding='utf-8') as json_file:
    direct_transfer_score = json.load(json_file)

structured_score_count = []
for i in structured_score:
    score = extract_and_parse_list(i)
    if score is not None:
        structured_score_count.append(score)
average_list = [0] * len(structured_score_count[0])
for i in range(len(structured_score_count[0])):
    sum_at_position = sum(lst[i] for lst in structured_score_count)
    average_list[i] = sum_at_position / len(structured_score_count)
print(f"structured_score:{average_list}")

direct_transfer_score_count = []
for i in direct_transfer_score:
    score = extract_and_parse_list(i)
    if score is not None:
        direct_transfer_score_count.append(score)
average_list = [0] * len(direct_transfer_score_count[0])
for i in range(len(direct_transfer_score_count[0])):
    sum_at_position = sum(lst[i] for lst in direct_transfer_score_count)
    average_list[i] = sum_at_position / len(direct_transfer_score_count)
print(f"direct_transfer_score:{average_list}")
