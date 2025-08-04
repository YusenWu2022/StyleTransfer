import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from utils import init_ds_model, ds_14b_chat
import json

def extract_kuohao_block(text):
    start_marker = "["
    end_marker = "]"
    start_index = text.find(start_marker)
    end_index = text.rfind(end_marker)+1
    if start_index == -1 or end_index == -1:
        return None
    code_block = text[start_index:end_index+1].strip()
    return code_block

# ------------------- 第一部分：生成段落模板 -------------------

def compute_edit_distance(seq1, seq2):
    """计算两个序列的编辑距离"""
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m+1, n+1))
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n]

def cluster_paragraphs(paragraphs, max_clusters=5):
    """层次聚类生成段落模板（修正对称性问题）"""
    n = len(paragraphs)
    # 计算所有i<j的编辑距离，保存为一维向量
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            distances.append(compute_edit_distance(paragraphs[i], paragraphs[j]))
    # 将一维向量转换为对称矩阵
    distance_matrix = squareform(distances)
    # 层次聚类
    Z = linkage(distance_matrix, method='average')
    clusters = fcluster(Z, max_clusters, criterion='maxclust')
    # 生成模板（此处简化逻辑，实际应计算中心序列）
    templates = {}
    for cluster_id in np.unique(clusters):
        cluster_data = [paragraphs[i] for i in range(n) if clusters[i] == cluster_id]
        templates[cluster_id] = cluster_data[0]
    return templates

# ------------------- 第二部分：模板匹配与对齐 -------------------

def needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-1):
    """Needleman-Wunsch全局序列对齐算法"""
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n+1, m+1))
    # 初始化
    for i in range(n+1):
        dp[i][0] = i * gap
    for j in range(m+1):
        dp[0][j] = j * gap
    # 填充矩阵
    for i in range(1, n+1):
        for j in range(1, m+1):
            match_score = dp[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
            delete = dp[i-1][j] + gap
            insert = dp[i][j-1] + gap
            dp[i][j] = max(match_score, delete, insert)
    # 回溯路径
    i, j = n, m
    align1, align2 = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch):
            align1.append(seq1[i-1])
            align2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + gap:
            align1.append(seq1[i-1])
            align2.append(None)  # 插入gap
            i -= 1
        else:
            align1.append(None)   # 插入gap
            align2.append(seq2[j-1])
            j -= 1
    return list(reversed(align1)), list(reversed(align2))


# ------------------- 第三部分：调用大模型改写 -------------------

def construct_prompt(original_sentences, template_structure):
    """构建大模型输入的Prompt"""
    prompt = f"""
    # 任务说明
    将以下原始段落按给定的风格参考段落改写，要求：
    - 保留关键信息，确保逻辑连贯
    - 在风格参考段落中选出较为接近的句子作为当前风格化句子的参考模板依据
    - 使得改写后的行文风格和参考段落接近，包括两个维度：段落结构的风格接近和使用句式骨架的风格接近

    # 段落模板骨架
    {template_structure}

    # 原始段落
    {chr(10).join(original_sentences)}

    # 改写要求
    请逐句对应模板结构，调整句式但保留原意，并确保段落连贯。
    """
    return prompt

def rewrite_with_gpt(original_text, template_structure, model, tokenizer):
    """调用大模型进行改写（示例使用OpenAI API）"""
    prompt = construct_prompt(original_text, template_structure)
    response = ds_14b_chat(prompt, model, tokenizer)
    return response

# ------------------- 主流程示例 -------------------

if __name__ == "__main__":
    model,tokenizer = init_ds_model(1)
    
    # 示例数据：每个段落是模板ID序列
    train_paragraphs = [
        [1, 3, 5],
        [2, 3, 4],
        [1, 3, 5, 2],
        [2, 4, 4]
    ]
    
    input_file = "/root/pku/yusen/research_assistant/style_infra/style/para_patterns.json"
    with open(input_file, "r", encoding="utf-8") as file:
        train_paragraphs = json.load(file)
    with open("/root/pku/yusen/research_assistant/style_infra/style/original_paras.json", "r", encoding="utf-8") as file:
        original_paras = json.load(file)
    
    # 抽象出的句式模板
    with open("/root/pku/yusen/research_assistant/style_infra/style/summaried_style_with_id.json", "r", encoding="utf-8") as file:
        pattern_sentences = json.load(file)
    
    # 遍历处理新段落
    # 需要先load test段落
    rewritten_list = []
    for para_id in range(len(train_paragraphs)):
        
        original_sentences = original_paras[para_id]
    
        template_structure = [pattern_sentence['pattern'] for pattern_sentence in pattern_sentences]
        rewritten = rewrite_with_gpt(original_sentences, template_structure, model, tokenizer)
        rewritten = extract_kuohao_block(rewritten)
        
        try:
            with open("/root/pku/yusen/research_assistant/style_infra/style/direct_rewritten.json", 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                if not isinstance(data, list):
                    raise ValueError("JSON 文件内容不是一个列表")
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        data.append(rewritten)
        with open("/root/pku/yusen/research_assistant/style_infra/style/direct_rewritten.json", 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
    
    
    