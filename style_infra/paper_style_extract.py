import json
import os
import json
import os
from multiprocessing import Pool
import math

# extract sections

# extract pattern
# prompt0_ch = '''
#   请阅读下面文章，整理出论文分为哪几个部分；每一句话总结后去重，每个部分的句子逻辑表达基本结构本质上哪几种
#   要求足够细化，使得这些列举出来的学术论文写作句子表达逻辑结构在人工智能学术论文写作中能够覆盖相应部分的全部表达结构。
#   不同的文章的这些基本结构在用词风格和具体内容上有不同，但整体逻辑是相似的。
#   可以用{xxx}来指代句子中可替换的具体内容，每一类都要完整总结出来，不要省略
#   要求对每一句话进行结构拆分；
#   对这些不同句子的结构进行去重；
#   最后得到的结构模板{}内包含的内容不需要写出，外面的内容要求是抽象的，不要包含和文章相关的具体信息.
#   输出应该类似如下["xxx","xxx","xxx"]（包括两侧的方括号）。
#   为一个严格的包含多个字符串的list，可用json解析，不要输出额外内容。其中每个字符串为一个句式模板。
#   下面给出需要抽取逻辑表达基本结构的论文文本。
# '''
prompt0_ch = '''
  你是一个总结文献写作风格的助理。请阅读下面研究文段，逐句抽象和总结出句子逻辑表达的基本结构骨架，过程中注意这些要求：
  1、要求总结出来的结构骨架只包含表达习惯上的语句结构，对句子中和具体研究内容有关的词句，用{xxx}来替代，例如"Here we adapt {method} to prove {conclusion}."；
  2、注意，要求严格对对每一句话进行结构拆分，不要跳过；
  3、对于基本结构骨架相似的句子，进行去重，保留它们的共同结构骨架；
  4、输出应该类似如下["xxx","xxx","xxx"]（包括两侧的方括号），为一个严格的包含多个字符串的list，可用json解析，不要输出额外内容。其中每个字符串为一个句式模板。
  下面给出需要抽取逻辑表达基本结构骨架的论文文本。
  
'''

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

struture = '''

{
  "Abstract": {
    "研究背景": "{领域}近年来发展迅速，但{问题}仍然存在。",
    "研究方法": "本文提出{方法}，结合{技术1}和{技术2}来解决{问题}。",
    "研究目标": "旨在{目标}，并支持{功能}。",
    "研究贡献": "通过{实验}和{评估}验证了{方法}的有效性。"
  },
  "Introduction": {
    "研究背景": "{领域}近年来发展迅速，但{问题}仍然存在。",
    "研究重要性": "{问题}对{领域}具有重要意义，因为{原因}。",
    "研究现状": "已有研究主要集中在{方向1}和{方向2}，但{不足}。",
    "研究目标": "本文旨在通过{方法}解决{问题}。",
    "研究贡献": "本文的主要贡献包括：{贡献1}：提出{方法}以解决{问题}。{贡献2}：设计{系统}支持{功能}。{贡献3}：通过{实验}验证{方法}的有效性。",
    "研究结构": "本文结构如下：第{X}节介绍{内容}，第{Y}节描述{方法}，最后总结{结论}。"
  },
  "Related Work": {
    "领域概述": "{领域}近年来受到广泛关注，主要研究方向包括{方向1}、{方向2}和{方向3}。",
    "已有研究": "{方法}被用于{任务}，但{不足}。",
    "本文贡献": "与{已有研究}不同，本文{创新点}。"
  },
  "Observational Study": {
    "专家实践": "通过与{专家}合作，我们观察到{现象}。",
    "专家瓶颈": "{问题}限制了{实践}。",
    "专家需求": "{需求}需要{解决方案}。",
    "研究假设": "基于{观察}，我们提出{假设}。"
  },
  "Approach Overview": {
    "系统设计": "{系统名称}包含{模块1}和{模块2}，分别用于{功能1}和{功能2}。",
    "工作流程": "{模块1}处理{数据}，{模块2}进行{分析}。",
    "系统目标": "{系统}旨在帮助{用户}完成{任务}。"
  },
  "Back-End Engine": {
    "数据处理": "{方法}用于将{数据类型}转化为{特征}。",
    "特征提取": "通过{技术}提取{特征}。",
    "模型选择": "{模型}因其{优势}被选为{任务}的候选模型。",
    "模型训练": "使用{数据集}训练{模型}。",
    "模型评估": "通过{指标}评估{模型}的性能。"
  },
  "Front-End Visualization": {
    "界面设计": "{视图名称}用于展示{信息类型}，支持{交互功能}。",
    "视图功能": "{视图}允许用户{操作}，以便{目的}。",
    "交互设计": "用户可以通过{交互方式}调整{参数}，观察{结果}。"
  },
  "Evaluation": {
    "案例研究": "通过{案例}验证了{系统}在{任务}中的{效果}。",
    "用户研究": "{方法}用于评估{系统}的{性能}。",
    "实验结果": "{结果}表明{系统}在{指标}上优于{对比方法}。",
    "专家反馈": "{专家}认为{系统}在{方面}表现良好。"
  },
  "Discussion and Limitation": {
    "系统优势": "{系统}在{方面}表现良好，因为{原因}。",
    "局限性": "{问题}限制了{系统}在{场景}中的应用。",
    "未来工作": "{改进方向}将被探索以解决{问题}。"
  },
  "Conclusion and Future Work": {
    "研究总结": "本文提出{系统}，通过{方法}解决了{问题}。",
    "研究意义": "{系统}为{领域}提供了{价值}。",
    "未来展望": "未来工作将探索{方向}以提升{性能}。"
  },
  "Title": {
    "句子逻辑表达基本结构": "{研究主题}：{研究方法或技术}用于{研究目标或应用领域}。"
  },
  "Authors": {
    "句子逻辑表达基本结构": "{作者姓名}，{作者单位}，{作者邮箱}。"
  },
  "Keywords": {
    "句子逻辑表达基本结构": "{关键词1}，{关键词2}，{关键词3}，{关键词4}。"
  },
  "Problem Formulation": {
    "研究目标": "本文的目标是{研究目标}。",
    "变量定义": "定义{变量名}为{变量描述}。",
    "约束条件": "{约束条件描述}。",
    "优化问题": "{优化问题描述}。",
    "解决方案框架": "我们提出了一种{解决方案框架}，{框架描述}。"
  },
  "Method": {
    "方法概述": "我们比较了{方法1}和{方法2}，并引入了{新方法}。",
    "方法细节": "{方法名称}通过{方法细节}实现。",
    "数据来源": "所有方法都基于{数据来源}。",
    "模型训练": "模型使用{训练方法}进行训练。",
    "性能评估": "模型性能通过{评估指标}进行评估。"
  },
  "Experimental Study": {
    "实验设计": "我们进行了{实验类型}，{实验设计描述}。",
    "实验结果": "{实验结果描述}。",
    "结果分析": "{结果分析描述}。",
    "比较分析": "{方法1}在{指标}上优于{方法2}，{比较分析描述}。",
    "动态调整": "{动态调整策略}，{动态调整效果}。"
  },
  "Conclusions": {
    "研究总结": "本文提出了一种{研究方法}，{研究方法的作用或优势}。",
    "研究结果": "通过{实验或测试}，结果表明{研究结果或结论}。",
    "研究贡献": "{研究贡献或创新点}。",
    "未来研究": "未来的研究可以{未来研究方向}。"
  }
}'''
# apply style to new paper
prompt2_ch = '''根据下面给出的风格模板，修改下面的语段
要求逐句进行修改，每一句都在相应部分的风格模板格式项中进行匹配，找到含义表征最接近的项，参考该项内容对原句进行改写，使得维持和文章具体内容相关信息不变的情况下，让表达风格转换为匹配到的项。
逐句替换后最终输出应该和输入的语段保持段数和段长接近,用一段英语自然语言表达。'''

# 匹配抽象出的风格表达项并修改原文对应句子
prompt3_ch = '''
现有论文写作中不同部分中不同句式模板骨架，但不包含风格（风格即这些骨架具体表达时所用的衔接词和格式等，与{}中由文章具体研究内容决定的具体专有用词无关），
现在请遍历下面给定的新论文片段中的每个句子，在模板骨架中寻找与新论文句子含义相近的骨架，将新论文中的具体研究内容填充到匹配的的句式骨架框架里面，然后用填充组合好的句子替换新论文中对应的句子构成最终输出，要求用英文。
'''

# 总结每个section的核心逻辑思路，提取共同模板

# 筛选出主要的idea确定不同section的任务，包括method部分中分为几部分，用一句话描述每部分的主题思路

# 根据总的section的主题思路，对每个section展开文段生成，生成每句话的时候都要使用模板


# 自动化查找引用


prompt_transfer='''下面是学术论文的若干逻辑表达基本结构，每个部分有若干项
请对每一部分中的基本结构逐个进行整理，如果语义不重复则直接沿用原来的功能描述和表达模板，如果一部分中存在语义逻辑重复的项，则进一步抽象出它们的表达模板作为结果;在表达模板中{}中代表针对不同任务可以修改的内容，{}外面是抽象出来的与具体任务无关的格式语句，这些语句代表了论文的写作风格。
最后的结果要求写为用json可解析的dict，类似
{
“abstract”:
{"xxx":"xxxx"...“xxx”:"xxxx"}
"function":
{"xxx":"xxxx",..."xxx":"xxxx"}
...
}

下面是需要处理的风格格式项
Abstract
研究背景：{领域}近年来发展迅速，但{问题}仍然存在。
研究方法：本文提出{方法}，结合{技术1}和{技术2}来解决{问题}。
研究目标：旨在{目标}，并支持{功能}。
研究贡献：通过{实验}和{评估}验证了{方法}的有效性。
Introduction
研究背景：{领域}近年来发展迅速，但{问题}仍然存在。
研究重要性：{问题}对{领域}具有重要意义，因为{原因}。
研究现状：已有研究主要集中在{方向1}和{方向2}，但{不足}。
研究目标：本文旨在通过{方法}解决{问题}。
研究贡献：本文的主要贡献包括：
{贡献1}：提出{方法}以解决{问题}。
{贡献2}：设计{系统}支持{功能}。
{贡献3}：通过{实验}验证{方法}的有效性。
研究结构：本文结构如下：第{X}节介绍{内容}，第{Y}节描述{方法}，最后总结{结论}。
Related Work
领域概述：{领域}近年来受到广泛关注，主要研究方向包括{方向1}、{方向2}和{方向3}。
已有研究：{方法}被用于{任务}，但{不足}。
本文贡献：与{已有研究}不同，本文{创新点}。
Observational Study
专家实践：通过与{专家}合作，我们观察到{现象}。
专家瓶颈：{问题}限制了{实践}。
专家需求：{需求}需要{解决方案}。
研究假设：基于{观察}，我们提出{假设}。
Approach Overview
系统设计：{系统名称}包含{模块1}和{模块2}，分别用于{功能1}和{功能2}。
工作流程：{模块1}处理{数据}，{模块2}进行{分析}。
系统目标：{系统}旨在帮助{用户}完成{任务}。
Back-End Engine
数据处理：{方法}用于将{数据类型}转化为{特征}。
特征提取：通过{技术}提取{特征}。
模型选择：{模型}因其{优势}被选为{任务}的候选模型。
模型训练：使用{数据集}训练{模型}。
模型评估：通过{指标}评估{模型}的性能。
Front-End Visualization
界面设计：{视图名称}用于展示{信息类型}，支持{交互功能}。
视图功能：{视图}允许用户{操作}，以便{目的}。
交互设计：用户可以通过{交互方式}调整{参数}，观察{结果}。
Evaluation
案例研究：通过{案例}验证了{系统}在{任务}中的{效果}。
用户研究：{方法}用于评估{系统}的{性能}。
实验结果：{结果}表明{系统}在{指标}上优于{对比方法}。
专家反馈：{专家}认为{系统}在{方面}表现良好。
Discussion and Limitation
系统优势：{系统}在{方面}表现良好，因为{原因}。
局限性：{问题}限制了{系统}在{场景}中的应用。
未来工作：{改进方向}将被探索以解决{问题}。
Conclusion and Future Work
研究总结：本文提出{系统}，通过{方法}解决了{问题}。
研究意义：{系统}为{领域}提供了{价值}。
未来展望：未来工作将探索{方向}以提升{性能}。
1. 标题（Title）
句子逻辑表达基本结构：
{研究主题}：{研究方法或技术}用于{研究目标或应用领域}。
2. 作者信息（Authors）
句子逻辑表达基本结构：
{作者姓名}，{作者单位}，{作者邮箱}。
3. 摘要（Abstract）
句子逻辑表达基本结构：
{研究背景}：{研究主题}已成为{研究领域}的重要组成部分。
{研究问题}：然而，{研究问题或挑战}。
{研究方法}：本文提出了一种基于{研究方法或技术}的{研究主题}。
{研究贡献}：该方法{研究贡献或创新点}。
{研究结果}：通过{实验或测试}，结果表明{研究结果或结论}。
4. 关键词（Keywords）
句子逻辑表达基本结构：
{关键词1}，{关键词2}，{关键词3}，{关键词4}。
5. 引言（Introduction）
句子逻辑表达基本结构：
{研究背景}：{研究领域}提供了{研究背景或现象}。
{研究问题}：然而，{研究问题或挑战}。
{研究目标}：本文旨在{研究目标}。
{研究方法}：通过{研究方法或技术}，{研究方法的作用或优势}。
{研究贡献}：本文的主要贡献包括：{列举研究贡献}。
{研究结构}：本文的结构如下：{研究结构概述}。
6. 问题定义（Problem Formulation）
句子逻辑表达基本结构：
{研究目标}：本文的目标是{研究目标}。
{变量定义}：定义{变量名}为{变量描述}。
{约束条件}：{约束条件描述}。
{优化问题}：{优化问题描述}。
{解决方案框架}：我们提出了一种{解决方案框架}，{框架描述}。
7. 方法（Method）
句子逻辑表达基本结构：
{方法概述}：我们比较了{方法1}和{方法2}，并引入了{新方法}。
{方法细节}：{方法名称}通过{方法细节}实现。
{数据来源}：所有方法都基于{数据来源}。
{模型训练}：模型使用{训练方法}进行训练。
{性能评估}：模型性能通过{评估指标}进行评估。
8. 实验（Experimental Study）
句子逻辑表达基本结构：
{实验设计}：我们进行了{实验类型}，{实验设计描述}。
{实验结果}：{实验结果描述}。
{结果分析}：{结果分析描述}。
{比较分析}：{方法1}在{指标}上优于{方法2}，{比较分析描述}。
{动态调整}：{动态调整策略}，{动态调整效果}。
9. 结论（Conclusions）
句子逻辑表达基本结构：
{研究总结}：本文提出了一种{研究方法}，{研究方法的作用或优势}。
{研究结果}：通过{实验或测试}，结果表明{研究结果或结论}。
{研究贡献}：{研究贡献或创新点}。
{未来研究}：未来的研究可以{未来研究方向}。'''

prompt_summary_by_section ='''
对下面给定的论文进行总结，分出不同section对应的内容，论述写作表达的思路内容，（如果有的话）要求细化到理论论证了某个内容、实验设计了某个具体实验、某个具体流程的设计、具体的背景和动机论述，特定方法的设计等方面的具体细节，按照顺序逐点给出。
要求对每个段落文本层次进行考察，总结主题思路
总结出来的分段主题思路项应适用于让学者学习做研究时的规划
要求输出格式类似如下，每个section的list里面都是完整的一个内容单元的总结，不同总结之间要按照原文抽象出联系
要求把每个具体的思路抽象为做了什么，如何服务于主题/子主题
将某项中局限于具体研究内容的具体行为或观察的全部具体文本用{xxx}替代，这里的xxx是抽象出来的具体行为或观察在文中的地位或内涵
{
“abstract":["xxx,xxx","xxx"..."xxx"],
"introduction:["xxx,xxx","xxx"..."xxx"],
...

}
'''

prompt_read ='''
下面给出若干文章-summary，请遍历并根据summary找到和给定主题最可能相关的文章，输出形式类似如下
{“article”:"xxx", "usage":"to xxx"}
{“article”:"xxx", "usage":"to xxx"}
{“article”:"xxx", "usage":"to xxx"}
请阅读下面给出的文章，找到能够起到用途{usage}的片段并进行总结，将能够用于论文的引用内容和延申进行返回，使之适用于论文写作的引用部分。要求输出格式类似如下
{“article”:"xxx", "usage":"to xxx", "content":"xxx"}
{“article”:"xxx", "usage":"to xxx", "content":"xxx"}

'''


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
def extract_style(args):
    articles_subset, device_id, output_sub_path = args
    # 设置当前进程使用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    
    # 初始化模型（每个进程独立加载）
    from transformers import AutoTokenizer, AutoModelForCausalLM
    deepseek_14b_path = '/root/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
    tokenizer = AutoTokenizer.from_pretrained(deepseek_14b_path)
    model = AutoModelForCausalLM.from_pretrained(deepseek_14b_path, device_map=f"cuda:0")
    
    style_collection = []
    for article in articles_subset:
        try:
            text = load_pdf(article)
            input_query = prompt0_ch + text
            ans = ds_14b_chat(input_query, model, tokenizer)
            ans = extract_json_block(ans)
            
            # 解析结果
            # try:
            #     results = json.loads(ans) if ans.startswith('[') else []
            # except:
            #     results = [s.strip('"') for s in ans.strip('[]').split(', ')]
            results = [s.strip('"') for s in ans.strip('[]').split(', ')]
            style_collection.extend(results)
        except Exception as e:
            print(f"处理文件 {article} 时出错: {str(e)}")
    
    # 保存子结果
    with open(output_sub_path, 'w', encoding='utf-8') as f:
        json.dump(style_collection, f, ensure_ascii=False, indent=4)

# 初始化函数（不再需要，各进程独立初始化）
def init_ds_model():
    pass  

# 合并结果函数
def merge_results(output_path, temp_files):
    merged_data = []
    for temp_file in temp_files:
        try:
            with open(temp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                merged_data.extend(data)
            # os.remove(temp_file)  # 删除临时文件
        except Exception as e:
            print(f"合并文件 {temp_file} 时出错: {str(e)}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # 配置参数
    pdf_path = "/root/pku/yusen/research_assistant/simple_search_paper/downloaded_pdfs"
    style_path = "/root/pku/yusen/research_assistant/style_infra/style/style.json"
    articles = get_file_paths(pdf_path)
    
    # GPU配置（假设使用4个GPU）
    num_gpus = 4
    temp_files = [f"style/style_temp_{i}.json" for i in range(num_gpus)]
    
    # 分割任务
    chunk_size = math.ceil(len(articles) / num_gpus)
    tasks = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = (i+1) * chunk_size
        subset = articles[start:end]
        tasks.append( (subset, i+4, temp_files[i]) )
    
    # 创建进程池并行处理
    with Pool(processes=num_gpus) as pool:
        pool.map(extract_style, tasks)
    
    # 合并结果
    merge_results(style_path, temp_files)
  


