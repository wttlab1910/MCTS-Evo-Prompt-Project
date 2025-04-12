# define task prompts for various datasets
import re
import sys
import os

# 添加数据目录到路径
sys.path.append(r"D:\FYP\MCTSEvo-Prompt-Project\backend")

try:
    # 尝试直接从data模块导入
    from data import load_dataset
except ImportError:
    # 如果失败，尝试添加data目录到路径
    try:
        sys.path.append(r"D:\FYP\MCTSEvo-Prompt-Project\backend\data")
        from load_dataset import load_dataset
    except ImportError:
        # 最后的备用实现
        def load_dataset(name, name2=None):
            """简单的备用load_dataset实现"""
            print(f"使用最小备用load_dataset: {name}, {name2}")
            
            # 根据任务名称创建简单的数据集结构
            if name == "super_glue" and name2 == "cb":
                return {
                    "train": [
                        {"premise": "The cat is on the mat.", "hypothesis": "There is a cat.", "label": 0}
                    ],
                    "validation": [
                        {"premise": "The dog is playing.", "hypothesis": "The dog is sleeping.", "label": 1}
                    ]
                }
            # 需要时为其他数据集添加更多情况
            
            return {"train": [], "validation": []}

from tasks.base_task import BaseDataset, BaseTask


class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size = None,  
                 
                 task_name = 'cb', 
                 task_discription = "",
                 seed=None, 
                 
                 post_instruction=True, 
                 **kwargs):
        super().__init__(task_name=task_name,
                         task_discription=task_discription,
                         seed=seed, 
                         train_size=train_size, 
                         eval_size=eval_size,
                         test_size = test_size, 
                         post_instruction=post_instruction,
                         )

        self.answer_format_prompt = '\nA:'
    
    def load_task_dataset(self, **kwargs):
        dataset = load_dataset('super_glue','cb')
        answer_dict = {0:'entailment', 1:'contradiction', 2:'neutral'}
        question_format = "Premise: {premise}\nHypothesis: {hypothesis}\nThat is the relationship between the preceding premise and the hypothesis?\nOptions:\n- Contradiction\n- Neutral\n- Entailment"
        new_dataset = dict(train=[], test=[])
        for example in dataset['train']:
            question_str = question_format.format(
                premise=example['premise'], 
                hypothesis = example['hypothesis'],
                )
            new_dataset['train'].append(dict(question=question_str, answer=answer_dict[example['label']]))
        for example in dataset['validation']:
            question_str = question_format.format(
                premise=example['premise'], 
                hypothesis = example['hypothesis'],
                )
            new_dataset['test'].append(dict(question=question_str, answer=answer_dict[example['label']]))
            
        return new_dataset
    
    def transform_format(self, data):
        return data
    
    def clean_response(self, response):
        clean_pattern = r"\b(entailment|contradiction|neutral)\b"
        match = re.findall(clean_pattern, response.lower())
        if len(match) != 0:
            return match[-1]
    
        return "N/A: format error."