# define task prompts for BigBench penguins_in_a_table task
from tasks.bigbench import CustomTask as BigBenchTask

class CustomTask(BigBenchTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size=None,  
                 
                 task_name = "penguins_in_a_table",
                 task_description = "Task from BigBench about penguins in a table",
                 # 修改这里指向绝对路径或相对路径
                 data_dir='data/penguins_in_a_table.json',  # 使用相对路径
                 seed=None, 
                 
                 post_instruction=True, 
                 option_num=5, 
                 **kwargs):
        
        # 如果数据路径未指定或找不到，使用默认位置
        import os
        if data_dir and not os.path.exists(data_dir):
            # 尝试不同的路径组合
            potential_paths = [
                data_dir,
                os.path.join('app', data_dir),
                os.path.join('app/data/tasks', os.path.basename(data_dir)),
                os.path.join('data', os.path.basename(data_dir)),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', os.path.basename(data_dir))
            ]
            
            # 查找第一个存在的路径
            for path in potential_paths:
                if os.path.exists(path):
                    data_dir = path
                    break
        
        super().__init__(
                        task_name = task_name,  
                        task_description = task_description, 
                        data_dir=data_dir,
                        seed = seed,
                        train_size = train_size,
                        eval_size=eval_size,
                        test_size = test_size,
                        post_instruction = post_instruction,
                        option_num=option_num,
                        **kwargs
                        )