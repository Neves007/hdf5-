import pandas as pd

class ResultsAggregatorHandler():
    def __init__(self, all_epoch_analyzer_dict):
        self.all_epoch_analyzer_dict = all_epoch_analyzer_dict


    def _init_result(self):
        """
        初始化结果，将 epoch_task 的测试结果转换为存储格式。
        """
        self.result = pd.DataFrame(self.all_epoch_analyzer_dict)

    def find_max_r_rows(self):
        # 要检查的列列表
        columns_to_check = ['NETWORK_NAME', 'DYNAMIC_NAME', 'MODEL_NAME', 'IS_WEIGHT', 'DEVICE',
                            'NUM_NODES', 'SEED_FREC', 'NUM_SAMPLES', 'T_INIT', 'EPOCHS']

        # 按 columns_to_check 分组，并查找每组中 R 值最大的行
        max_r_rows = self.result.loc[self.result.groupby(columns_to_check)['R'].idxmax()]
        return max_r_rows

    def _analyze_result(self):
        max_r_rows = self.find_max_r_rows()
        self.analyze_result = {
            "max_r_rows": max_r_rows,
        }
        return max_r_rows

    def run(self):
        self._init_result()
        self._analyze_result()
        pass

    def get_output(self):
        return self.analyze_result