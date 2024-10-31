from tqdm import tqdm
from mydynalearn.analyze.analyzer import runModelOnTestData,epochAnalyzer
from mydynalearn.config import Config
import os

class AnalyzeManager():
    def __init__(self,train_experiment_manager):
        '''
        分析epoch

        :param train_experiment_manager:
        '''
        config_analyze = Config.get_config_analyze()
        self.config = config_analyze['default']
        self.train_experiment_manager = train_experiment_manager
        self.epoch_analyzer = epochAnalyzer(self.config)

    def _get_model_executor_generator(self):
        exp_generator = self.train_experiment_manager.get_exp_generator()  # 所有的实验对象
        for exp in exp_generator:
            # 读取数据
            model_executor = runModelOnTestData(self.config,
                                                model_exp=exp,
                                                dataset_exp=exp,
                                                )
            yield model_executor

    def get_analyze_result_generator_for_best_epoch(self):
        '''
        最佳结果的生成器
        :return:
        '''
        model_executor_generator = self._get_model_executor_generator()
        for model_executor in model_executor_generator:
            best_epoch_index = self.epoch_analyzer.get_best_epoch_index(**model_executor.exp_info)
            EPOCHS = model_executor.EPOCHS
            best_epoch = list(range(EPOCHS))[best_epoch_index]
            best_epoch_analyze_result = model_executor.run(best_epoch)
            yield best_epoch_analyze_result
            
    def get_analyze_result_generator_for_all_epochs(self):
        '''
        所有结果的生成器
        :return:
        '''
        model_executor_generator = self._get_model_executor_generator()
        for model_executor in model_executor_generator:
            EPOCHS = model_executor.EPOCHS
            for model_exp_epoch_index in range(EPOCHS):
                analyze_result = model_executor.run(model_exp_epoch_index)
                yield analyze_result

    def buid_anayze_result(self):
        '''
        将所有实验的数据集引入到自己的模型中，输出analyze_result
        '''
        model_executor_generator = self._get_model_executor_generator()
        for model_executor in model_executor_generator:
            EPOCHS = model_executor.EPOCHS
            for model_exp_epoch_index in range(EPOCHS):
                need_to_run = model_executor.check_run_necessity(model_exp_epoch_index)
                if need_to_run:
                    model_executor.run(model_exp_epoch_index)

    def analyze_all_epochs(self):
        all_epoch_dataframe_is_exist = os.path.exists(self.epoch_analyzer.all_epoch_dataframe_file_path)
        if not all_epoch_dataframe_is_exist:
            analyze_result_generator = self.get_analyze_result_generator_for_all_epochs()
            for analyze_result in analyze_result_generator:
                    self.epoch_analyzer.add_epoch_result(analyze_result)
            self.epoch_analyzer.save_all_epoch_dataframe()
        else:
            self.epoch_analyzer.load_all_epoch_dataframe()


    def analyze_best_epoch(self):
        '''最好的epoch

        io:
        ''' 
        if not os.path.exists(self.epoch_analyzer.best_epoch_dataframe_file_path):
            self.epoch_analyzer.analyze_best_epoch()
        else:
            self.epoch_analyzer.load_best_epoch_dataframe()

    def analyze_time_evolution(self):
        '''分析时间演化
        生成真实的动力学模型，和预测的动力学模型生成的时间序列对比数据
        :io:
        '''
        # 获取best_epoch的analyze_result
        best_epoch_analyze_result_generator = self.get_analyze_result_generator_for_best_epoch()
        for best_epoch_analyze_result in best_epoch_analyze_result_generator:
            true_dynamics = best_epoch_analyze_result['dynamics']
        pass

    def run(self):
        '''
        分析训练数据，为画图做准备
        输出：
        '''
        print("*" * 10 + " ANALYZE TRAINED MODEL " + "*" * 10)
        print("buid anayze result")
        self.buid_anayze_result()
        print("analyze all epochs")
        self.analyze_all_epochs()
        print("analyze best epoch")
        self.analyze_best_epoch()
        print()
        self.analyze_time_evolution()




