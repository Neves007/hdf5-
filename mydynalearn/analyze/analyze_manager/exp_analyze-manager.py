

class ExpAnalyzeManager():
    def __init__(self,config_analyze,exp_generator):
        self.config_analyze = config_analyze
        self.exp_generator = exp_generator

    @property
    def analyze_result_generator(self):
        for epoch_analyzer in self.epoch_analyzer_generator:
            yield epoch_analyzer.get_analyze_result()

    @property
    def epoch_analyzer_generator(self):
        '''
        把每个exp的每个epoch_task分配给EpochAnalyzer
        :return:
        '''
        for exp in self.exp_generator:
            for epoch_task in exp.model.epoch_task_generator:
                epoch_analyzer = EpochAnalyzer(self.config_analyze,epoch_task)
                yield epoch_analyzer

    def run(self):
        '''
        EpochAnalyzer 对每个epoch_task结果进行分析
        :return:
        '''
        for epoch_analyzer in self.epoch_analyzer_generator:
            epoch_analyzer.run()
