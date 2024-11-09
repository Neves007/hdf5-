from mydynalearn.analyze.analyzer import EpochAnalyzer

class EpochAnalyzeManager():
    def __init__(self,config_analyze,exp_generator):
        self.config_analyze = config_analyze
        self.exp_generator = exp_generator
        self.epoch_analyzer_generator = self.get_epoch_analyzer_generator()

    def get_epoch_analyzer_generator(self):
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
        for epoch_analyzer in list(self.epoch_analyzer_generator):
            epoch_analyzer.run()
