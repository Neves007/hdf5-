from mydynalearn.analyze.analyze_result_handler import EpochResultHandlerGeneralPerformance
class EpochAnalyzer():
    def __init__(self, config_analyze, epoch_task):
        self.config_analyze = config_analyze
        self.epoch_task = epoch_task
        self.epoch_result_handler_general_performance = EpochResultHandlerGeneralPerformance(config_analyze, epoch_task)

    def run(self):
        self.epoch_result_handler_general_performance.run()


    def get_analyze_result(self):
         self.epoch_result_handler_general_performance.get_dataset()