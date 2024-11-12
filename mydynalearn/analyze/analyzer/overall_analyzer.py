from mydynalearn.analyze.analyze_result_handler import OverallResultHandler


class OverallAnalyzer():
    def __init__(self, config_analyze, all_model_analyzer):
        self.config_analyze = config_analyze
        self.all_model_analyzer = all_model_analyzer
        self.overall_result_handler = OverallResultHandler(config_analyze, all_model_analyzer)

    def run(self):
        self.overall_result_handler.run()
    def get_analyze_result(self):
        pass
