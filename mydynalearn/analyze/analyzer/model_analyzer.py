from mydynalearn.analyze.analyze_result_handler import ModelResultHandler
class ModelAnalyzer():
    def __init__(self, config_analyze, all_epoch_analyzer):
        self.config_analyze = config_analyze
        self.all_epoch_analyzer = all_epoch_analyzer
        self.model_result_handler = ModelResultHandler(config_analyze, all_epoch_analyzer)

    def run(self):
        self.model_result_handler.run()

    def get_analyze_result(self):
        dataset = self.model_result_handler.get_dataset()
        metadata = self.model_result_handler.get_metadata()
        dataset.update(metadata)
        return dataset