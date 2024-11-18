from ...analyze.analyze_result_handler import ResultsAggregatorHandler

class ResultsAggregatorAnalyzer():

    def get_output(self):
        return self.results_aggregator_handler.get_output()

    def run(self, all_epoch_analyzer_dict):
        self.results_aggregator_handler = ResultsAggregatorHandler(all_epoch_analyzer_dict)
        self.results_aggregator_handler.run()
