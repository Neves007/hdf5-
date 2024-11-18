
from .visualizer.getter import get as get_visualizer
class visualizerManager():
    def __init__(self,analyze_result_output):
        self.analyze_result_output = analyze_result_output
        self.visualizer_name_list = [
            "FigYtrureYpred",
        ]

    def run(self):
        for visualizer_name in self.visualizer_name_list:
            Visualizer = get_visualizer(visualizer_name)
            visualizer = Visualizer(self.analyze_result_output)
            visualizer.run()




