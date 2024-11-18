
from mydynalearn.visualize.visualizer.getter import get as get_visualizer
class visualizerManager():
    def __init__(self,analyze_result_output):
        self.analyze_result_output = analyze_result_output
        # todo 更新参数存档点 4：修改visualizer_name_list调整所需可视化任务
        self.visualizer_name_list = [
            "LossK1K2"
            "FigYtrureYpred"
        ]

    def run(self):
        for visualizer_name in self.visualizer_name_list:
            Visualizer = get_visualizer(visualizer_name)
            visualizer = Visualizer(self.analyze_result_output)
            visualizer.run()




