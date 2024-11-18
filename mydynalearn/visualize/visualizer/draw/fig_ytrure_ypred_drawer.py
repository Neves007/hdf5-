from ...visualize_handler import FigYtrureYpredHandler
class FigYtrureYpredDrawer():
    def __init__(self, analyze_result_output):
        self.name = "FigYtrureYpredDrawer"
        self.visualizer_dir = f"output/fig/{self.name}"
        self.set_visualizer_data(analyze_result_output)
        pass

    def set_visualizer_data(self, analyze_result_output):
        self.visualizer_data_df = analyze_result_output['max_r_rows']
        pass


    def run(self):
        for index, visualizer_data in self.visualizer_data_df.iterrows():
            visualize_hanlder = FigYtrureYpredHandler(self.visualizer_dir, visualizer_data)
            visualize_hanlder.run()
