from ...visualize_handler import LossK1K2Handler
class LossK1K2Drawer():
    def __init__(self, analyze_result_output):
        self.name = "LossK1K2Drawer"
        self.visualizer_dir = f"output/fig/{self.name}"
        self.set_visualizer_data(analyze_result_output)
        pass

    def set_visualizer_data(self, analyze_result_output):
        self.visualizer_data_df = analyze_result_output['max_r_rows']
        pass


    def run(self):
        for index, visualizer_data in self.visualizer_data_df.iterrows():
            visualize_hanlder = LossK1K2Handler(self.visualizer_dir, visualizer_data)
            visualize_hanlder.run()

