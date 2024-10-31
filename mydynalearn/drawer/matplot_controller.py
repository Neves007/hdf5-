import torch
from mydynalearn.analyze.analyzer import *
from mydynalearn.drawer.matplot_drawing_task.matplot_drawing_task import *
import os
class MatplotController():
    def __init__(self,analyze_manager):
        self.TASKS = {
            "FigYtrureYpredDrawingTask": FigYtrureYpredDrawingTask(analyze_manager),
            "FigActiveNeighborsTransProb": FigActiveNeighborsTransProbDrawingTask(analyze_manager),
            "FigConfusionMatrix": FigConfusionMatrixDrawingTask(analyze_manager),
            "FigKLoss": FigKLossDrawingTask(analyze_manager),
            "FigKDistribution": FigKDistributionDrawingTask(analyze_manager),
            "FigTimeEvolution": FigTimeEvolutionDrawingTask(analyze_manager),
        }
    def run(self):
        for task in self.TASKS.keys():
            if task in self.TASKS:
                f = self.TASKS[task]
                f.run()
            else:
                raise ValueError(
                    f"{task} is an invalid task, possible tasks are `{self.TASKS}`"
                )




