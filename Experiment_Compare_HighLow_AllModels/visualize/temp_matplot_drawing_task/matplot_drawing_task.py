from mydynalearn.logger import Log


class MatplotDrawingTask():
    '''
    Matplot图片绘制任务
    - 准备绘图数据
    - 使用数据绘制图像
    - 保存图像
    '''

    def __init__(self):
        self.logger = Log("MatplotDrawingTask")

    def _get_analyze_result(self, data):
        '''
        输入一行self.best_epoch_dataframe数据，返回对应的分析结果
        :param data: 一行数据记录
        :return:
        '''
        # type is time_evolution or normal_performance
        # normal_performance_analyze_result = ModelAnalyzerNormalPerformance.load_from_dataframe_item(data)
        # return normal_performance_analyze_result
        pass

    def _get_drawing_data_generator(self):
        '''
        准备绘图数据
        :return:
        '''
        # 拿到最佳结果的数据
        best_epoch_dataframe = self.best_epoch_dataframe
        for index, data_info in best_epoch_dataframe.iterrows():
            # 通过数据信息拿到数据结果
            data_result = self._get_analyze_result(data_info)
            # 组成绘图数据
            drawing_data = {
                "data_info": data_info,
                "data_result": data_result,
            }
            yield drawing_data

    def run(self):
        # 绘图数据的generator
        drawing_data_generator = self._get_drawing_data_generator()
        for drawing_data in drawing_data_generator:
            # 遍历每一个数据进行绘制
            fig_drawer = self.run(drawing_data)
            # 图片保存
            fig_drawer.save_fig()

    def run(self, drawing_data):
        pass


class FigYtrureYpredDrawingTask(MatplotDrawingTask):
    def __init__(self):
        super().__init__()



    def run(self, drawing_data):
        '''
        绘制图片run
        :param drawing_data: 绘制数据
        :return:
        '''
        # 结果数据dataframe
        self.logger.log("DRAW FigYtrureYpred")
        data_result = drawing_data['data_result']
        # 图像绘制
        fig_drawer = FigYtrureYpred(**data_result)
        fig_drawer.do_output()  # 绘制
        fig_drawer.edit_ax()  # 编辑
        return fig_drawer


class FigBetaRhoDrawingTask(MatplotDrawingTask):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset


class FigConfusionMatrixDrawingTask(FigYtrureYpredDrawingTask):
    '''
    混淆矩阵表格
    '''

    def __init__(self):
        # todo: 绘制混淆矩阵
        super(FigConfusionMatrixDrawingTask, self).__init__(analyze_manager)


    def run(self, drawing_data):
        '''
        绘制图片run
        :param drawing_data: 绘制数据
        :return:
        '''
        # 结果数据dataframe
        data_result = drawing_data['data_result']
        self.logger.log("DRAW FigConfusionMatrix")
        # 图像绘制
        fig_drawer = FigConfusionMatrix(**data_result)
        fig_drawer.do_output()  # 绘制
        fig_drawer.edit_ax()  # 编辑
        return fig_drawer


class FigActiveNeighborsTransProbDrawingTask(FigYtrureYpredDrawingTask):
    '''
    激活态邻居数量-迁移概率图
    '''

    def __init__(self):
        super(FigActiveNeighborsTransProbDrawingTask, self).__init__(analyze_manager)

    def _get_drawing_data_generator(self):
        '''
        准备绘图数据
        :yield: 一个模型训练的数据，用于画图
        '''
        # 设置配置
        best_epoch_dataframe = self.best_epoch_dataframe
        model_network_name = "SCER"
        model_dynamics_name = "SCUAU"
        dataset_network_name = "SCER"
        dataset_dynamics_name = "SCUAU"
        # 拿到详细信息
        best_epoch_dataframe_ER_UAU = best_epoch_dataframe[
            (best_epoch_dataframe['model_network_name'] == model_network_name) &
            (best_epoch_dataframe['model_dynamics_name'] == model_dynamics_name) &
            (best_epoch_dataframe['dataset_network_name'] == dataset_network_name) &
            (best_epoch_dataframe['dataset_dynamics_name'] == dataset_dynamics_name)
            ]
        # 拿到数据
        for index, data_info in best_epoch_dataframe_ER_UAU.iterrows():
            # 通过数据信息拿到数据结果
            data_result = self._get_analyze_result(data_info)
            # 组成绘图数据
            drawing_data = {
                "data_info": data_info,
                "data_result": data_result,
            }
            yield drawing_data

    def run(self, drawing_data):
        '''
        绘制图片run
        :param drawing_data: 绘制数据
        :return:
        '''
        # 结果数据dataframe
        data_result = drawing_data['data_result']
        self.logger.log("DRAW FigActiveNeighborsTrans")
        # 图像绘制
        fig_drawer = FigActiveNeighborsTransprob(**data_result)
        fig_drawer.do_output()  # 绘制
        fig_drawer.edit_ax()  # 编辑
        return fig_drawer

class FigKLossDrawingTask(FigYtrureYpredDrawingTask):
    '''
    邻居数量-loss图
    '''

    def __init__(self):
        super(FigKLossDrawingTask, self).__init__(analyze_manager)



    def run(self, drawing_data):
        '''
        绘制图片run
        :param drawing_data: 绘制数据
        :return:
        '''
        # 结果数据dataframe
        data_result = drawing_data['data_result']
        self.logger.log("DRAW FigKLoss")
        # 图像绘制
        fig_drawer = FigKLoss(**data_result)
        fig_drawer.do_output()  # 绘制
        fig_drawer.edit_ax()  # 编辑
        return fig_drawer

class FigKDistributionDrawingTask(FigYtrureYpredDrawingTask):
    '''
    网络度分布图
    X：一阶度
    Y：如果是高阶网络，则为二阶度
    '''

    def __init__(self):
        super(FigKDistributionDrawingTask, self).__init__(analyze_manager)

    def _get_drawing_data_generator(self):
        '''
        准备绘图数据
        :yield: 一个模型训练的数据，用于画图
        '''
        # 按照 'model_network_name' 分类，并在每个分类中选择第一个条目
        first_entries_per_model = self.best_epoch_dataframe.groupby('model_network_name').first().reset_index()

        # 拿到数据
        for index, data_info in first_entries_per_model.iterrows():
            # 通过数据信息拿到数据结果
            data_result = self._get_analyze_result(data_info)
            # 组成绘图数据
            drawing_data = {
                "data_info": data_info,
                "data_result": data_result,
            }
            yield drawing_data


    def run(self, drawing_data):
        '''
        绘制图片run
        :param drawing_data: 绘制数据
        :return:
        '''
        # 结果数据dataframe
        data_result = drawing_data['data_result']
        network = data_result['network']
        self.logger.log("DRAW FigKDistribution")
        # 图像绘制
        fig_drawer = FigKDistribution(**data_result)
        fig_drawer.do_output()  # 绘制
        fig_drawer.edit_ax()  # 编辑
        return fig_drawer

class FigTimeEvolutionDrawingTask(FigYtrureYpredDrawingTask):
    '''
    时间演化图
    '''
    def __init__(self):
        super(FigTimeEvolutionDrawingTask, self).__init__(analyze_manager)

    def _get_analyze_result(self, data):
        '''
        输入一行self.best_epoch_dataframe数据，返回对应的分析结果
        :param data: 一行数据记录
        :return:
        '''
        # type is time_evolution or normal_performance
        normal_performance_analyze_result = ModelAnalyzerTimeEvolution.load_from_dataframe_item(data)
        return normal_performance_analyze_result


    def run(self, drawing_data):
        '''
        绘制图片run
        :param drawing_data: 绘制数据
        :return:
        '''
        # 结果数据dataframe
        data_result = drawing_data['data_result']
        network = data_result['network']
        self.logger.log("DRAW TimeEvolution")
        # 图像绘制
        fig_drawer = FigTimeEvolution(**data_result)
        fig_drawer.do_output()  # 绘制
        fig_drawer.edit_ax()  # 编辑
        return fig_drawer