from Dao import DataHandler
from abc import abstractmethod

class AnalyzeResultHandler(DataHandler):
    def __init__(self,parent_group, cur_group):
        DataHandler.__init__(self, parent_group, cur_group)
        pass

    @abstractmethod
    def init_result(self):
        '''
        初始化要分析的结果
        :return:
        '''
        pass

    @abstractmethod
    def analyze_result(self):
        '''
        分析结果
        :return: 分析的结果
        '''
        pass


    def build_dataset(self):
        self.init_result()
        dataset = self.analyze_result()
        self.set_dataset(dataset)
