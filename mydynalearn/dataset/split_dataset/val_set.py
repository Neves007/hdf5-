from .train_set import TrainSet
from mydynalearn.logger import Log
class ValSet(TrainSet):
    def __init__(self, config, parent_group):
        self.parent_group = parent_group
        self.cur_group = "val_set"
        super().__init__(config, self.parent_group,self.cur_group)
        self.log = Log("ValSet")

    def init_metadata(self):
        dataset_config = self.config.dataset
        metadata = {}
        metadata['NUM_VAL'] = int((dataset_config['NUM_SAMPLES'] - dataset_config['NUM_TEST'])/2)
        metadata['IS_WEIGHT'] = dataset_config['IS_WEIGHT']
        self.set_metadata(metadata)
