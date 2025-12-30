from collections import OrderedDict
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..TrainJob import TrainJob
from .BaseTrainProcess import BaseTrainProcess


class TrainFineTuneProcess(BaseTrainProcess):
    def __init__(self, process_id: int, job: 'TrainJob', config: OrderedDict):
        super().__init__(process_id, job, config)

    def run(self):
        # implement in child class
        # be sure to call super().run() first
        pass
