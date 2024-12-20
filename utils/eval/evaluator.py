from typing import Any, List, Optional, Dict
from warnings import warn

import numpy as np

from .evaluation import evaluate_list


class Evaluator:
    LABELS = 'labels'
    PREDICTIONS = 'predictions'

    def __init__(self, dataset: str):
        super().__init__()
        assert dataset in {'waymo'}
        self.dataset = dataset

        self._buffer = None
        self._buffer_empty = True
        self._reset_buffer()

    def _reset_buffer(self):
        self._buffer_empty = True
        self._buffer = {
            self.LABELS: list(),
            self.PREDICTIONS: list(),
        }

    def _add_to_buffer(self, key: str, value: List[np.ndarray]):
        assert isinstance(value, list)
        for entry in value:
            assert isinstance(entry, np.ndarray)
        self._buffer_empty = False
        assert self._buffer is not None
        self._buffer[key].extend(value)

    def _get_from_buffer(self, key: str) -> List[np.ndarray]:
        assert not self._buffer_empty
        assert self._buffer is not None
        return self._buffer[key]

    def add_predictions(self, predictions: List[np.ndarray]):
        self._add_to_buffer(self.PREDICTIONS, predictions)

    def add_labels(self, labels: List[np.ndarray]):
        self._add_to_buffer(self.LABELS, labels)

    def reset_buffer(self) -> None:
        # E.g. call in on_validation_epoch_start
        self._reset_buffer()

    def has_data(self):
        return not self._buffer_empty

    def evaluate_buffer(self, img_height: int, img_width: int) -> Optional[Dict[str, Any]]:
        # e.g call in on_validation_epoch_end
        if self._buffer_empty:
            warn("Attempt to use prophesee evaluation buffer, but it is empty", UserWarning, stacklevel=2)
            return

        print("getting labels")
        labels = self._get_from_buffer(self.LABELS)
        predictions = self._get_from_buffer(self.PREDICTIONS)

        print("got labels", len(labels))
        assert len(labels) == len(predictions)

        print("starting evaluation")
        metrics = evaluate_list(result_boxes_list=predictions,
                                gt_boxes_list=labels,
                                height=img_height,
                                width=img_width,
                                camera=self.dataset)
        return metrics
