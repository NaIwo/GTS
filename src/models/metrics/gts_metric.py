from src.datasets import Dataset
from src.datasets.domain import IgnoreIndex, GTSMatrixID, Sentence
from src.config_reader import config
from ..utils import trim2d

import tensorflow as tf
from typing import List, TypeVar, Dict, Optional

S = TypeVar('S', bound='Span')


class Span:
    sentence_id: int = 0

    def __init__(self, left: int, right: int, up: Optional[int] = None, down: Optional[int] = None,
                 sentiment: Optional[int] = None):
        self.id: int = Span.sentence_id
        self.left: int = left
        self.right: int = right
        self.up: int = up
        self.down: int = down
        self.sentiment: int = sentiment

    def __hash__(self) -> int:
        return hash((self.id, self.left, self.right, self.up, self.down, self.sentiment))

    def __str__(self) -> str:
        return f'{self.id}-{self.left}-{self.right}-{self.up}-{self.down}-{self.sentiment}'

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def from_word_list(cls, spans: List) -> List[S]:
        return [cls(left, right) for left, right in spans]


class GtsMetric:  # (tf.metrics.Metric)

    def __init__(self, **kwargs):
        # super(GtsMetric, self).__init__(name=source, **kwargs)
        self.true_target_spans: List[Span] = list()
        self.true_opinion_spans: List[Span] = list()
        self.pred_target_spans: List[Span] = list()
        self.pred_opinion_spans: List[Span] = list()
        self.true_union_spans: List[Span] = list()
        self.pred_union_spans: List[Span] = list()

    def reset_state(self):
        self.true_target_spans = list()
        self.true_opinion_spans = list()
        self.pred_target_spans = list()
        self.pred_opinion_spans = list()
        self.true_union_spans = list()
        self.pred_union_spans = list()
        Span.sentence_id = 0

    def update_state(self, y_true: Dataset, y_pred: tf.Tensor, sample_weight=None):
        y_true_gts_matrix: tf.Tensor = trim2d(tf.convert_to_tensor(y_true.gts_matrix), y_pred.shape[1])
        y_pred = tf.math.argmax(y_pred, axis=-1)
        y_pred = tf.where(y_true_gts_matrix == IgnoreIndex.IGNORE_INDEX.value, y_true_gts_matrix, y_pred)

        def get_span_ranges(matrix: tf.Tensor, type_value: int) -> List[Span]:
            diag: tf.Tensor = tf.linalg.diag_part(matrix)
            span: List[List] = [[]]

            max_len: int = true_data.sentence_length

            idx: int
            for idx in range(max_len):
                start: int
                start, _ = true_data.token_range[idx]
                if diag[start] == type_value:
                    if len(span[-1]) == 0:
                        span[-1].append(start)
                elif diag[start] != type_value:
                    if len(span[-1]) > 0:
                        span[-1].append(true_data.token_range[idx - 1][0])
                        span.append([])
                if idx + 1 == max_len:
                    if len(span[-1]) > 0:
                        span[-1].append(start)
                    else:
                        span = span[:-1]

            return Span.from_word_list(span)

        prediction: tf.Tensor
        true_data: Sentence
        true_gts_matrix: tf.Tensor
        for prediction, true_data, true_gts_matrix in zip(y_pred, y_true.sentences, y_true_gts_matrix):
            self.true_target_spans += get_span_ranges(true_gts_matrix, GTSMatrixID.TARGET.value)
            self.true_opinion_spans += get_span_ranges(true_gts_matrix, GTSMatrixID.OPINION.value)
            self.pred_target_spans += get_span_ranges(prediction, GTSMatrixID.TARGET.value)
            self.pred_opinion_spans += get_span_ranges(prediction, GTSMatrixID.OPINION.value)

            self.true_union_spans += self.get_union_ranges(true_gts_matrix, source_name='true')
            self.pred_union_spans += self.get_union_ranges(prediction, source_name='pred')

            Span.sentence_id += 1

    def get_union_ranges(self, matrix: tf.Tensor, source_name: str) -> List[Span]:
        ranges: List[Span] = list()

        target_values: List[Span] = list(
            filter(lambda item: item.id == Span.sentence_id, getattr(self, f'{source_name}_target_spans')))
        opinion_values: List[Span] = list(
            filter(lambda item: item.id == Span.sentence_id, getattr(self, f'{source_name}_opinion_spans')))

        t_span: Span
        for t_span in target_values:
            o_span: Span
            for o_span in opinion_values:
                if t_span.left < o_span.left:
                    self.update_ranges(matrix, ranges, t_span, o_span)
                else:
                    self.update_ranges(matrix, ranges, o_span, t_span)
        return ranges

    def update_ranges(self, matrix: tf.Tensor, ranges: List, fist_span: Span, second_span: Span) -> None:
        sub_matrix: tf.Tensor = matrix[fist_span.left:fist_span.right + 1, second_span.left:second_span.right + 1]
        relation: int = self.get_relation(sub_matrix)
        if relation != GTSMatrixID.OTHER.value and relation != GTSMatrixID.NOT_RELEVANT.value:
            ranges.append(Span(fist_span.left, fist_span.right, second_span.left, second_span.right, relation))

    @staticmethod
    def get_relation(matrix: tf.Tensor) -> int:
        if config['task']['type'] == 'pair':
            if bool(tf.where(matrix == GTSMatrixID.PAIR, 1, 0)):
                return GTSMatrixID.PAIR.value
            else:
                return GTSMatrixID.OTHER.value
        else:
            values: tf.Tensor
            count: tf.Tensor
            values, _, count = tf.unique_with_counts(tf.reshape(matrix, shape=[-1]))
            count = tf.where(values == GTSMatrixID.NOT_RELEVANT.value, 0, count)
            sorted_idx: tf.Tensor = tf.argsort(values, direction='DESCENDING')
            count = tf.gather(count, sorted_idx)
            values = tf.gather(values, sorted_idx)

            return int(values[tf.math.argmax(count)])

    def result(self):
        return {
            'Opinion score': self.scores(self.true_opinion_spans, self.pred_opinion_spans),
            'Target score': self.scores(self.true_target_spans, self.pred_target_spans),
            'Union score': self.scores(self.true_union_spans, self.pred_union_spans)
        }

    def scores(self, true_span: List[Span], pred_span: List[Span]) -> Dict:
        correct_num: int = len(set(true_span) & set(pred_span))
        precision: float = self.safe_division(correct_num, len(pred_span))
        recall: float = self.safe_division(correct_num, len(true_span))
        f1: float = 2 * precision * self.safe_division(recall, (precision + recall))
        return {'precision': precision,
                'recall': recall,
                'f1': f1}

    @staticmethod
    def safe_division(numerator: float, denominator: float) -> float:
        if denominator == 0.0:
            return 0.0
        else:
            return numerator / denominator
