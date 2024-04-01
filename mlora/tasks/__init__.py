from mlora.tasks.evaluator import EvaluateConfig, evaluate
from mlora.tasks.common import BasicMetric, AutoMetric
from mlora.tasks.common import BasicTask, CasualTask, SequenceClassificationTask, CommonSenseTask, MultiTask
from mlora.tasks.common import task_dict
from mlora.tasks import glue_tasks, qa_tasks

glue_tasks.update_task_dict(task_dict)
qa_tasks.update_task_dict(task_dict)


__all__ = [
    "EvaluateConfig",
    "evaluate",
    "BasicMetric",
    "AutoMetric",
    "BasicTask",
    "CasualTask",
    "SequenceClassificationTask",
    "CommonSenseTask",
    "MultiTask",
    "task_dict"
]
