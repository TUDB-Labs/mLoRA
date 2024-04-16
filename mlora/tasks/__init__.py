from .common import (
    BasicMetric,
    AutoMetric,
    BasicTask,
    CasualTask,
    SequenceClassificationTask,
    CommonSenseTask,
    MultiTask,
    task_dict,
)
from .qa_tasks import QuestionAnswerTask
from . import glue_tasks, qa_tasks

glue_tasks.update_task_dict(task_dict)
qa_tasks.update_task_dict(task_dict)


__all__ = [
    "BasicMetric",
    "AutoMetric",
    "BasicTask",
    "CasualTask",
    "SequenceClassificationTask",
    "CommonSenseTask",
    "QuestionAnswerTask",
    "MultiTask",
    "task_dict"
]
