from . import glue_tasks, qa_tasks
from .common import (
    AutoMetric,
    BasicMetric,
    BasicTask,
    CasualTask,
    CommonSenseTask,
    MultiTask,
    SequenceClassificationTask,
    task_dict,
)
from .qa_tasks import QuestionAnswerTask

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
    "task_dict",
]
