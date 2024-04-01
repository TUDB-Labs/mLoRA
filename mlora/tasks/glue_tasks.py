from mlora.tasks.common import SequenceClassificationTask
import torch


def update_task_dict(task_dict):
    task_dict.update({
        "glue:cola": SequenceClassificationTask(
            task_name="glue:cola",
            task_type="single_label_classification",
            num_labels=2,
            label_dtype=torch.long,
            dataload_function=lambda data_point: (
                [data_point["sentence"]],
                [int(data_point["label"])],
                {"bos": True, "eos": True}
            ),
        ),
        "glue:mnli": SequenceClassificationTask(
            task_name="glue:mnli",
            task_type="single_label_classification",
            num_labels=3,
            label_dtype=torch.long,
            dataload_function=lambda data_point: (
                [data_point["premise"], data_point["hypothesis"]],
                [int(data_point["label"])],
                {"bos": True, "eos": True},
            ),
        ),
        "glue:mrpc": SequenceClassificationTask(
            task_name="glue:mrpc",
            task_type="single_label_classification",
            num_labels=2,
            label_dtype=torch.long,
            dataload_function=lambda data_point: (
                [data_point["sentence1"], data_point["sentence2"]],
                [int(data_point["label"])],
                {"bos": True, "eos": True},
            ),
        ),
        "glue:qnli": SequenceClassificationTask(
            task_name="glue:qnli",
            task_type="single_label_classification",
            num_labels=2,
            label_dtype=torch.long,
            dataload_function=lambda data_point: (
                [data_point["question"], data_point["sentence"]],
                [int(data_point["label"])],
                {"bos": True, "eos": True},
            ),
        ),
        "glue:qqp": SequenceClassificationTask(
            task_name="glue:qqp",
            task_type="single_label_classification",
            num_labels=2,
            label_dtype=torch.long,
            dataload_function=lambda data_point: (
                [data_point["question1"], data_point["question2"]],
                [int(data_point["label"])],
                {"bos": True, "eos": True},
            ),
        ),
        "glue:rte": SequenceClassificationTask(
            task_name="glue:rte",
            task_type="single_label_classification",
            num_labels=2,
            label_dtype=torch.long,
            dataload_function=lambda data_point: (
                [data_point["sentence1"], data_point["sentence2"]],
                [int(data_point["label"])],
                {"bos": True, "eos": True},
            ),
        ),
        "glue:sst2": SequenceClassificationTask(
            task_name="glue:sst2",
            task_type="single_label_classification",
            num_labels=2,
            label_dtype=torch.long,
            dataload_function=lambda data_point: (
                [data_point["sentence"]],
                [int(data_point["label"])],
                {"bos": True, "eos": True},
            ),
        ),
        "glue:wnli": SequenceClassificationTask(
            task_name="glue:wnli",
            task_type="single_label_classification",
            num_labels=2,
            label_dtype=torch.long,
            dataload_function=lambda data_point: (
                [data_point["sentence1"] + " </s> " + data_point["sentence2"]],
                [int(data_point["label"])],
                {"bos": True, "eos": True},
            ),
        ),
    })
