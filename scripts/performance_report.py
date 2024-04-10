import argparse
import sqlite3
import logging
import csv

from typing import List, Dict, Tuple

# Command Line Arguments
parser = argparse.ArgumentParser(description='Performance report.')
parser.add_argument('--db', type=str, required=True, help='NSys sqlite file.')
parser.add_argument('--output', type=str, required=True,
                    help='Export csv file.')

args = parser.parse_args()

logging.basicConfig(format="[%(asctime)s] Performance: %(message)s",
                    level="INFO",
                    handlers=[logging.StreamHandler()],
                    force=True)

G_CREATE_TEMP_TABLE = """
CREATE TEMPORARY TABLE IF NOT EXISTS TEMP_KERN_INFOS AS
SELECT R.start AS API_START, R.end AS API_END,
       K.start AS KERN_START, K.end AS KERN_END,
       R.start AS T_START,
       MAX(R.end, K.end) AS T_END,
       KNAME.value AS KERN_NAME
    FROM
        CUPTI_ACTIVITY_KIND_KERNEL AS K
    JOIN
        CUPTI_ACTIVITY_KIND_RUNTIME AS R
        ON K.correlationId == R.correlationId
    LEFT JOIN
        StringIds AS KNAME
        ON KNAME.id == K.demangledName;
"""

G_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS TEMPINDEX ON TEMP_KERN_INFOS(T_START);
"""

G_SELECT_NVTX_KERN = """
SELECT
    E_NAME AS ENVET_NAME,
    API_START AS KERN_API_START,
    API_END AS KERN_API_END,
    KERN_START AS KERN_START,
    KERN_END AS KERN_END,
    KERN_NAME AS KERN_NAME
FROM
    (SELECT start AS E_START, end AS E_END, text AS E_NAME FROM NVTX_EVENTS)
LEFT JOIN
    TEMP_KERN_INFOS
ON
    E_START <= T_START AND E_END >= T_START;
"""

G_SELECT_NVTX = """
SELECT start AS E_START, end AS E_END, text AS E_NAME FROM NVTX_EVENTS;
"""


class KernInfo:
    name_: str = ""
    api_start_: int = 0
    api_end_: int = 0
    kern_start_: int = 0
    kern_end_: int = 0

    api_time_: int = 0
    kern_time_: int = 0
    queue_time_: int = 0

    def __init__(self, name: str,
                 api_start: int, api_end: int,
                 kern_start: int, kern_end: int):
        self.name_ = name
        self.api_start_ = api_start
        self.api_end_ = api_end
        self.kern_start_ = kern_start
        self.kern_end_ = kern_end

        self.queue_time_ = 0 if api_end > kern_start else kern_start - api_end
        self.api_time_ = api_end - api_start
        self.kern_time_ = kern_end - kern_start

        assert self.queue_time_ >= 0


class KernSummary:
    name_: str = ""
    api_time_: int = 0
    queue_time_: int = 0
    kern_time_: int = 0

    kern_cnt_: int = 0

    def __init__(self, name: str,
                 api_time: int, queue_time: int, kern_time: int,
                 kern_cnt: int) -> None:
        self.name_ = name
        self.api_time_ = api_time
        self.queue_time_ = queue_time
        self.kern_time_ = kern_time
        self.kern_cnt_ = kern_cnt


class EventInfo:
    name_: str = ""
    cnt_: int = 0
    ttime_: int = 0
    # all kernels
    kerns_: Dict[str, List[KernInfo]] = {}

    def __init__(self, name: str):
        self.name_ = name
        self.cnt_ = 0
        self.kerns_ = {}

    def add_kern(self, name: str,
                 api_start: int, api_end: int,
                 kern_start: int, kern_end: int):
        if api_start is None or api_end is None:
            return
        assert kern_start is not None
        assert kern_end is not None
        if name not in self.kerns_:
            self.kerns_[name] = []
        self.kerns_[name].append(
            KernInfo(name, api_start, api_end, kern_start, kern_end))

    def sum(self) -> Dict[str, KernSummary]:
        summary_ret: Dict[str, KernSummary] = {}

        def sum_kern_list(kerns: List[KernInfo]) -> Tuple[int, int, int]:
            t_api_time: int = 0
            t_kern_time: int = 0
            t_queue_time: int = 0
            for kern_item in kerns:
                t_api_time += kern_item.api_time_
                t_kern_time += kern_item.kern_time_
                t_queue_time += kern_item.queue_time_
            return t_api_time, t_queue_time, t_kern_time

        for kern_name, kern_info in self.kerns_.items():
            t_api_time, t_queue_time, t_kern_time = sum_kern_list(kern_info)
            summary_ret[kern_name] = KernSummary(
                kern_name, t_api_time, t_queue_time, t_kern_time, len(kern_info))

        return summary_ret


class EventSum:
    name_: str = ""
    cnt_: int = 0

    api_time_: int = 0
    queue_time_: int = 0
    kern_time_: int = 0

    ttime_: int = 0

    def __init__(self, event: EventInfo) -> None:
        self.name_ = event.name_
        self.cnt_ = event.cnt_

        self.ttime_ = event.ttime_

        self.__summary(event)

    def __summary(self, event: EventInfo):
        t_api_time: int = 0
        t_queue_time: int = 0
        t_kern_name: int = 0

        for _, kern_sum in event.sum().items():
            t_api_time += kern_sum.api_time_
            t_queue_time += kern_sum.queue_time_
            t_kern_name += kern_sum.kern_time_

        self.api_time_ = t_api_time
        self.queue_time_ = t_queue_time
        self.kern_time_ = t_kern_name

    def avg(self) -> Tuple[int, int, int]:
        aapi = round(self.api_time_ / self.cnt_)
        aqueue = round(self.queue_time_ / self.cnt_)
        akern = round(self.kern_time_ / self.cnt_)
        return aapi, aqueue, akern

    def sum(self) -> Tuple[int, int, int, int]:
        return self.ttime_, self.api_time_, self.queue_time_, self.kern_time_, self.cnt_

    def __str__(self) -> str:
        api, queue, kern = self.avg()
        return f"{api} {queue} {kern}"


if __name__ == "__main__":
    conn = sqlite3.connect(args.db)

    logging.info("To Init the sqlite file.")

    conn.execute(G_CREATE_TEMP_TABLE)
    conn.execute(G_CREATE_INDEX)

    logging.info("Init the sqlite file done.")

    events: Dict[str, EventInfo] = {}

    logging.info("To count the NVTX event.")
    for row in conn.execute(G_SELECT_NVTX):
        start_time, end_time, event_name = row

        if event_name not in events:
            events[event_name] = EventInfo(event_name)
        event_item = events[event_name]
        event_item.cnt_ += 1
        event_item.ttime_ += (end_time - start_time)
    logging.info("Count the NVTX event done.")

    logging.info("To get kernel info.")
    for row in conn.execute(G_SELECT_NVTX_KERN):
        event_name, api_start, api_end, kern_start, kern_end, kern_name = row
        event_item = events[event_name]
        event_item.add_kern(kern_name, api_start,
                            api_end, kern_start, kern_end)
    logging.info("Get kernel info done.")

    with open(args.output, "w") as csv_f:
        writer = csv.writer(csv_f)
        for event_name, event_item in events.items():
            event_sum = EventSum(event_item).sum()
            if event_sum[0] == 0:
                continue
            event_sum = (event_name,) + event_sum
            writer.writerow(event_sum)
