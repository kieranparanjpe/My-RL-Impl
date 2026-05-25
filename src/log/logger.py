from __future__ import annotations

import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict
from .recorder import BaseRecorder, Recorder

import wandb

class Logger(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def finish(self):
        pass

    @abstractmethod
    def upload_videos(self, recorder: Recorder):
        pass

    @abstractmethod
    def add_elements(self, elements: Dict[str, Any]):
        pass

    @abstractmethod
    def reset(self, *fields: str):
        pass

    @abstractmethod
    def set_log_data(self, kvps: Dict[str, Any]):
        pass

    @abstractmethod
    def sum_log_data(self, kvps: Dict[str, Any]):
        pass

    @abstractmethod
    def log_data(self, *fields):
        pass

class WandBLogger(Logger):
    def __init__(self, run_name : str, environment_id: str, algorithm_id: str, policy_id: str,
                 hyperparameters: Dict[str, Any], elements: Dict[str, Any]):
        super().__init__()
        self.run = self.wandb_run = wandb.init(
            entity="kieranparanjpe-mcgill-university",
            project="RL_Project1",
            name=run_name,
            tags=[f"{algorithm_id}", f"{policy_id}", f"{environment_id}"],
            job_type="train",
            config=hyperparameters,
        )

        self.elements_start = elements
        self.elements = deepcopy(self.elements_start)

        self.run.define_metric("*", step_metric="global_step")

    def finish(self):
        self.run.finish()

    def upload_videos(self, recorder : BaseRecorder):
        if not recorder.enabled:
            return

        videos = [f"{recorder.path}/{video}" for video in os.listdir(recorder.path) if ".mp4" in video]
        for video in videos:
            step = int(video.split('-step-')[-1].split(".mp4")[0])

            self.run.log({
                "video/recording": wandb.Video(video, fps=30, format="mp4"),
                "global_step": step
            })


    def add_elements(self, elements : Dict[str, Any]):
        self.elements.update(deepcopy(elements))
        self.elements_start.update(elements)

    def reset(self, *fields : str):
        if fields is None or len(fields) == 0:
            self.elements = deepcopy(self.elements_start)
        else:
            for field in fields:
                self.elements[field] = self.elements_start[field]

    def set_log_data(self, kvps : Dict[str, Any]):
        self.elements.update(kvps)

    def sum_log_data(self, kvps : Dict[str, Any]):
        for k, v in kvps.items():
            self.elements[k] += v

    def log_data(self, *fields):
        if fields is None or len(fields) == 0:
            self.run.log(data=self.elements)
        else:
            data = {k : v for k, v in self.elements.items() if k in fields}
            self.run.log(data=data)

class NullLogger(Logger):

    def finish(self):
        pass

    def upload_videos(self, recorder: Recorder):
        pass

    def add_elements(self, elements: Dict[str, Any]):
        pass

    def reset(self, *fields: str):
        pass

    def set_log_data(self, kvps: Dict[str, Any]):
        pass

    def sum_log_data(self, kvps: Dict[str, Any]):
        pass

    def log_data(self, *fields):
        pass