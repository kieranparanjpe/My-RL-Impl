from abc import ABC, abstractmethod


class BaseRecorder(ABC):

    def __init__(self, **kwargs):
        self.new_episode = True
        self.path = ""
        self.enabled = False
        pass

    @abstractmethod
    def should_record(self, step_id):
        pass

class Recorder(BaseRecorder):

    def __init__(self, path: str, number_videos: int, total_timesteps: int):
        super().__init__()
        self.path = path
        self.number_videos = number_videos
        self.recorded_videos = 0
        self.total_timesteps = total_timesteps
        self.enabled = True


    def should_record(self, step_id):
        threshold = self.recorded_videos * (self.total_timesteps // (self.number_videos - 1)) - 5000

        if step_id > threshold and self.new_episode:
            self.recorded_videos += 1
            return True
        return False

class NullRecorder(BaseRecorder):

    def __init__(self):
        super().__init__()
        self.enabled = False

    def should_record(self, step_id):
        return False