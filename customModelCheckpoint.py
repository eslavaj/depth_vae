from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import os
import shutil
from typing import Any, Dict, Optional, Set
from torch import Tensor


class CustomModelCheckpoint(ModelCheckpoint):
    def _save_last_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: Dict[str, Tensor]) -> None:
        if not self.save_last:
            return

        filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST)

        if self._enable_version_counter:
            version_cnt = self.STARTING_VERSION
            while self.file_exists(filepath, trainer) and filepath != self.last_model_path:
                filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST, ver=version_cnt)
                version_cnt += 1

        # set the last model path before saving because it will be part of the state.
        previous, self.last_model_path = self.last_model_path, filepath
        
        # if self._fs.protocol == "file" and self._last_checkpoint_saved and self.save_top_k != 0:
        #     self._link_checkpoint(trainer, self._last_checkpoint_saved, filepath)
        # else:
        #     self._save_checkpoint(trainer, filepath)

        self._save_checkpoint(trainer, filepath)
        if previous and self._should_remove_checkpoint(trainer, previous, filepath):
            self._remove_checkpoint(trainer, previous)