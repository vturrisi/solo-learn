import logging
import math

import numpy as np
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


class TempDALIGenericIterator(DALIGenericIterator):
    """Temporary fix to avoid epoch-skiping when setting last_batch_policy=Drop."""

    def _advance_and_check_drop_last(self, dry_run=False, end_iteration=True):
        """
        Checks whether the current batch is not fully filled and whether it should be dropped.

        It could be dry run without changing the iterator state and not raising StopIteration
        """
        # check if for given initial count in any GPU with the current value of the samples read
        # if we read one more batch would we overflow
        counter = self._counter
        should_end = False
        if self._reader_name:
            counter += self.batch_size
            if self._last_batch_policy == LastBatchPolicy.DROP:
                should_end = np.any(self._counter_per_gpu + counter > self._shard_sizes_per_gpu)
        else:
            counter += self._num_gpus * self.batch_size
            if self._last_batch_policy == LastBatchPolicy.DROP:
                should_end = counter > self._size

        if not dry_run:
            self._counter = counter
            if should_end and end_iteration:
                self._end_iteration()

        return should_end

    def reset(self):
        """
        Resets the iterator after the full epoch.
        DALI iterators do not support resetting before the end of the epoch
        and will ignore such request.
        """
        # in the case of the DROP policy the user who runs DALI, based on the iterator length,
        # can assume there is no more data in the pipeline where there still is the last,
        # incomplete batch, we need to extract from the pipeline and drop before rising
        # StopIteration indicating the pipeline is depleted. Here we first check if that
        # is the case, and if so we run the pipeline and drop the last batch
        if self._last_batch_policy == LastBatchPolicy.DROP and not ():
            should_end = self._advance_and_check_drop_last(dry_run=True, end_iteration=False)
            already_ended = self._size > 0 and self._counter >= self._size
            if should_end and not already_ended:
                self._get_outputs()
                self._schedule_runs()
                self._advance_and_check_drop_last(end_iteration=False)

        if self._counter >= self._size or self._size < 0:
            if self._last_batch_policy == LastBatchPolicy.FILL and not self._last_batch_padded:
                if self._reader_name:
                    # accurate way
                    # get the number of samples read in this epoch by each GPU
                    # self._counter had initial value of min(self._counter_per_gpu) so subtract
                    # this to get the actual value
                    self._counter -= min(self._counter_per_gpu)
                    self._counter_per_gpu = self._counter_per_gpu + self._counter
                    # check how much each GPU read ahead from next shard, as shards have different
                    # size each epoch GPU may read ahead or not
                    self._counter_per_gpu = self._counter_per_gpu - self._shard_sizes_per_gpu
                    # to make sure that in the next epoch we read the whole shard we need
                    # to set start value to the smallest one
                    self._counter = min(self._counter_per_gpu)
                else:
                    # legacy way
                    self._counter = self._counter % self._size
            else:
                self._counter = 0
            # advance to the next shard
            if self._reader_name:
                if not self._is_stick_to_shard:
                    # move shards id for wrapped pipelines
                    self._shards_id = (self._shards_id + 1) % self._shards_num
                # revaluate _size
                if self._last_batch_policy == LastBatchPolicy.FILL and not self._last_batch_padded:
                    # move all shards ids GPU ahead
                    if not self._is_stick_to_shard:
                        self._shard_sizes_per_gpu = np.roll(self._shard_sizes_per_gpu, 1)
                    # check how many samples we need to reach from each shard in next epoch
                    # per each GPU taking into account already read
                    read_in_next_epoch = self._shard_sizes_per_gpu - self._counter_per_gpu
                    # get the maximum number of samples and round it up to full batch sizes
                    self._size = (
                        math.ceil(max(read_in_next_epoch) / self.batch_size) * self.batch_size
                    )
                    # in case some epoch is skipped because we have read ahead in this epoch so
                    # much that in the next one we done already
                    if self._size == 0:
                        # it means that self._shard_sizes_per_gpu == self._counter_per_gpu,
                        # so we can jump to the next epoch and zero self._counter_per_gpu
                        self._counter_per_gpu = np.zeros(self._shards_num, dtype=np.int64)
                        # self._counter = min(self._counter_per_gpu), but just set 0
                        # to make it simpler
                        self._counter = 0
                        # roll once again
                        self._shard_sizes_per_gpu = np.roll(self._shard_sizes_per_gpu, 1)
                        # as self._counter_per_gpu is 0 we can just use
                        # read_in_next_epoch = self._shard_sizes_per_gpu
                        self._size = (
                            math.ceil(max(self._shard_sizes_per_gpu) / self.batch_size)
                            * self.batch_size
                        )

            for p in self._pipes:
                p.reset()
                if p.empty():
                    with p._check_api_type_scope(types.PipelineAPIType.ITERATOR):
                        p.schedule_run()
        else:
            logging.warning(
                "DALI iterator does not support resetting while epoch is not finished. \
                             Ignoring..."
            )
