import torch
from pathlib import Path
import torchsnapshot
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, LazyMemmapStorage, ListStorage, TensorDictPrioritizedReplayBuffer
path = Path('/home/nono/Documents/Dassault/Water-Bomber-Env/results/smac-medium-buffer_2023-09-26_12:42:35')
rb_path = str(path / 'rb' / '0')

buffer_size = 1000000
batch_size = 10000

rb_storage = LazyTensorStorage(buffer_size, device='cpu')
replay_buffer = TensorDictReplayBuffer(
            storage=rb_storage,
            batch_size=batch_size,
        )

data = TensorDict(
    {
        "a": torch.arange(512).view(128, 4),
        ("b", "c"): torch.arange(1024).view(128, 8),
    },
    batch_size=[128],
)

print("loading buffer from", rb_path)
snapshot = torchsnapshot.Snapshot(path=rb_path)
target_state = {
    "state": replay_buffer
}
snapshot.restore(app_state=target_state)

print(replay_buffer)
print(replay_buffer[:])
print(len(replay_buffer))

rb_storage = LazyTensorStorage(buffer_size, device='cpu')
replay_buffer = TensorDictReplayBuffer(
            storage=rb_storage,
            batch_size=batch_size,
        )
small_buffer_size = 1000000
buffer_size = 20_000_000


list_rbs_path = [x for x in (path/'rb').iterdir() if x.is_dir()]
for rb_path in list_rbs_path:
  small_rb_storage = LazyTensorStorage(small_buffer_size, device='cpu')
  small_replay_buffer = TensorDictReplayBuffer(
              storage=rb_storage,
              batch_size=batch_size,
          )
  len(replay_buffer)