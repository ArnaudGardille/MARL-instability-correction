from torchrl.data import TensorDictReplayBuffer
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, ListStorage
from tensordict import tensorclass, TensorDict
import torch
# We define the maximum size of the buffer
size = 10_000

data = TensorDict(
    {
        "a": (torch.arange(12)+1).view(3, 4),
        ("b", "c"): torch.arange(15).view(3, 5),
    },
    batch_size=[3],
)
print(data)

rb_storage = LazyTensorStorage(size)

buffer_lazymemmap = TensorDictReplayBuffer(
        #self.replay_buffer = TensorDictReplayBuffer(
        #storage=ListStorage(self.buffer_size),
        storage=rb_storage,
        #collate_fn=lambda x: x, 
        #priority_key="td_error",
        batch_size=32,
    )
"""buffer_lazymemmap = TensorDictReplayBuffer(
    storage=LazyMemmapStorage(size, scratch_dir="/tmp/memmap/"), batch_size=12
)"""
buffer_lazymemmap.extend(data)
print(f"The buffer has {len(buffer_lazymemmap)} elements")
sample = buffer_lazymemmap.sample()
print("sample:", sample['index'])
indices = buffer_lazymemmap.extend(data)
print(indices)










