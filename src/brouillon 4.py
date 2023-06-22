
import torch

from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
from tensordict import TensorDict

torch.manual_seed(0)

rb = TensorDictPrioritizedReplayBuffer(alpha=0.7, beta=1.1, storage=LazyTensorStorage(10), batch_size=5)
data = TensorDict({"a": torch.ones(10, 3), ("b", "c"): torch.zeros(10, 3, 1)}, [10])
rb.extend(data)
print("len of rb", len(rb))
sample = rb.sample(5)
print(sample)

print("index", sample["index"])