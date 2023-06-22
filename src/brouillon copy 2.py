
from torchrl.data import TensorDictReplayBuffer
from tensordict import tensorclass, TensorDict
from torchrl.data.replay_buffers.samplers import RandomSampler, PrioritizedSampler, SamplerWithoutReplacement
from torchrl.data import LazyTensorStorage, LazyMemmapStorage, ListStorage
import torch
# We define the maximum size of the buffer
size = 30

petit_point = TensorDict(
    {
        "a": torch.arange(120).view(-1),
        ("b", "c"): torch.arange(150).view(-1),
        #"td_error":0.0,
        #'index':0,
    },
    batch_size=[],
)
"""
point = TensorDict(
    {
        "a": torch.arange(120).view(1, -1),
        ("b", "c"): torch.arange(150).view(1, -1),
        "td_error":[0.0],
        'index':[0],
    },
    batch_size=[1],
)

data = TensorDict(
    {
        "a": torch.arange(120).view(30, 4),
        ("b", "c"): torch.arange(150).view(30, 5),
    },
    batch_size=[30],
)"""
storage=LazyMemmapStorage(size)

rb = TensorDictReplayBuffer(
    storage=storage,
    sampler=PrioritizedSampler(size, alpha=0.8, beta=1.1),
    priority_key="td_error",
    batch_size=1024,
)

#data["td_error"] = torch.arange(data.numel())
#rb.update_tensordict_priority(data)
#rb.extend(data)
for i in range(30):
    #point['td_error'] = [i]
    #petit_point['td_error'] = float(i)
    print(petit_point)
    #rb.extend(point)
    rb.add(petit_point)


sample = rb.sample()
from matplotlib import pyplot as plt

#
plt.hist(sample["index"].numpy())
plt.show()

sample = rb.sample()
print('index', sample["index"])
sample["td_error"] = 30 - sample["index"]
rb.update_tensordict_priority(sample)


sample = rb.sample()
plt.hist(sample["index"].numpy())
plt.show()

rb2 = TensorDictReplayBuffer(
    storage=storage,
    sampler=SamplerWithoutReplacement(),
    batch_size=1,
)

for i, x in enumerate(rb2):
    print(i, x['index'])