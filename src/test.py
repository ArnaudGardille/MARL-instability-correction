#from tensorboardX import SummaryWriter

from torch.utils.tensorboard import SummaryWriter
import numpy as np
writer = SummaryWriter()
for i in range(10):
    values = np.random.random(1000) + i
    counts, limits= np.histogram(values, bins=10, range=(0.0, 10.0))
    #l = [a for nb, a in zip(nbs, inter) for _ in range(nb) ]
    #print(l)
    #writer.add_histogram('distribution centers',l, i, bins=11, max_bins=11)


    #values = np.array(dummy_data).astype(float).reshape(-1)
    #counts, limits = np.histogram(values, bins=bins)
    sum_sq = values.dot(values)
    writer.add_histogram_raw(
        tag='histogram_with_raw_data',
        min=values.min(),
        max=values.max(),
        num=len(values),
        sum=values.sum(),
        sum_squares=sum_sq,
        bucket_limits=limits[1:].tolist(),
        bucket_counts=counts.tolist(),
        global_step=i)
writer.close()
#%%
"""
from torch.utils.tensorboard import SummaryWriter
import numpy as np
writer = SummaryWriter()
dummy_data = []
for idx, value in enumerate(range(50)):
    dummy_data += [idx + 0.001] * value

bins = list(range(50+2))
bins = np.array(bins)
values = np.array(dummy_data).astype(float).reshape(-1)
counts, limits = np.histogram(values, bins=bins)
sum_sq = values.dot(values)
writer.add_histogram_raw(
    tag='histogram_with_raw_data',
    min=values.min(),
    max=values.max(),
    num=len(values),
    sum=values.sum(),
    sum_squares=sum_sq,
    bucket_limits=limits[1:].tolist(),
    bucket_counts=counts.tolist(),
    global_step=0)
writer.close()
"""