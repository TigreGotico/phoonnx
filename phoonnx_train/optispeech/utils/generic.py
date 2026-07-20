import io
from math import ceil

import numpy as np
import torch

from phoonnx_train.optispeech.utils import pylogger

log = pylogger.get_pylogger(__name__)


def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def plot_tensor(tensor):
    """Render a 2D tensor to an RGB numpy image for TensorBoard logging.

    matplotlib is imported lazily so the training package stays importable in
    environments without a plotting stack (e.g. CI); it is only needed when
    validation actually logs spectrogram images.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def get_phoneme_durations(durations, phones):
    prev = durations[0]
    merged_durations = []
    # Convolve with stride 2
    for i in range(1, len(durations), 2):
        if i == len(durations) - 2:
            # if it is last take full value
            next_half = durations[i + 1]
        else:
            next_half = ceil(durations[i + 1] / 2)

        curr = prev + durations[i] + next_half
        prev = durations[i + 1] - next_half
        merged_durations.append(curr)

    assert len(phones) == len(merged_durations)
    assert len(merged_durations) == (len(durations) - 1) // 2

    merged_durations = torch.cumsum(torch.tensor(merged_durations), 0, dtype=torch.long)
    start = torch.tensor(0)
    duration_json = []
    for i, duration in enumerate(merged_durations):
        duration_json.append(
            {
                phones[i]: {
                    "starttime": start.item(),
                    "endtime": duration.item(),
                    "duration": duration.item() - start.item(),
                }
            }
        )
        start = duration

    assert list(duration_json[-1].values())[0]["endtime"] == sum(
        durations
    ), f"{list(duration_json[-1].values())[0]['endtime'], sum(durations)}"
    return duration_json


def get_model_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    num_bytes = len(buf.getbuffer())
    return num_bytes // 1e6


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
