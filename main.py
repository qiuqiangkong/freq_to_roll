import matplotlib.pyplot as plt
import torch
from torch import Tensor


def pitch_to_roll(
    pitch: Tensor, 
    bins_per_semitone: int = 1,
    smooth_bins: float = 1.
) -> Tensor:
    r"""Convert 1D pitch array to 2D roll.

    Args:
        pitch: (p.shape, t), e.g., (t,) | (b, t)
        intervals_per_semitone: int
        halfwidth: float

    Returns:
        roll: (p.shape, f)
    """

    # Reserve space
    F = 128 * bins_per_semitone
    x = torch.zeros(pitch.shape + (F,))  # (p.shape, f)
    x[..., :] = torch.arange(F)  # (p.shape, f)

    # y = ax + b
    a = 1 / smooth_bins  # scalar
    b = 1 - (pitch * bins_per_semitone) / smooth_bins  # (p.shape,)
    roll = a * x + b[..., None]  # (p.shape, f)

    # Assign values to roll
    indices = torch.nonzero(roll > 1, as_tuple=True)
    roll[indices] = 2 - roll[indices]
    roll = torch.clamp(roll, 0, 1)

    return roll


def freq_to_pitch(freq: Tensor) -> Tensor:
    pitch = 12 * torch.log2(freq / 440.) + 60.
    return pitch


def pitch_to_freq(pitch: Tensor) -> Tensor:
    freq = 440. * (2 ** ((pitch - 60.) / 12))
    return freq


if __name__ == '__main__':

    # Example
    freq = torch.Tensor([430 * 2**(i/12.) for i in range(-12, 13)])  # 215 - 860 Hz, (t,)
    roll = pitch_to_roll(freq_to_pitch(freq))  # (t, f)
    print(roll.shape)
    
    # Visualization
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(freq)
    axes[1].matshow(roll.T, origin='lower', aspect='auto', cmap='jet')
    axes[0].set_title("Freq array")
    axes[0].set_ylabel("Freq (Hz)")
    axes[1].set_title("Roll")
    axes[1].set_ylabel("Pitch bins")
    axes[1].set_xlabel("Time (frames)")
    axes[1].xaxis.set_ticks_position("bottom")
    plt.tight_layout()
    plt.savefig("out.pdf")
    print("Write out to out.pdf")