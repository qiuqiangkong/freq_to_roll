from torch import Tensor
import torch
import matplotlib.pyplot as plt


def pitch_to_roll(
	pitch: Tensor, 
	intervals_per_semitone: int = 1,
	halfwidth: float = 1.
) -> Tensor:
	r"""Convert 1D pitch array to 2D roll.

	Args:
		pitch: (t,)
		intervals_per_semitone: int
		halfwidth: float

	Returns:
		roll: (t, f)
	"""

	T = pitch.shape[0]
	F = 128 * intervals_per_semitone
	x = torch.arange(F)[None, :].repeat(T, 1)  # (t, f)

	a = 1 / halfwidth  # scalar
	b = 1 - (pitch * intervals_per_semitone) / halfwidth  # (t,)
	roll = a * x + b[:, None]  # (t, f)

	indices = torch.nonzero(roll > 1, as_tuple=True)
	roll[indices[0], indices[1]] = 2 - roll[indices[0], indices[1]]
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
	freq = torch.arange(200, 3000, 20)  # (t,) 20 Hz - 3000 Hz
	pitch = freq_to_pitch(freq)  # (t,)
	roll = pitch_to_roll(pitch, intervals_per_semitone=1, halfwidth=1.)  # (t, f)
	print(roll.shape)
	
	# Visualization
	fig, axes = plt.subplots(2, 1, sharex=True)
	axes[0].plot(freq)
	axes[1].matshow(roll.T, origin='lower', aspect='auto', cmap='jet')
	axes[0].set_ylabel("Freq (Hz)")
	axes[1].set_ylabel("Pitch bins")
	axes[1].set_xlabel("Time (frames)")
	plt.savefig("out.pdf")
	print("Write out to out.pdf")