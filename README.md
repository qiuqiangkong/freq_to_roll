# Pitch to Rolls

This repo converts frequency/pitch array to a 2D roll.

## Usage

```python
freq = torch.arange(200, 3000, 20)  # (t,) 20 Hz - 3000 Hz
pitch = freq_to_pitch(freq)  # (t,)
roll = pitch_to_roll(pitch, intervals_per_semitone=1, halfwidth=1.)  # (t, f)
```

## Visualization

<img src="" width="800">
