# Pitch to Rolls

This repo converts frequency/pitch array to a 2D roll.

## Usage

```python
import torch
from main import freq_to_pitch, pitch_to_roll

# Convert an array
freq = torch.Tensor([430, 450, 470, 490])  # Hz, (t,)
roll = pitch_to_roll(freq_to_pitch(freq))  # (t, f)
print(roll.shape)

# Convert in batch
freq = torch.Tensor([[430, 450, 470, 490]])  # Hz, (b, t,)
roll = pitch_to_roll(freq_to_pitch(freq))  # (b, t, f)
```

Run more examples by:

```python
python main.py
```

## Visualization

<img src="https://github.com/user-attachments/assets/25d7e22f-4259-41dc-9ea1-f39b69cf58ca" width="800">
