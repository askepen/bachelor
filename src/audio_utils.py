import IPython.display
import numpy as np
import json
import torch

from IPython.display import display
from matplotlib import pyplot as plt


def Audio(audio: np.ndarray, rate: int):
  """
  Use instead of IPython.display.Audio as a workaround for VS Code.
  `audio` is an array with shape (channels, samples) or just (samples,) for mono.
  """
  
  if np.ndim(audio) == 1:
    channels = [audio.tolist()]
  else:
    channels = audio.tolist()
  
  return IPython.display.HTML("""
    <script>
      if (!window.audioContext) {
        window.audioContext = new AudioContext();
        window.playAudio = function(audioChannels, rate) {
          const buffer = audioContext.createBuffer(audioChannels.length, audioChannels[0].length, rate);
          for (let [channel, data] of audioChannels.entries()) {
            buffer.copyToChannel(Float32Array.from(data), channel);
          }

          const source = audioContext.createBufferSource();
          source.buffer = buffer;
          source.connect(audioContext.destination);
          source.start();
        }
      }
    </script>
    <button onclick="playAudio(%s, %s)">Play</button>
  """ % (json.dumps(channels), rate))

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
  plt.show(block=False)


def play_audio(waveform, sample_rate):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  if num_channels == 1:
    display(Audio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    display(Audio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=False)

