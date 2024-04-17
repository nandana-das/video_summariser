from moviepy.editor import *

# Replace video.flv with the path to your video file
video = VideoFileClip(r"video4.mp4")

# Extract audio from the video
audio = video.audio

# Save the audio to a WAV file
audio.write_audiofile(r"audio4.wav")