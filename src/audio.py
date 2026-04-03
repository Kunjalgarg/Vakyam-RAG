import uuid
import time
from gtts import gTTS
import pygame
import os


class Audio:

    def __init__(self):
        pygame.mixer.init()

    def speak(self, text, lang="hi"):
        if not text.strip():
            return

        os.makedirs("outputs/audio", exist_ok=True)

        file = f"outputs/audio/{uuid.uuid4().hex}.mp3"

        gTTS(text=text, lang=lang).save(file)

        pygame.mixer.music.load(file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)