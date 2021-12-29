#STEP 1: install these packages
#pip install gTTS
#pip install playsound

from gtts import gTTS
import os
from playsound import playsound

#----------------------------- PLAY MP4 CLASS --------------------------------
# class Video(object):
    # def __init__(self,path):
        # self.path = path

    # def play(self):
        # from os import startfile
        # startfile(self.path)

# class Movie_MP4(Video):
    # type = "MP4"
#----------------------------- You can use this class if you want to --------------------------------

#STEP 2: example of text. Replace this text to the text you want to convert to speech
# The text that you want to convert to audio
mytext = 'Good luck for your presentation!'
  
# Language in which you want to convert (you can use other languages if you want, check here https://gtts.readthedocs.io/en/latest/module.html#languages-gtts-lang )
language = 'en'

# Passing the text and language to the engine, 
# here we have marked slow=False. Which tells 
# the module that the converted audio should 
# have a high speed
myobj = gTTS(text=mytext, lang=language, slow=False)
  
# Saving the converted audio in a mp3 file named
# welcome 
myobj.save("audio.mp4")
playsound(r'audio.mp4')
#movie = Movie_MP4(r"welcome.mp4")
#movie.play()
