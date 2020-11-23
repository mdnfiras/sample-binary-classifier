import os
import sys
import urllib.request
import pandas
import tensorflow
import keras

print("Running...")

if (not(os.path.isfile("report"))):
    os.mkdir("report")
with open("report/metrics.txt", "w") as f:
    print("Nice job", file=f)
url = "https://thumbs.dreamstime.com/b/card-minimal-cartoonish-style-vector-templates-nice-job-lettering-speech-bubble-89326194.jpg"
urllib.request.urlretrieve(url, 'report/image.jpg')

