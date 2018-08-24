import os

for (path, dirs, files) in os.walk('G:\\04_dataset\\eye_verification\\eye_only_v3\\train\\right'):
    for name in dirs:
        print(name)

    for name in files:
        print(path)