from PIL import Image
import subprocess

def cleanFile(filePath, newFilePath):
    image = Image.open(filePath)

    # 회색 임계점을 설정하고 이미지를 저장합니다.
    image = image.point(lambda x: 0 if x < 143 else 255)
    image.save(newFilePath)

    # 새로 만든 이미지를 테서랙트로 읽습니다.
    # print(os.getcwd()+'\\'+newFilePath) # 절대 경로
    subprocess.call(['tesseract', newFilePath, 'output'])

    # 결과 텍스트 파일을 열어 읽습니다.
    outputFile = open('output.txt', 'r')
    print(outputFile.read())
    outputFile.close()

cleanFile('text_2.png', 'text_2_clean.png')