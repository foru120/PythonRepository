from urllib.request import urlopen
from pdfminer.pdfinterp import PDFResourceManager, process_pdf
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO
from io import open

def readPDF(pdfFile):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)

    process_pdf(rsrcmgr, device, pdfFile)
    device.close()

    content = retstr.getvalue()
    retstr.close()

    return content

pdfFile = urlopen('http://pythonscraping.com/pages/warandpeace/chapter1.pdf')  # 원격지 파일 open
# pdfFile = open('downloads/chapter1.pdf', 'rb')  # 로컬 파일 open(파이썬 파일 객체를 반환받지 않고 가능)
outputString = readPDF(pdfFile)
print(outputString)
pdfFile.close()