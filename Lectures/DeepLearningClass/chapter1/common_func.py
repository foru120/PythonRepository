import csv

class CommonFunc(object):
    @classmethod
    def returnCsvData(cls, filename):
        file = open('D:\\KYH\\02.PYTHON\\PythonRepository\\DeepLearningClass\\chapter1\\data\\'+filename, 'r', encoding='utf-8')
        return csv.reader(file)

    @classmethod
    def returnCsvDataDef(cls, filename):
        file = open('D:\\KYH\\02.PYTHON\\PythonRepository\\DeepLearningClass\\chapter1\\data\\'+filename, 'r')
        return csv.reader(file)

    @classmethod
    def returnTxtData(cls, filename):
        file = open('D:\\KYH\\02.PYTHON\\PythonRepository\\DeepLearningClass\\chapter1\\data\\' + filename, 'r')
        return file