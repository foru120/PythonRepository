from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os
import fnmatch

class Wordcloud(object):
    DIRPATH = 'D:\\02.Python\\data\\'

    def __init__(self, genre, age):
        self.genre = genre
        self.age = age
        self.data = ''

    def create_word_cloud(self):
        for f in [filename for filename in os.listdir(Wordcloud.DIRPATH) if fnmatch.fnmatch(filename, self.genre+'_'+self.age+'_*.txt')]:
            with open(os.path.join(Wordcloud.DIRPATH, f), mode="r", encoding="UTF-8") as file:
                self.data += file.read()

        wordcloud = WordCloud(font_path='C://Windows//Fonts//a몬스터.TTF',
                              stopwords=STOPWORDS, background_color='black',
                              width=500,
                              height=400,
                              colormap='spring').generate(self.data)
        return wordcloud

GENRE = {'POP': (1950, 2010), 'KPOP': (1960, 2010)}

for genre, ages in GENRE.items():
    col_cnt = round(((ages[1]-ages[0])/10+1)/2)
    cnt = 1
    fig = plt.figure(figsize=(18, 9))
    for age in range(ages[0], ages[1]+1, 10):
        plt.subplot(2, col_cnt, cnt)
        plt.title(str(age)+"'s")
        plt.imshow(Wordcloud(genre, str(age)).create_word_cloud())
        plt.axis('off')
        cnt += 1
    plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    fig.savefig('D:\\02.Python\\data\\'+genre+'_'+'plot.jpg', dpi=100)
    plt.show()