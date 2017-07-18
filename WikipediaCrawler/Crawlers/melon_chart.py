import re
import matplotlib.pyplot as plt
import operator

class MelonChart(object):
    DIRPATH = 'D:\\02.Python\\data\\'

    def __init__(self, filename):
        self.filename = filename
        self.data = {}

    def setting_data(self):
        with open(MelonChart.DIRPATH+self.filename, 'rt', encoding='utf-8') as f:
            for line in iter(f):
                temp = line.split(';')
                if temp[0] == 'KPOP':
                    if self.data.get(int(temp[1])) is None:
                        self.data[int(temp[1])] = {'ENG': 0, 'KOR': 0}
                    if re.search('[a-zA-Z]+', temp[3]) is None:
                        self.data[int(temp[1])]['KOR'] += 1
                    else:
                        self.data[int(temp[1])]['ENG'] += 1
                else:
                    break

    def create_chart(self):
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        fig.subplots_adjust(left=.06, right=.75, bottom=.05, top=.94)

        ax.set_xlim(1964, 2016)
        ax.set_ylim(-0.25, 101)

        plt.xticks(range(1970, 2011, 10), fontsize=14)
        plt.yticks(range(0, 101, 10), fontsize=14)
        ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.0f}'.format))
        ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))

        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

        plt.tick_params(axis='both', which='both', bottom='off', top='off',
                        labelbottom='on', left='off', right='off', labelleft='on')

        color_sequence = ['#1f77b4', '#aec7e8']
        majors = ['KOR', 'ENG']
        y_offsets = {'KOR': 0.5, 'ENG': 0.5}

        for rank, column in enumerate(majors):
            plt.plot(tuple(year for year in range(1964, 2017)),
                     tuple(d[1][column] / (d[1]['KOR']+d[1]['ENG']) * 100 for d in sorted(self.data.items(), key=operator.itemgetter(0))),
                     lw=2.5,
                     color=color_sequence[rank])

            y_pos = self.data[2016][column]/(self.data[2016]['KOR'] + self.data[2016]['ENG']) * 100

            plt.text(2016.5, y_pos, column, fontsize=14, color=color_sequence[rank])

        # fig.suptitle('', fontsize=18, ha='center')
        plt.show()

chart = MelonChart('chart_data.txt')
chart.setting_data()
chart.create_chart()
