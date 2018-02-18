# ▣ 2.1 bit.ly의 1.usa.gov 데이터
path = 'PythonForDataAnalysis/ch02/usagov_bitly_data2012-03-16-1331923249.txt'
print(open(path).readline())

# - json 모듈의 loads 함수로 내려받은 샘플 파일을 한 줄씩 읽는다.
import json
path = 'PythonForDataAnalysis/ch02/usagov_bitly_data2012-03-16-1331923249.txt'
records = [json.loads(line) for line in open(path, encoding='utf-8')]
print(records[0])

# - records 의 개별 아이템에서 접근하려는 값의 키를 문자열로 넘겨서 쉽게 읽어온다.
print(records[0]['tz'])

# ■ 2.1.1 순수 파이썬으로 표준시간대 세어보기
time_zones = [rec['tz'] for rec in records if 'tz' in rec]
time_zones[:10]

# - 파이썬 표준 라이브러리를 통해 카운팅
def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts

from collections import defaultdict
def get_counts2(sequence):
    counts = defaultdict(int)  # 값은 0으로 초기화된다.
    for x in sequence:
        counts[x] += 1
    return counts

counts = get_counts(time_zones)
counts['America/New_York']
len(time_zones)

# - 가장 많이 등장하는 상위 10개의 표준시간대를 확인하는 방법
def top_counts(count_dict, n=10):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort(reverse=True)
    return value_key_pairs[:n]
top_counts(counts)

# - 파이썬 표준 라이브러리의 collections.Counter 클래스를 이용한 카운팅
from collections import Counter
counts = Counter(time_zones)
counts.most_common(10)

# ■ 2.1.2 pandas 로 표준시간대 세어보기
from pandas import DataFrame, Series
import pandas as pd
frame = DataFrame(records)
frame.info()
frame['tz'][:10]

# - Series 객체의 value_counts 메서드를 통해 카운팅
tz_counts = frame['tz'].value_counts()
tz_counts[:10]

# - plot 메서드를 통한 수평 막대 그래프 그리기
tz_counts[:10].plot(kind='barh', rot=0)

# - URL 을 축약하는 데 사용한 브라우저, 단말기, 애플리케이션에 대한 정보를 담은 필드
frame['a'][1]
frame['a'][50]
frame['a'][51]

results = Series([x.split()[0] for x in frame.a.dropna()])
results[:5]
results.value_counts()[:8]

# - agent 값이 없는 데이터를 제외
cframe = frame[frame.a.notnull()]

# - 각 행이 윈도우인지 아닌지 검사
import numpy as np
operating_system = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
operating_system[:5]

by_tz_os = cframe.groupby(['tz', operating_system])

# - size 함수로 그룹별 합계를 구하고, 결과는 unstack 함수를 이용해서 표로 재배치한다.
agg_counts = by_tz_os.size().unstack().fillna(0)
agg_counts[:10]

# - 전체 표준시간대의 순위를 모아보자.
indexer = agg_counts.sum(1).argsort()
indexer[:10]

# - agg_counts 에 take 를 사용해 행을 정렬된 순서 그대로 선택하고 마지막 10개 행만 잘라낸다.
count_subset = agg_counts.take(indexer)[-10:]
count_subset.plot(kind='barh', stacked=True)

# - 각 행에서 총합을 1로 정규화한 후 표를 만든다.
normed_subset = count_subset.div(count_subset.sum(1), axis=0)
normed_subset.plot(kind='barh', stacked=True)

# ▣ 2.2 MovieLens 의 영화 평점 데이터
import pandas as pd
encoding = 'latin1'

upath = 'PythonForDataAnalysis\\ch02\\users.dat'
rpath = 'PythonForDataAnalysis\\ch02\\ratings.dat'
mpath = 'PythonForDataAnalysis\\ch02\\movies.dat'

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
mnames = ['movie_id', 'title', 'genres']

users = pd.read_csv(upath, sep='::', header=None, names=unames, encoding=encoding)
ratings = pd.read_csv(rpath, sep='::', header=None, names=rnames, encoding=encoding)
movies = pd.read_csv(mpath, sep='::', header=None, names=mnames, encoding=encoding)

users[:5]
ratings[:5]
movies[:5]
ratings

# - 3 개의 테이블에 대해 merge 수행
data = pd.merge(pd.merge(ratings, users), movies)
data
data.ix[0]

# - 성별에 따른 각 영화의 평균 평점은 pivot_table 메서드를 사용해서 구한다.
pd.pivot_table()
mean_ratings = data.pivot_table('rating', index='title', columns='gender', aggfunc='mean')
mean_ratings[:5]

# - 250 건 이상의 평점 정보가 있는 영화만 추출
ratings_by_title = data.groupby('title').size()
ratings_by_title[:10]
active_titles = ratings_by_title.index[ratings_by_title >= 250]
active_titles
