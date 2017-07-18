from urllib.request import urlopen
from bs4 import BeautifulSoup
from WikipediaCrawler.database import Database

class SolutionFound(RuntimeError):
    def __init__(self, message):
        self.message = message

def getLinks(fromUrlId):
    Database.createConn()
    conn = Database.getConnection()
    cur = conn.cursor()
    cur.execute('select tourlid from links where fromurlid = :1', [fromUrlId])
    data = cur.fetchall()
    Database.releaseConn(cur, conn)

    if data is None:
        return None
    else:
        return [x[0] for x in data]

def constructDict(currentUrlId):
    links = getLinks(currentUrlId)
    if links:
        return dict(zip(links, [{}]*len(links)))
    return {}

# 링크 트리가 비어 있거나 링크가 여러 개 들어 있습니다.
def searchDepth(targetUrlId, currentUrlId, linkTree, depth):
    if depth == 0:
        # 재귀를 중지하고 함수를 끝냅니다.
        return linkTree

    if not linkTree:
        linkTree = constructDict(currentUrlId)
        print(linkTree)
        if not linkTree:
            # 링크가 발견되지 않았으므로 이 노드에서는 계속할 수 없습니다.
            return {}

    if targetUrlId in linkTree.keys():
        print('TARGET '+str(targetUrlId)+' FOUND!')
        raise SolutionFound('PAGE : '+str(currentUrlId))

    for branchKey, branchValue in linkTree.items():
        try:
            # 재귀적으로 돌아와서 링크 트리를 구축합니다.
            linkTree[branchKey] = searchDepth(targetUrlId, branchKey, branchValue, depth-1)
        except SolutionFound as e:
            print(e.message)
            raise SolutionFound('PAGE : '+str(currentUrlId))
    return linkTree

try:
    searchDepth(2000, 1, {}, 4)
    print('No solution found')
except SolutionFound as e:
    print(e.message)