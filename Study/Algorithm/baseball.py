import random
import re

###################################################################################################
## 기록 관련 클래스
###################################################################################################
class Record:
    def __init__(self):
        self.__hit = 0  # 안타 수
        self.__homerun = 0  # 홈런 수
        self.__atbat = 0  # 타수
        self.__avg = 0.0  # 타율

    @property
    def hit(self):
        return self.__hit

    @hit.setter
    def hit(self, hit):
        self.__hit = hit

    @property
    def homerun(self):
        return self.__homerun

    @homerun.setter
    def homerun(self, homerun):
        self.__homerun = homerun

    @property
    def atbat(self):
        return self.__atbat

    @atbat.setter
    def atbat(self, atbat):
        self.__atbat = atbat

    @property
    def avg(self):
        return self.__avg

    @avg.setter
    def avg(self, avg):
        self.__avg = avg

    # 타자 기록 관련 메서드
    def batter_record(self, hit, homerun):
        self.hit += hit
        self.homerun += homerun
        self.atbat += 1
        self.avg = self.hit / self.atbat


###################################################################################################
## 선수 관련 클래스
###################################################################################################
class Player:
    def __init__(self, team_name, number, name):
        self.__team_name = team_name  # 팀 이름
        self.__number = number  # 타순
        self.__name = name  # 이름
        self.__record = Record()  # 기록

    @property
    def team_name(self):
        return self.__team_name

    @property
    def number(self):
        return self.__number

    @property
    def name(self):
        return self.__name

    @property
    def record(self):
        return self.__record

    @property
    def player_info(self):
        return self.__team_name + ', ' + str(self.__number) + ', ' + self.__name

    # 선수 타율 관련 메서드
    def hit_and_run(self, hit, homerun):
        self.__record.batter_record(hit, homerun)


###################################################################################################
## 팀 관련 클래스
###################################################################################################
class Team:
    def __init__(self, team_name, players):
        self.__team_name = team_name  # 팀 이름
        self.__player_list = self.init_player(players)  # 해당 팀 소속 선수들 정보

    @property
    def team_name(self):
        return self.__team_name

    @property
    def player_list(self):
        return self.__player_list

    # 선수단 초기화
    def init_player(self, players):
        temp = []
        for player in players:
            number, name = list(player.items())[0]
            temp.append(Player(self.__team_name, number, name))
        return temp

    def show_players(self):
        for player in self.__player_list:
            print(player.player_info)


###################################################################################################
## 게임 관련 클래스
###################################################################################################
class Game:
    TEAM_LIST = {
        '한화': ({1: '정근우'}, {2: '이용규'}, {3: '송광민'}, {4: '최진행'}, {5: '하주석'}, {6: '장민석'}, {7: '로사리오'}, {8: '이양기'}, {9: '최재훈'}),
        '롯데': ({1: '나경민'}, {2: '손아섭'}, {3: '최준석'}, {4: '이대호'}, {5: '강민호'}, {6: '김문호'}, {7: '정훈'}, {8: '번즈'}, {9: '신본기'}),
        '삼성': ({1: '박해민'}, {2: '강한울'}, {3: '구자욱'}, {4: '이승엽'}, {5: '이원석'}, {6: '조동찬'}, {7: '김헌곤'}, {8: '이지영'}, {9: '김정혁'}),
        'KIA': ({1: '버나디나'}, {2: '이명기'}, {3: '나지완'}, {4: '최형우'}, {5: '이범호'}, {6: '안치홍'}, {7: '서동욱'}, {8: '김민식'}, {9: '김선빈'}),
        'SK': ({1: '노수광'}, {2: '정진기'}, {3: '최정'}, {4: '김동엽'}, {5: '한동민'}, {6: '이재원'}, {7: '박정권'}, {8: '김성현'}, {9: '박승욱'}),
        'LG': ({1: '이형종'}, {2: '김용의'}, {3: '박용택'}, {4: '히메네스'}, {5: '오지환'}, {6: '양석환'}, {7: '임훈'}, {8: '정상호'}, {9: '손주인'}),
        '두산': ({1: '허경민'}, {2: '최주환'}, {3: '민병헌'}, {4: '김재환'}, {5: '에반스'}, {6: '양의지'}, {7: '김재호'}, {8: '신성현'}, {9: '정진호'}),
        '넥센': ({1: '이정후'}, {2: '김하성'}, {3: '서건창'}, {4: '윤석민'}, {5: '허정협'}, {6: '채태인'}, {7: '김민성'}, {8: '박정음'}, {9: '주효상'}),
        'KT': ({1: '심우준'}, {2: '정현'}, {3: '박경수'}, {4: '유한준'}, {5: '장성우'}, {6: '윤요섭'}, {7: '김사연'}, {8: '오태곤'}, {9: '김진곤'}),
        'NC': ({1: '김성욱'}, {2: '모창민'}, {3: '나성범'}, {4: '스크럭스'}, {5: '권희동'}, {6: '박석민'}, {7: '지석훈'}, {8: '김태군'}, {9: '이상호'})
    }

    INNING = 1  # 1 이닝부터 시작
    CHANGE = 0  # 0 : hometeam, 1 : awayteam
    STRIKE_CNT = 0  # 스트라이크 개수
    OUT_CNT = 0  # 아웃 개수
    ADVANCE = [0, 0, 0]  # 진루 상황
    SCORE = [0, 0]  # [home, away]
    BATTER_NUMBER = [1, 1]  # [home, away] 타자 순번

    def __init__(self, game_team_list):
        print('====================================================================================================')
        print('== 선수단 구성')
        print('====================================================================================================')
        print(game_team_list[0]+' : ', Game.TEAM_LIST[game_team_list[0]])
        print(game_team_list[1]+' : ', Game.TEAM_LIST[game_team_list[1]])
        print('====================================================================================================')
        self.__hometeam = Team(game_team_list[0], Game.TEAM_LIST[game_team_list[0]])
        self.__awayteam = Team(game_team_list[1], Game.TEAM_LIST[game_team_list[1]])
        print('== 선수단 구성이 완료 되었습니다.\n')

    @property
    def hometeam(self):
        return self.__hometeam

    @property
    def awayteam(self):
        return self.__awayteam

    # 게임 수행 메서드
    def start_game(self):
        while Game.INNING <= 1:
            print('====================================================================================================')
            print('== {} 이닝 {} 팀 공격 시작합니다.'.format(Game.INNING, self.hometeam.team_name if Game.CHANGE == 0 else self.awayteam.team_name))
            print('====================================================================================================\n')
            self.attack()

            if Game.CHANGE == 2:  # 이닝 교체
                Game.INNING += 1
                Game.CHANGE = 0

        print('====================================================================================================')
        print('== 게임 종료!!!')
        print('====================================================================================================\n')
        self.show_record()

    # 팀별 선수 기록 출력
    def show_record(self):
        print('====================================================================================================')
        print('==  {} | {}   =='.format(self.hometeam.team_name.center(44, ' ') if re.search('[a-zA-Z]+', self.hometeam.team_name) is not None else self.hometeam.team_name.center(42, ' '),
                                        self.awayteam.team_name.center(44, ' ') if re.search('[a-zA-Z]+', self.awayteam.team_name) is not None else self.awayteam.team_name.center(42, ' ')))
        print('==  {} | {}   =='.format(('('+str(Game.SCORE[0])+')').center(44, ' '), ('('+str(Game.SCORE[1])+')').center(44, ' ')))
        print('====================================================================================================')
        print('== {} | {} | {} | {} | {} '.format('이름'.center(8, ' '), '타율'.center(5, ' '), '타석'.center(4, ' '), '안타'.center(3, ' '), '홈런'.center(3, ' ')), end='')
        print('| {} | {} | {} | {} | {}  =='.format('이름'.center(8, ' '), '타율'.center(5, ' '), '타석'.center(4, ' '), '안타'.center(3, ' '), '홈런'.center(3, ' ')))
        print('====================================================================================================')

        hometeam_players = self.hometeam.player_list
        awayteam_players = self.awayteam.player_list

        for i in range(9):
            hp = hometeam_players[i]
            hp_rec = hp.record
            ap = awayteam_players[i]
            ap_rec = ap.record

            print('== {} | {} | {} | {} | {} |'.format(hp.name.center(6+(4-len(hp.name)), ' '), str(hp_rec.avg).center(7, ' '),
                                                      str(hp_rec.atbat).center(6, ' '), str(hp_rec.hit).center(5, ' '), str(hp_rec.homerun).center(5, ' ')), end='')
            print(' {} | {} | {} | {} | {}  =='.format(ap.name.center(6+(4-len(ap.name)), ' '), str(ap_rec.avg).center(7, ' '),
                                                        str(ap_rec.atbat).center(6, ' '), str(ap_rec.hit).center(5, ' '), str(ap_rec.homerun).center(5, ' ')))
        print('====================================================================================================')

    # 공격 수행 메서드
    def attack(self):
        curr_team = self.hometeam if Game.CHANGE == 0 else self.awayteam
        player_list = curr_team.player_list

        if Game.OUT_CNT < 3:
            player = self.select_player(Game.BATTER_NUMBER[Game.CHANGE], player_list)
            print('====================================================================================================')
            print('== [{}] {}번 타자[{}] 타석에 들어섭니다.'.format(curr_team.team_name, player.number, player.name))
            print('====================================================================================================\n')

            while True:
                random_numbers = self.throws_numbers()  # 컴퓨터가 랜덤으로 숫자 4개 생성
                print('== [전광판] =========================================================================================')
                print('==   {}      | {} : {}'.format(Game.ADVANCE[1], self.hometeam.team_name, Game.SCORE[0]))
                print('==  {}  {}    | {} : {}'.format(Game.ADVANCE[2], Game.ADVANCE[0], self.awayteam.team_name, Game.SCORE[1]))
                print('== [OUT : {}, STRIKE : {}]'.format(Game.OUT_CNT, Game.STRIKE_CNT))
                print('====================================================================================================')
                print('== 현재 타석 : {}번 타자[{}], 타율 : {}'.format(player.number, player.name, player.record.avg))

                try:
                    hit_numbers = set(int(hit_number) for hit_number in input('== 숫자를 입력하세요(1~40) : ').split(' '))  # 유저가 직접 숫자 4개 입력
                    if self.hit_number_check(hit_numbers) is False:
                        raise Exception()
                except Exception:
                    print('== ▣ 잘못된 숫자가 입력되었습니다.')
                    print('====================================================================================================')
                    print('▶ 컴퓨터가 발생 시킨 숫자 : {}\n'.format(random_numbers))
                    continue
                print('====================================================================================================')
                print('▶ 컴퓨터가 발생 시킨 숫자 : {}\n'.format(random_numbers))

                hit_cnt = self.hit_judgment(random_numbers, hit_numbers)  # 안타 판별
                if hit_cnt == 0:  # strike !!!
                    Game.STRIKE_CNT += 1
                    print('== ▣ 스트라이크!!!\n')
                    if Game.STRIKE_CNT == 3:
                        print('== ▣ 삼진 아웃!!!\n')
                        Game.STRIKE_CNT = 0
                        Game.OUT_CNT += 1
                        break
                else:
                    Game.STRIKE_CNT = 0
                    if hit_cnt != 4:
                        print('== ▣ {}루타!!!\n'.format(hit_cnt))
                    else:
                        print('== ▣ 홈런!!!\n')
                    self.advance_setting(hit_cnt)
                    break

            player.hit_and_run(1 if hit_cnt > 0 else 0, 1 if hit_cnt == 4 else 0)
            if Game.BATTER_NUMBER[Game.CHANGE] == 9:
                Game.BATTER_NUMBER[Game.CHANGE] = 1
            else:
                Game.BATTER_NUMBER[Game.CHANGE] += 1
            self.attack()
        else:
            Game.CHANGE += 1
            Game.STRIKE_CNT = 0
            Game.OUT_CNT = 0
            Game.ADVANCE = [0, 0, 0]
            print('====================================================================================================')
            print('== 공수교대 하겠습니다.')
            print('====================================================================================================\n')

    # 진루 및 득점 설정하는 메서드
    def advance_setting(self, hit_cnt):
        if hit_cnt == 4:  # 홈런인 경우
            Game.SCORE[Game.CHANGE] += Game.ADVANCE.count(1)
            Game.ADVANCE = [0, 0, 0]
        else:
            for i in range(len(Game.ADVANCE), 0, -1):
                if Game.ADVANCE[i-1] == 1:
                    if (i + hit_cnt) > 3:  # 기존에 출루한 선수들 중 득점 가능한 선수들에 대한 진루 설정
                        Game.SCORE[Game.CHANGE] += 1
                        Game.ADVANCE[i-1] = 0
                    else:  # 기존 출루한 선수들 중 득점권에 있지 않은 선수들에 대한 진루 설정
                        Game.ADVANCE[i-1 + hit_cnt] = 1
                        Game.ADVANCE[i-1] = 0
            Game.ADVANCE[hit_cnt-1] = 1  # 타석에 있던 선수에 대한 진루 설정

    # 컴퓨터가 생성한 랜덤 수와 플레이어가 입력한 숫자가 얼마나 맞는지 판단
    def hit_judgment(self, random_ball, hit_numbers):
        cnt = 0
        for hit_number in hit_numbers:
            if hit_number in random_ball:
                cnt += 1
        return cnt

    # 선수가 입력한 숫자 확인
    def hit_number_check(self, hit_numbers):
        if len(hit_numbers) == 4:
            for hit_number in hit_numbers:
                if hit_number <= 0 or hit_number > 40:
                    return False
            return True
        return False

    # 선수 선택
    def select_player(self, number, player_list):
        for player in player_list:
            if number == player.number:
                return player

    # 랜덤으로 숫자 생성(1~20)
    def throws_numbers(self):
        random_balls = set()
        while True:
            random_balls.add(random.randint(1, 40))  # 1 ~ 20 중에 랜덤 수를 출력
            if len(random_balls) == 4:  # 생성된 ball 이 4개 이면(set 객체라 중복 불가)
                return random_balls


if __name__ == '__main__':
    while True:
        game_team_list = []
        print('====================================================================================================')
        game_team_list = input('=> 게임을 진행할 두 팀을 입력하세요 : ').split(' ')
        print('====================================================================================================\n')
        if (game_team_list[0] in Game.TEAM_LIST) and (game_team_list[1] in Game.TEAM_LIST):
            game = Game(game_team_list)
            game.start_game()
            break
        else:
            print('입력한 팀 정보가 존재하지 않습니다. 다시 입력해주세요.')