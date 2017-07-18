f = open('D:/02.Python/ch12/direct/num2.txt', 'r+')
item_list = []

for line in f:
    item_list.append(line.strip().split('|'))

while True:
    print('-------------------------------------')
    print('-- 01. 전체 물품 출력(a) ------------')
    print('-- 02. 기존 물품 수량 변경(b) -------')
    print('-- 03. 새로운 물품 등록(c) ----------')
    print('-- 04. 종료(q) ----------------------')
    print('-------------------------------------')

    menu = input()
    
    if menu=='q':
        break
    elif menu=='a':
        for item in item_list:
            print(item)
    elif menu=='b':
        print('물품명과 수량을 입력하세요.(물품명 수량)')
        temp = input().strip().split(' ')
        bol = False

        for item in item_list:
            if item[0]==temp[0]:
                item[2]=temp[1]
                bol = True
                break

        if bol==False:
            print('입력하신 물품은 존재하지 않습니다.')
    elif menu=='c':
        print('새로운 물품을 등록하세요.(물품명 가격 수량)')
        temp = input().strip().split(' ')
        bol = False

        for item in item_list:
            if item[0]==temp[0]:
                print('이미 존재하는 물품입니다')
                bol = True
                break

        if bol == False:
            item_list.append(temp)
    else:
        print('존재하지 않는 메뉴입니다.')

i=0         
for item in item_list:
    item_list[i]='|'.join(item_list[i])
    i+=1
    
f.seek(0, 0)
f.write('\n'.join(item_list))    
f.close()