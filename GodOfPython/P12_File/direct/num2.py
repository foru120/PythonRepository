f = open('D:/02.Python/ch12/direct/num2.txt', 'r+')
item_list = []

for line in f:
    item_list.append(line.strip().split('|'))

while True:
    print('데이터를 입력하세요.(종료 : q)')
    data = input()
    
    if data=='q':
        break
    else:
        temp = data.strip().split(' ')
        bol = False
        for item in item_list:
            if item[0]==temp[0] and item[1]==temp[1]:
                item[2] = str(int(item[2])+int(temp[2]))
                bol = True

        if bol==False:
            item_list.append(temp)

i=0         
for item in item_list:
    item_list[i]='|'.join(item_list[i])
    i+=1
    
f.seek(0, 0)
f.write('\n'.join(item_list))    
f.close()