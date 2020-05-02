def bubble(list):
    for iter_num in range(len(list)-1,0,-1):
        for idx in range(iter_num):
            if list[idx]>list[idx+1]:
                temp = list[idx]
                list[idx] = list[idx+1]
                list[idx+1] = temp

list = [19,2,31,45,6,11,121,27]
bubble(list=list)
print(list)

def insertion(list):
    for i in range(1, len(list)):
        j = i-1
        nxt_element = list[i]
        while (list[j] > nxt_element) and (j >= 0):
            list[j+1] = list[j]
            j = j-1
        list[j+1] = nxt_element

insertion(list)
print(list)