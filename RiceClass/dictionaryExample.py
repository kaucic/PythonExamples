# -*- coding: utf-8 -*-

d = {1:'a', 2:'b', 4:'d', 5:'e'}

print d.keys()
print d.values()

d2 = {}
l2 = "zyxwvutsrq"

for i in range(5):
    d2[i**2] = l2[i]
    
print d2
