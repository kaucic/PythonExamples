def get_data(degree):
  a = []
  for i in range(0,degree+1):
    msg = 'Please input coefficient for x^' + repr(degree-i) + ' ' 
    print (msg)
    coeff = input()
    a.insert(i,coeff)
  return a

n = int (raw_input ('Please input degree of polynomial '))
b = get_data(n)

msg2 = 'Your polynomial is '
for i in range(0,n+1):
  val2 = repr(b[i]) + 'x^' + repr(n-i) + ' + ' 
  msg2 = msg2 + val2
print msg2

