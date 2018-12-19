# -*- coding: utf-8 -*-
"""
Created on Wed Sep 09 20:15:02 2015

@author: Kimberly
"""
  
try:
    import simplegui
except ImportError:
    import SimpleGUICS2Pygame.simpleguics2pygame as simplegui 
    
import math

class Overload:
    def __init__(self,a,b=4):
        self.x = a
        self.y = b
        
    def __str__(self):
        out = "Pair = (" + str(self.x)  + "," + str(self.y) + ")"
        return (out)
    
        
x1 = Overload(3)
x2 = Overload(3,5)
print x1,x2
print "\n"

class BankAccount:
    def __init__(self, initial_balance):
        """Creates an account with the given balance."""
        self.balance = initial_balance
        self.fees = 0
    
    def deposit(self, amount):
        """Deposits the amount into the account."""
        self.balance += amount
        
    def withdraw(self, amount):
        """
        Withdraws the amount from the account.  Each withdrawal resulting in a
        negative balance also deducts a penalty fee of 5 dollars from the balance.
        """
        self.balance -= amount
        if (self.balance < 0):
            self.balance -= 5
            self.fees += 5
            
    def get_balance(self):
        """Returns the current balance in the account."""
        return (self.balance)
        
    def get_fees(self):
        """Returns the total fees ever deducted from the account."""
        return (self.fees)        
        
my_account = BankAccount(10)
my_account.withdraw(5)
my_account.deposit(10)
my_account.withdraw(5)
my_account.withdraw(15)
my_account.deposit(20)
my_account.withdraw(5) 
my_account.deposit(10)
my_account.deposit(20)
my_account.withdraw(15)
my_account.deposit(30)
my_account.withdraw(10)
my_account.withdraw(15)
my_account.deposit(10)
my_account.withdraw(50) 
my_account.deposit(30)
my_account.withdraw(15)
my_account.deposit(10)
my_account.withdraw(5) 
my_account.deposit(20)
my_account.withdraw(15)
my_account.deposit(10)
my_account.deposit(30)
my_account.withdraw(25) 
my_account.withdraw(5)
my_account.deposit(10)
my_account.withdraw(15)
my_account.deposit(10)
my_account.withdraw(10) 
my_account.withdraw(15)
my_account.deposit(10)
my_account.deposit(30)
my_account.withdraw(25) 
my_account.withdraw(10)
my_account.deposit(20)
my_account.deposit(10)
my_account.withdraw(5) 
my_account.withdraw(15)
my_account.deposit(10)
my_account.withdraw(5) 
my_account.withdraw(15)
my_account.deposit(10)
my_account.withdraw(5) 
print my_account.get_balance(), my_account.get_fees()
print "\n"

account1 = BankAccount(20)
account1.deposit(10)
account2 = BankAccount(10)
account2.deposit(10)
account2.withdraw(50)
account1.withdraw(15)
account1.withdraw(10)
account2.deposit(30)
account2.withdraw(15)
account1.deposit(5)
account1.withdraw(10)
account2.withdraw(10)
account2.deposit(25)
account2.withdraw(15)
account1.deposit(10)
account1.withdraw(50)
account2.deposit(25)
account2.deposit(25)
account1.deposit(30)
account2.deposit(10)
account1.withdraw(15)
account2.withdraw(10)
account1.withdraw(10)
account2.deposit(15)
account2.deposit(10)
account2.withdraw(15)
account1.deposit(15)
account1.withdraw(20)
account2.withdraw(10)
account2.deposit(5)
account2.withdraw(10)
account1.deposit(10)
account1.deposit(20)
account2.withdraw(10)
account2.deposit(5)
account1.withdraw(15)
account1.withdraw(20)
account1.deposit(5)
account2.deposit(10)
account2.deposit(15)
account2.deposit(20)
account1.withdraw(15)
account2.deposit(10)
account1.deposit(25)
account1.deposit(15)
account1.deposit(10)
account1.withdraw(10)
account1.deposit(10)
account2.deposit(20)
account2.withdraw(15)
account1.withdraw(20)
account1.deposit(5)
account1.deposit(10)
account2.withdraw(20)
print account1.get_balance(), account1.get_fees(), account2.get_balance(), account2.get_fees()


CARD_SIZE = (73,98)
CARD_CENTER = (36.5,49)

r = CARD_SIZE[0]*12 + CARD_CENTER[0]
c = CARD_SIZE[1]*3 + CARD_CENTER[1]
print r,c
print "\n"

n = 1000
numbers = [i for i in range(2,n)]
results = []
while (len(numbers) > 0):
    new_num = numbers[0]
    results.append(new_num)
    numbers.remove(new_num)
    remove_list = []
    for num in numbers:
        if (num % new_num == 0):
            remove_list.append(num)
    for element in remove_list:
        numbers.remove(element)

# print results
print len(results)
print "\n"
        
slow = 1000
fast = 1

i = 1
while (slow > fast):
    if (i % 10 == 0):
        print i, slow, fast
    slow *= 1.2
    fast *= 1.4
    i += 1
print "Fast exceeds slow at the start of the ",i,"th year"
print "\n"


