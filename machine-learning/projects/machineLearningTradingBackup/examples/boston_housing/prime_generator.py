# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 23:09:13 2016

@author: rcortez
"""
import math

def get_primes(number):
    while True:
        if is_prime(number):
            yield number
        number += 1
        
def solve_number_10():
    # She *is* working on Project Euler #10, I knew it!
    total = 2
    for next_prime in get_primes(3):
        if next_prime < 200:
            print next_prime
            total += next_prime
        else:
            print(total)
            return        
            
def is_prime(number):
    if number > 1:
        if number == 2:
            return True
        if number % 2 == 0:
            return False
        for current in range(3, int(math.sqrt(number) + 1), 2):
            if number % current == 0: 
                return False
        return True
    return False            
            
if __name__ == "__main__":            
    print solve_number_10()