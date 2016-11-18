# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:41:04 2016

@author: rcortez
""" 
#==============================================================================
#  GENERATORS
#==============================================================================
import math
import random



def main():
#    solve_number_10()
#    print_successive_primes(10)
    for power in range(0, 3):
        get_numPrimes_base10(10 ** power)


def get_primes(number):
    while True:
        if is_prime(number):
            yield number
        number += 1            
   
def get_numPrimes_base10(base = 10):
    prime_gen = get_primes(base)
    prime_gen.send(None)
    count = 0
    for next_prime in get_primes(base):
        if next_prime < base * 10:
#            print next_prime
            count += 1
        else:
            print "total: " + str(count)
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
    
def simple_generator():
    yield 1
    yield 2
    yield 3
    
def solve_number_10():
    #euler #10
    total = 2
    for next_prime in get_primes(3000000):
        if next_prime < 3030000:            
            total += next_prime
            print next_prime            
        else:
            print total
            return 
    
def print_successive_primes(iterations, base = 10):
    prime_generator = get_primes(base)    
    prime_generator.send(None)
    for power in range(iterations):        
        print prime_generator.send(base ** power)
    
if __name__ == "__main__":
    main()    