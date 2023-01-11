#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 08:43:18 2023

@author: ai_ds-a1
"""
import numpy as np
import random

cd={'A':6,'C':6,'G':2,'T':2}
random.seed(10)
def gen_s(s=list('AAAAAACCCCCCGGTT')):
    s1=[]
    num=list(range(16))
    for i in range(16):
        c=random.choice(num)
        s1.append(s[c])
        num.remove(c)
    return s1
s1=gen_s()
s2=gen_s()
#s1="ACACACTA"
#s2="AGCACACA"
match=5
mismatch=4

print(s1,s2)
l1,l2=16,16
al_mat=np.zeros((16, 16),dtype=("int"))
#%%
def call(al_mat,i=1,j=1):
    if i==l1 and j==l2:
        return al_mat
    if s1[i-1]!=s2[j-1]:
        print(s1[i-1],s2[j-1])
        x = max (al_mat[i][j-1],al_mat[i-1][j],al_mat[i-1][j-1])
        al_mat[i][j]=x-mismatch
    else:
        print(s1[i-1],s2[j-1])
        x = al_mat[i-1][j-1]+match
        al_mat[i][j]=x
    if j==l2:
        i+=1
        j=1
    else:
        j+=1
    return call(al_mat,i,j)
        