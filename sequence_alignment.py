import numpy as np
import random

def gen_s(s=list('AAAAAACCCCCCGGTT')):
    s1 = []
    num = list(range(16))
    for i in range(16):
        c = random.choice(num)
        s1.append(s[c])
        num.remove(c)
    return s1

in_ch = input("<m> for manual string input... ")
if in_ch == "m":
    s1 = input()
    s2 = input()
else:
    s1 = gen_s()
    s2 = gen_s()

match = 5
mismatch = 4

print("Sequence 1:", "".join(s1))
print("Sequence 2:", "".join(s2))
l1, l2 = 16, 16
al_mat = np.zeros((l1 + 1, l2 + 1), dtype=int)


def call(al_mat, i=1, j=1):
    if i == l1 + 1:
        return al_mat
    if j == l2 + 1:
        return call(al_mat, i + 1, 1)

    if s1[i - 1] != s2[j - 1]:
        x = max(al_mat[i, j - 1], al_mat[i - 1, j], al_mat[i - 1, j - 1])
        al_mat[i, j] = x - mismatch
    else:
        x = al_mat[i - 1, j - 1] + match
        al_mat[i, j] = x

    return call(al_mat, i, j + 1)


def traceback(al_mat, i, j, al_seq1, al_seq2):
    if i == 0 and j == 0:
        return al_seq1[::-1], al_seq2[::-1]

    if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
        al_seq1 += s1[i - 1]
        al_seq2 += s2[j - 1]
        return traceback(al_mat, i - 1, j - 1, al_seq1, al_seq2)
    elif j > 0 and al_mat[i, j] == al_mat[i, j - 1] - mismatch:
        al_seq1 += '-'
        al_seq2 += s2[j - 1]
        return traceback(al_mat, i, j - 1, al_seq1, al_seq2)
    elif i > 0 and al_mat[i, j] == al_mat[i - 1, j] - mismatch:
        al_seq1 += s1[i - 1]
        al_seq2 += '-'
        return traceback(al_mat, i - 1, j, al_seq1, al_seq2)
    if i > 0:
        al_seq1 += s1[i - 1]
        al_seq2 += '-'
        return traceback(al_mat, i - 1, j, al_seq1, al_seq2)
    elif j > 0:
        al_seq1 += '-'
        al_seq2 += s2[j - 1]
        return traceback(al_mat, i, j - 1, al_seq1, al_seq2)


def traceback_route(al_mat, al_seq1, al_seq2):
    route = np.full(al_mat.shape, " ")
    i, j = l1, l2
    route[i][j] = "*"

    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            i -= 1
            j -= 1
            route[i][j] = "*"
        elif j > 0 and al_mat[i][j] == al_mat[i][j - 1] - mismatch:
            j -= 1
            route[i][j] = "*"
        elif i > 0 and al_mat[i][j] == al_mat[i - 1][j] - mismatch:
            i -= 1
            route[i][j] = "*"
        elif i > 0:
            i -= 1
            route[i][j] = "*"
        elif j > 0:
            j -= 1
            route[i][j] = "*"
    for row in route:
        print(" ".join(row))


alignment_matrix = call(al_mat)

print("Alignment Matrix:")
print(alignment_matrix)

al_seq1, al_seq2 = traceback(alignment_matrix, l1, l2, "", "")
print("Aligned Seq1:", al_seq1)
print("Aligned Seq2:", al_seq2)

print("Traceback Route:")
traceback_route(al_mat, al_seq1, al_seq2)
