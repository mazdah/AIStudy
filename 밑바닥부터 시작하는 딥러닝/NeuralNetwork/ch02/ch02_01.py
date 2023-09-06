import sys

from matplotlib import pyplot as plt

sys.path.append('..')
import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity, most_similar, ppmi, create_contexts_target, convert_one_hot

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

print(corpus)
print(id_to_word)

# text에 대한 동시발생 행렬을 수작업으로 작성
C = np.array([
    [0, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0]
], dtype=np.int32)

print('== 단어 벡터 출력 ======================================================')
print(C[0])
print(C[4])
print(C[word_to_id['goodbye']])

print('')
print('== 동시발생 행렬 출력 ======================================================')
co_matrix = create_co_matrix(corpus, 7, 1)
print(co_matrix)

print('')
print('== 단어 벡터 유사도 출력 ======================================================')
c0 = C[word_to_id['you']]
c1 = C[word_to_id['i']]
print(cos_similarity(c0, c1))

print('')
print('== 유사 단어 랭킹 출력 ======================================================')
most_similar('you', word_to_id, id_to_word, C, top=5)

print('')
print('== PPMI 행렬 출력 ======================================================')
W = ppmi(C)
print(C)
print('-'*50)
print('PPMI')
print(W)

print('')
print('== SVD 출력 ======================================================')
U, S, V = np.linalg.svd(W)
print(C[0])
print(W[0])
print(U[0])
print(U[0, :2])

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()

print('')
print('== 맥락과 타겟 출력 ======================================================')
contexts, target = create_contexts_target(corpus, window_size=1)
print(contexts)
print(target)

print('')
print('== 맥락과 타겟을 원핫인코딩으로 출력 ======================================================')
target = convert_one_hot(target, len(word_to_id))
contexts = convert_one_hot(contexts, len(word_to_id))
print(target)
print(contexts)