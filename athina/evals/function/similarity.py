import math
from abc import ABC, abstractmethod

def _tokenize(string):
    return string.lower().split()

def _create_combined_set(string1, string2):
    return set(_tokenize(string1)).union(set(_tokenize(string2)))

def _vectorize(string, combined_set):
    vector = []
    for word in combined_set:
        vector.append(string.count(word))
    return vector

def _normalised_levenshtein_distance(str1, str2):
    m, n = len(str1), len(str2)
    # Create a matrix to store the distances
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Initialize the first row and first column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    # Calculate the distance
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i]
                                   [j - 1], dp[i - 1][j - 1])
    if (len(str1) >= len(str2)): 
        return dp[m][n] / len(str1);
    else:
        return dp[m][n] / len(str2);

def normalised_levenshtein_similarity(str1, str2):
    return 1 - _normalised_levenshtein_distance(str1, str2)


class Comparator(ABC):
    @abstractmethod
    def compare(self, string1, string2):
        pass

class CosineSimilarity(Comparator):

    def compare(string1, string2):
        # Tokenize and create a combined set of unique words
        combined_set = _create_combined_set(string1, string2)
        # Vectorize the strings
        vector1 = _vectorize(string1, combined_set)
        vector2 = _vectorize(string2, combined_set)
        dot_product = sum(p*q for p, q in zip(vector1, vector2))
        magnitude_vec1 = math.sqrt(sum([val**2 for val in vector1]))
        magnitude_vec2 = math.sqrt(sum([val**2 for val in vector2]))
        if magnitude_vec1 * magnitude_vec2 == 0:
            # Avoid division by zero
            return 0
        return dot_product / (magnitude_vec1 * magnitude_vec2)

# Example strings
string1 = "This is a sample string."
string2 = "This is another sample string."

# Compute cosine similarity
cosine_similarity = cosine_similarity(string1, string2)
print("cosine_similarity")
print(cosine_similarity)

# Compute levenshtein distance
normalised_levenshtein_similarity = normalised_levenshtein_similarity(string1, string2)
print("normalised_levenshtein_similarity")
print(normalised_levenshtein_similarity)