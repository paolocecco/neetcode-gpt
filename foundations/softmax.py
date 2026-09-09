import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        
        sum = np.sum(np.exp(z - np.max(z)))

        expArr = np.exp(z - np.max(z)) / sum

        return np.round(expArr, 4)
