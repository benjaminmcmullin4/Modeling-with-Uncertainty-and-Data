import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error as mse

class NMFRecommender:
    def __init__(self, random_state=15, rank=2, maxiter=200, tol=1e-3):
        """
        Initialize the NMFRecommender with specified parameters.
        
        Parameters:
            random_state (int): Seed for random number generation.
            rank (int): Number of components in the decomposition.
            maxiter (int): Maximum number of iterations for the update algorithm.
            tol (float): Tolerance to declare convergence.
        """
        self.random_state = random_state
        self.tol = tol
        self.maxiter = maxiter
        self.rank = rank

    def initialize_matrices(self, m, n):
        """
        Initialize the matrices W and H with random numbers.
        
        Parameters:
            m (int): Number of rows.
            n (int): Number of columns.
            
        Returns:
            W (array): Matrix W of shape (m, rank).
            H (array): Matrix H of shape (rank, n).
        """
        np.random.seed(self.random_state)
        self.W = np.random.random((m, self.rank))
        self.H = np.random.random((self.rank, n))
        return self.W, self.H

    def _compute_loss(self, V, W, H):
        """
        Compute the loss of the algorithm using the Frobenius norm.
        
        Parameters:
            V (array): Original matrix.
            W (array): Matrix W.
            H (array): Matrix H.
            
        Returns:
            Loss (float): Frobenius norm of the difference between V and WH.
        """
        return np.linalg.norm(V - W @ H, 'fro')

    def _update_matrices(self, V, W, H):
        """
        Update matrices W and H using the multiplicative update step.
        
        Parameters:
            V (array): Original matrix.
            W (array): Matrix W.
            H (array): Matrix H.
            
        Returns:
            Updated W (array).
            Updated H (array).
        """
        H1 = H * (W.T @ V) / (W.T @ W @ H)
        W1 = W * (V @ H1.T) / (W @ H1 @ H1.T)
        return W1, H1

    def fit(self, V):
        """
        Fit the W and H matrices to the original matrix V.
        
        Parameters:
            V (array): Original matrix.
            
        Returns:
            W (array): Fitted matrix W.
            H (array): Fitted matrix H.
        """
        W, H = self.initialize_matrices(np.shape(V)[0], np.shape(V)[1])

        for _ in range(self.maxiter):
            W1, H1 = self._update_matrices(V, W, H)
            W, H = W1, H1

            if self._compute_loss(V, W1, H1) < self.tol:
                break

        self.W = W
        self.H = H
        return W, H

    def reconstruct(self):
        """
        Reconstruct the original matrix using matrices W and H.
        
        Returns:
            Reconstructed V (array).
        """
        V = self.W @ self.H
        return V

def prob4(rank=2):
    """
    Run NMF recommender on the grocery store example.
    
    Parameters:
        rank (int): Rank for NMF.
        
    Returns:
        W (array): Matrix W.
        H (array): Matrix H.
        Peeps (int): Number of people with higher component 2 than component 1 scores.
    """
    V = np.array([[0,1,0,1,2,2],
                  [2,3,1,1,2,2],
                  [1,1,1,0,1,1],
                  [0,2,3,4,1,1],
                  [0,0,0,0,1,0]])

    nmf = NMFRecommender(rank=rank)
    W, H = nmf.fit(V)

    peeps = 0
    for person in range(np.shape(H)[1]):
        if H[:, person][1] > H[:, person][0]:
            peeps += 1

    return W, H, peeps

def prob5(filename='artist_user.csv'):
    """
    Find the optimal rank for NMF and reconstruct the original matrix V.
    
    Parameters:
        filename (str): Name of the file containing artist user data.
        
    Returns:
        rank (int): Optimal rank for NMF.
        V (array): Reconstructed matrix V.
    """
    df = pd.read_csv("artist_user.csv", index_col=0)
    bench = np.linalg.norm(df, 'fro') * .0001

    rank = 3
    while True:
        nmf = NMF(n_components=rank, init='random', random_state=0)
        W = nmf.fit_transform(df)
        H = nmf.components_
        V = W @ H

        if np.sqrt(mse(df, V)) < bench:
            break

        rank += 1

    return rank, V

def discover_weekly(user_id, V):
    """
    Create recommended weekly list for a given user.
    
    Parameters:
        user_id (int): User ID.
        V (array): Reconstructed array.
        
    Returns:
        Recommended list (list): List of recommended items.
    """
    df = pd.read_csv("artist_user.csv", index_col=0)
    df.rename(columns=pd.read_csv('artists.csv').astype(str).set_index('id').to_dict()['name'], inplace=True)
    user_all = df.loc[user_id, :].reset_index()
    user_all.rename(columns={'index': 'artist', user_id: 'listens'}, inplace=True)

    user_all['weights'] = V[user_id - 2]
    user_all = user_all[user_all['listens'] == 0]
    user_all.sort_values(by='weights', ascending=False, inplace=True)

    return user_all['artist'].tolist()[:30]

if __name__ == "__main__":
    pass

    # W, H, peeps = prob4()
    # print(f"Number of people with higher component 2 than component 1 scores: {peeps}")

    # rank, V = prob5()
    # print(f"Optimal rank for NMF: {rank}")

    # recommended = discover_weekly(2, V)
    # print(f"Recommended list for user 2: {recommended}")