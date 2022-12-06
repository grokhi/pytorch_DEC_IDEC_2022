import torch
import torch.nn as nn

from sklearn.cluster import KMeans

class ClusterAssignment(nn.Module):
    def __init__(
        self,
        num_cluster: int,
        hidden_dim: int,
        alpha: float = 1.0,
        centroids: torch.tensor = None,
    ):
        """ 
        student t-distribution, as same as used in t-SNE algorithm.
                    q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            num_cluster: 
            hidden_dim: 
            alpha:
            centroids:
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        super().__init__()
        self.num_cluster = num_cluster
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.centroids = centroids
        
        if centroids is None:
            centroids = torch.zeros(
                self.num_cluster, self.hidden_dim, dtype=torch.float
            )
            nn.init.xavier_uniform_(centroids)

        self.centroids = nn.Parameter(centroids)


    def forward(self, z):

        diff = torch.sum((z.unsqueeze(1) - self.centroids) ** 2, 2)
        numerator = 1.0 / (1.0 + (diff / self.alpha))
        power = (self.alpha + 1.0) / 2
        numerator = numerator ** power
        q = numerator / torch.sum(numerator, dim=1, keepdim=True)
        return q


class DEC(nn.Module):
    def __init__(
        self, 
        autoencoder: nn.Module,
        n_clusters: int = 10,
        alpha: float = 1.0,
        centroids = None,
    ):
        '''
        Deep Embedded Clustering
        '''
        super().__init__()
        self.encoder = autoencoder.encoder
        self.n_clusters = n_clusters
        self.hidden_dim = autoencoder.hid_dim
        self.alpha = alpha

        self.assignment = ClusterAssignment(
            n_clusters, autoencoder.hid_dim, alpha, centroids
        )
        self.kmeans = KMeans(n_clusters, n_init=20)
        self.initialized = False
        

    def forward(self, x):
        return self.assignment(self.encoder(x))

    def get_target_distribution(self, q):
        numerator = (q ** 2) / torch.sum(q, 0)
        p = (numerator.t() / torch.sum(numerator, 1)).t()
        return p


class IDEC(DEC):
    def __init__(
        self, 
        autoencoder: nn.Module,
        n_clusters: int = 10,
        alpha: float = 1.0,
    ):    
        super().__init__(
            autoencoder,
            n_clusters,
            alpha,
        ) 
        self.decoder = autoencoder.decoder
   
    
    def forward(self, x):
        return(
            self.assignment(self.encoder(x)),
            self.decoder(self.encoder(x))
        )
    


class IDEC_loss(nn.Module):
    def __init__(self, gamma=.1):
        super().__init__();
        self.gamma = gamma

    def forward(self, pred_cl, targ_cl, pred_rec, targ_rec, ):
        """
        See original paper for explanation 
        """
        loss_cl = nn.KLDivLoss(reduction='batchmean')
        loss_rec = nn.MSELoss()
        self.cl_loss = self.gamma * loss_cl(pred_cl, targ_cl) #value
        self.rec_loss = loss_rec(pred_rec, targ_rec) #value
        
        return self.rec_loss + self.cl_loss
    
