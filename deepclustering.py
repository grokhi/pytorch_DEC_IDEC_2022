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
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        Arguments:
            num_cluster (int): number of clusters
            hidden_dim (int): dimension of bottleneck layer 
            alpha (float): parameter representing the degrees of freedom in the t-distribution, default 1.0 
            centroids: clusters centers to initialise, if None then use Xavier uniform

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
    '''
    DEC algorithm implementation. 
    Read Unsupervised Deep Embedding for Clustering Analysis (2016) by Junyuan Xie et al. for details.

    Arguments:
        autoencoder (nn.Module): autoencoder to use
        n_clusters (int): number of clusters
        alpha (float): parameter representing the degrees of freedom in the t-distribution, default 1.0 
        centroids: clusters centers to initialise

    Main attributes:
        encoder (nn.Module): pretrained encoder which will be used for cluster assignment
        assignment (nn.Module): soft cluster assignment with shape == (batch_size, hid_dim, n_clusters)
        kmeans (KMeans): need for initial cluster initialization

    Methods:
        get_target_distribution: get t-disributiion as described in Xie et al. (2016)
        forward: compute cluster assignment

    Return:
        Soft cluster assignment (batch_size, hid_dim, n_clusters)
    '''
    def __init__(
        self, 
        autoencoder: nn.Module,
        n_clusters: int = 10,
        alpha: float = 1.0,
        centroids = None,
    ):
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
        '''
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.

        :x: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        '''
        return self.assignment(self.encoder(x))

    def get_target_distribution(self, q):
        numerator = (q ** 2) / torch.sum(q, 0)
        p = (numerator.t() / torch.sum(numerator, 1)).t()
        return p


class IDEC(DEC):
    '''
    IDEC algorithm implementation.
    Read "Improved Deep Embedded Clustering with Local Structure Preservation" (2017) by Xifeng Guo et al. for details.

    Arguments:
        autoencoder (nn.Module): autoencoder to use
        n_clusters (int): number of clusters
        alpha (float): parameter representing the degrees of freedom in the t-distribution, default 1.0 
        centroids: clusters centers to initialise

    Main attributes:
        encoder (nn.Module): pretrained encoder used for minimization of clustering loss 
        decoder (nn.Module): pretrained decoder used for minimization of reconstruction loss 
        assignment (nn.Module): 
        kmeans (KMeans): need for initial cluster initialization

    Methods:
        get_target_distribution: get t-disributiion as described in Xie et al. (2016)
        forward: compute cluster assignment

    Returns:
        forward: (soft cluster assignment, reconstructed embeddings)
    '''    
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
    '''
    IDEC loss used for optimization of DEC algorithm.
    Read "Improved Deep Embedded Clustering with Local Structure Preservation" (2017) by Xifeng Guo et al. for details.

    '''
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
    
