# Social-Media-Link-Prediction-Using-Random-Forest-and-XGBoost-Machine-Learning-Algorithms

## Problem statement: 
Given a directed social graph, predict missing links to recommend users (missing link prediction in a network).

## Data overview
Data  sourced from Facebook's Recruiting Challenge on Kaggle https://www.kaggle.com/c/FacebookRecruiting  

Data contains two columns - source and destination:

    - Data columns (total 2 columns):  
    - source_node         int64  
    - destination_node    int64  


## Business objectives and constraints:  
- No low-latency requirements.
- Probability of prediction is useful to recommend highest probability links first. 

## Formulating a supervised machine learning task:

Given the directed graph, we generat training samples of good and bad links (i.e.existing links and missing links) and for each link we engineer new features such as 'number of followers', 'is user followed back', 'page rank', katz score', 'adar index', some weight features etc. Thereafter, I train Machine Learning models on these features to predict if the link is present or not. 

   References: 
   
    1. https://www.cs.cornell.edu/home/kleinber/link-pred.pdf
    2. https://www3.nd.edu/~dial/publications/lichtenwalter2010new.pdf


## Performance metrics for supervised learning:  
- Both Precision and Recall is important so F1 will be a good choice.
- Confusion matrix. 
- Accuracy can also be reported. 

## My Results & Summary

     
| Sr.no. |     Model     | F1 Score | Accuracy(in %) |
|--------|---------------|----------|----------------|      
|   1.   | Random Forest |  0.9343  |     93.77      |
|   2.   |    XGBoost    |  0.9318  |     93.54      |

## My Approach
        

1. Given a directed Social graph, the task is to predict the missing links in order to recommend users. The dataset consisted of 2 columns -

    column1: source_node (int64)

    column2: destinantion_node(int64)
  

2. Social-media in itself is a giant network in which the connections can be visualized as a graph. I've used NetworkX Python API to create a graph framework from the given source and destination (nodes) data. 


3. Exploratory Data Analysis revealed that - 
    * the dataset consists of nearly 1.86 million distinct Users making a total of 9.43 Million connections with other Users. 
    * On an average each User followed 5 different Users, and on an average each User got followed back by 5 another Users.
    * Majority of the Users had very less followers (99% of Users had 40 or less followers). At least half of the Users had less than 2 followers. 
    * Likewise, Followee stats wasn't much different either. More than half of the Users followed just 1 or 2 other Users, and almost 90% of the Users followed at most 6 other Users.
  

4. Formulating a machine learning task: Our dataset consist of the pair of nodes that are already linked. This can be assigned class label = 1, as there is already a link present between them. In order to convert the given task into a Machine Learning classification problem, we randomly introduced equal number of 'missing links' or 'false links' into the dataset. That is, we randomly generated a pair of nodes from the existing set of all nodes, such that they are not already linked. To these pair of nodes, we assigned class label = 0.


5. Next we combine these postive and negative class label datapoints and split them into train and test in ratio of 80:20 for training and testing respectively. 


6. Before going to build machine learning model, I performed extensive feature engineering and added over 28 new features. This includes: 
    * Set specific features - Jaccard Index, Adar index, Otsuka-Ochiai similarity coefficient, preferential attachment etc. This we implement for both followees and followers. 
    * Graph specific features - Page Rank index, Shortest path length, Katz Centrality, Hubs and Authorities score, etc. These are implemented seperately for source and destination node(as and when applicable).
    * Some other features like - If source and destination node belong to the same community(weakly connected components), is User following back, Weight of the nodes based on followers and followees stats etc. 
  
    Note that most of these above feature we constructed required the knowledge of predecessors and successors of each node in the graph; and this information can be easily obtained by using NetworkX API for Python. 


7. One important thing I noticed in this is that some of the features like HITS Score, Page Rank etc. had exteremly low magnitude (i.e. of order ~ 10^-15 to 10^-20). So these features might slow down computation or may even get ignored internally in tree-based algos. So, in order to make these features meaningful, I decided to take their negative inverse logarithm i.e. - 1/log10(feat) instead of using them directly. 


8. If we decide to include all 18 millions datapoints for building our machine learning model, it would be computationally very expensive. Considering the limited RAM & GPU, I randomly selected 100k datapoints as our Train data from over 15 million Train records, and for test dataset, I randomly selected 50k datapoints from over 3 million Test records in order to build a machine learning model. 


9. As the data we are working on is static, latency requirement isn't a concern as of now. The predictions can be precomputed and stored in a hastable like DS and passed into the UI upon User's login.  


10. For this project, I trained 2 tree-based ensemble models - Random Forest and XGBoost. The performance metric selected was F1 Score. For every model, Accuracy, Confusion matrix, Precision matrix, and Recall was also reported.


11. It was observed that both the models gave nearly similar performances. The feature importance analysis reveals that 'follows_back', which is a binary feature representing whether there is a directed link from destination to source node is the most important feature in our prediction task. Other important features include Jaccard index, Similarity coefficient, Shortest path length, Preferential attachment and Weight features based on followers and followees. 


12. Although our analysis and the subequent modelling was based on the static dataset, in real world, the follower-followee graph is very dynamic in nature. Everyday people follow and unfollow many other Users. This Project helped me set the right foundation to dive into the real-word problem of handling dynamic datasets in future. 
