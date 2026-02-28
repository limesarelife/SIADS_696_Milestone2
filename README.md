**Authors**   
Jacqueline Skunda, Tamara Qawasmeh, Malini Varadarajan

**Project Overview/Motivation**    
Stranger Things is a Netflix series with a popular social media following. Our main goal for our project is to utilize various Supervised and Unsupervised machine learning approaches and build an algorithm that can identify which character in the series said what in regards to the Stranger Things scripts/teleplays. We also plan to use the r/StrangerThings subreddit data (titles, post self text, and comments associated with posts) to identify topics; topic modeling will be done via unsupervised clustering methods.  To note we have no stakeholders involved in this project as we are doing this out of our pure curiosity and fascination with the Netflix series and are fans ourselves.

**Supervised Learning**     
Main & External Datasets:    
We are using PDF teleplays from the website 8flix (https://8flix.com). Scripts for each season and every episode was downloaded from the above website. We also used Wikipedia to get the main and recurring character names in order to only keep lines spoken by main and/or recurring characters so the classifier will have a more balanced representation of each character.

Supervised Learning Approaches and Feature Representations:     
“Who said What” Analysis: Based on our engineered feature set, we will apply supervised learning methods to help predict which Stranger Things character said what.  This is a multi-classification task to predict the character who spoke based on speech characteristics from lines spoken dialog.  The feature representations that are important to this problem are having the scripts for every episode in every season. We will be using the library PDFMiner in order to read in and help parse the scripts.  For every line spoken in the series, we tag the character that speaks the line and we use this data in order to train the classifier properly.  For example, when speaking at said moment, a character could say one sentence or four sentences but regardless that is their said spoken line. Therefore, a character line refers to when they are speaking in a script but does not imply only one phrase or sentence being spoken, it could be one word or multiple sentences. Besides having the character name in their text spoken as a feature, having the season and episode number for the line spoken might also prove to be helpful. We will be creating a dictionary using the name of each character as a key and adding lines as the value for every time a character spoke. The words/lines spoken will be added as a list for said character’s key then the dictionary will be converted to a dataframe after going through each episode per season, combining into one large dataframe. Each list will be exploded onto its own row to retrieve the intended character spoken.  As mentioned above we will be using only recurring or main characters to a season and series overall as to help with issues where one character might appear in one episode and only speak three lines in total. Besides word representations, it could be helpful to generate other features such as sentence readability, word familiarity/difficulty or character traits/details could be added if needed. 

We plan to try these various supervised learning classification techniques:     
- Logistic Regression: to understand the strength of relationships by leveraging signals to approximate the most appropriate classification.  This will be our baseline for classification.
- Random Forest Classifier: to perform classification tasks on noisy and/or highly dimensional data by making use of an ensemble of decision trees for classification.
- LinearSVM based regression: generalization capability of the words/lines with high prediction accuracy for character name, robust to outliers. Will need to transform all lines of text into vectors in order to encode and leverage the power of SVM.

Evaluation Metrics:    
- F1 Score : mean of precision and recall scores
- Precision : explains how many of the correctly predicted cases actually turned out to be positive. Precision is useful  where False Positives are a higher concern than False Negatives.
- AUC (Area Under the Curve): to measure how well the classifier was able to distinguish different classes, if our dataset proves to be imbalanced which might be the case since characters do speak different volumes of lines etc.
- K-Fold cross validation accuracy: to measure and test if our various models' accuracy would be better, worse or the same if we had used a different section of the data set as a validation set. Evaluation is performed using different shuffling and chunking of the dataset through various iterations. This will be great because we can see the various scoring/evaluation metrics above at once (such as accuracy, precision, recall and f1)

Visualizations:    
- Confusion matrix: in order to compare actual label versus predicted label (TP, TN, FP, FN) and identify proper (best) evaluation metrics/scoring to use. 
- Histograms: to show frequency of characters predicted (by probability with use of bins)
- We will try to inspect model performance via visualizations between different characters (classes).
- Line charts for visualizing AUC-ROC and other results/metrics.
- All visualizations will keep the Stranger Things aesthetic as well.  Will look into building a visualization which is composed of characters face/faces with model output probabilities showing the confidence prediction.  Or sentiment of characters lines spoken, we are sure there are characters which have more negative sentiment overall than others.  We will scope out over the course of project work. 

**Unsupervised learning**    
Main & External Datasets:      
We will be using a Reddit Bot we create to retrieve the various Stranger Things r/StrangerThings subreddit posts using the PRAW (The Python API Wrapper) library. We will be basically generating our own dataset in order to do the topic modeling (clustering).

Unsupervised Learning Approaches and Feature Representations:      
The goal for our unsupervised learning is to extract context and representation of what reddit users in the r/StrangerThings subreddit thread are talking about and predict groupings/topics based on the subreddit self-text, and comments. The text will be used as a feature for clustering to uncover topics within the Stranger Things Subreddit community. The data manipulation required is extraction of the subreddit text using the Reddit API and text clean up using regex and other pattern finding techniques. Removing non-alphanumeric characters and non-breaking space signifiers, newline characters, etc. Rows containing [deleted] and posts with pictures will also be removed. Standard NLP wlil be applied to remove stopwords, tokenize, and lemmatize the data.  We will also utilize BERT besides TF-IDF and Word2Vec to see which best represents the data for handling loss of context.

We plan to try these various supervised learning classification techniques:        
- K-means clustering: unsupervised classification algorithm that groups objects into k groups based on their characteristics, we need to figure out the ideal number of clusters so this might have some wild results initially until tuning since it holds the static number of k clusters and keeps moving the the members of a group around via recalculation of cluster centroids until reassignment stops between clusters.
- LDA (Latent Dirichlet Allocation): LDA is a three-level hierarchical Bayesian model and an example of a topic model. Will look to create word embedding vectors to handle any sparse or high dimensional features.
- HCA (Hierarchical Cluster Analysis): we will try Agglomerative first which is bottom up where each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy. Wards will be used for the method parameter first, then complete and average and compare but we believe Ward would be best.
- LSA will be utilized for dimensionality reduction (if needed).
* Note we might switch out K Means for NMF (Non negative Matrix Factorization)

Evaluation Metrics:      
- Elbow Method: for clustering in order to find/measure the optimal k by using the WSS (the compactness of the clusters value) based on the number of clusters, where the bend occurs is a marker for the ideal number of clusters.
- Silhouette Score: to see how distinct the clusters are in space and relation to one another and their quality via a bounded score.
- CH Index (Calinski Harabasz Index): to evaluate stability of the resulting clusters per model.  A higher score will show which model performed best. Score is not bounded like the silhouette score.
- DBI (Davies-Bouldin index): great score for measuring split between clusters, a lower average similarity score means the clusters are separated and a good indication of better performance.

Visualization (for evaluation metrics as well):     
- Line Chart: to visualize the Elbow method for the k means clustering in order to find the optimal k by using the WSS (the compactness of the clusters value) based on the number of clusters, where the bend occurs is a marker for the ideal number of clusters. (Will be used for both K means and HCA)
- Cluster Dendrogram: to visualize the number of clusters and their connections (stems)
- Scatter matrix of all features with color coding for easier to be seen representations.
- Bubble Charts, Scatter plots to visualize clusters and topics, we will attempt a word cloud to display top keywords per topic but this did take up valuable space in our previous milestone project (different project topic) in the final report and was removed. Will look into t-sne for visualization possibilities as well.
- All visualizations will keep the Stranger Things aesthetic as well.  


**Challenges and Limitations**    
For both datasets we will be cautious to watch out for and examine imbalances and adjust models accordingly.

“Who said What” Analysis: With the manipulation and cleaning of text there are always challenges but we will do our best to gather and clean the scripts as best as possible. Given that the  scripts will output as PDF files, and given that their encodings are hard to parse, we will rely on regex and other pattern finding techniques to gather the lines spoken by a specific character. Also another concern, characters might tend to speak in short sentences, and the vocabulary of the kids in the series could be very different from Hopper's or Joyce’s complexity. We would think this should help the classification process but will be sure to see if it has negative consequences which should be handled.

“Topic Modeling” Subreddit Analysis: The dataset itself is completely generated based on returns from our bot via use of the PRAW library and based on posts in the subreddit community therefore we need to remove posts which may have title text but no self text. These are most likely images (such as drawings or memes) that were posted, these could pose issues when collecting enough data (aka posts with actual text) since many posts can be pictures, memes, gifs etc.  Also because we are only using the main subreddit for Stranger Things, we need to be diligent to make sure to test for the ideal number of clusters.  It also can be hard to distinguish between what scale is being discussed in the post. It can be hard to measure the generality since we are only generating insights based on text/comments within the single subreddit. The only context we have besides knowing the subreddit will be the concept applied after the fact to fit the data, leading to potential unknown biases.

  
