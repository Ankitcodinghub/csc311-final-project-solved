# csc311-final-project-solved
**TO GET THIS SOLUTION VISIT:** [CSC311 Final Project Solved](https://www.ankitcodinghub.com/product/csc311-final-project-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;91522&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;5&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (5 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSC311 Final Project Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (5 votes)    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
<ol start="0">
<li style="list-style-type: none;">
<ol start="0">
<li>1 Introduction
One of CSC311’s main objectives is to prepare you to apply machine learning algorithms to real- world tasks. The final project aims to help you get started in this direction. You will be performing the following tasks:

• Try out existing algorithms to real-world tasks.

• Modify an existing algorithm to improve performance. • Write a short report analyzing the result.

The final project is not intended to be a stressful experience. It is a good chance for you to experiment, think, play, and hopefully have fun. These tasks are what you will be doing daily as a data analyst/scientist or machine learning engineer.
</li>
</ol>
</li>
</ol>
2 Background &amp; Task

Online education services, such as Khan Academy and Coursera, provide a broader audience with access to high-quality education. On these platforms, students can learn new materials by watching a lecture, reading course material, and talking to instructors in a forum. However, one disadvantage of the online platform is that it is challenging to measure students’ understanding of the course material. To deal with this issue, many online education platforms include an assessment component to ensure that students understand the core topics. The assessment component is often composed of diagnostic questions, each a multiple choice question with one correct answer. The diagnostic question is designed so that each of the incorrect answers highlights a common misconception. An example of the diagnostic problem is shown in figure 1. When students incorrectly answer the diagnostic question, it reveals the nature of their misconception and, by understanding these misconceptions, the platform can offer additional guidance to help resolve them.

1https://markus.teach.cs.toronto.edu/csc311-2021-09

</div>
</div>
<div class="layoutArea">
<div class="column">
Final Project

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
Figure 1: An example diagnostic question [1].

In this project, you will build machine learning algorithms to predict whether a student can correctly answer a specific diagnostic question based on the student’s previous answers to other questions and other students’ responses. Predicting the correctness of students’ answers to as yet unseen diagnostic questions helps estimate the student’s ability level in a personalized education platform. Moreover, these predictions form the groundwork for many advanced customized tasks. For in- stance, using the predicted correctness, the online platform can automatically recommend a set of diagnostic questions of appropriate difficulty that fit the student’s background and learning status.

You will begin by applying existing machine learning algorithms you learned in this course. You will then compare the performances of different algorithms and analyze their advantages and dis- advantages. Next, you will modify existing algorithms to predict students’ answers with higher accuracy. Lastly, you will experiment with your modification and write up a short report with the results.

You will measure the performance of the learning system in terms of prediction accuracy, although you are welcome to include other metrics in your report if you believe they provide additional insight:

Prediction Accuracy = The number of correct predictions The number of total predictions

3 Data

We subsampled answers of 542 students to 1774 diagnostic questions from the dataset provided by Eedi2, an online education platform that is currently being used in many schools [1]. The platform offers crowd-sourced mathematical diagnostic questions to students from primary to high school (between 7 and 18 years old). The truncated dataset is provided in the folder /data.

2 https://eedi.com/

</div>
</div>
<div class="layoutArea">
<div class="column">
2

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
CSC311 Fall 2021

</div>
<div class="column">
Final Project

</div>
</div>
<div class="layoutArea">
<div class="column">
3.1 Primary Data

</div>
</div>
<div class="layoutArea">
<div class="column">
Figure 2: An example sparse matrix [1].

</div>
</div>
<div class="layoutArea">
<div class="column">
The primary data, train_data.csv, is the main dataset you will be using to train the learning algorithms throughout the project. There is also a validation set valid_data.csv that you should use for model selection and a test set test_data.csv that you should use for reporting the final performance. All primary data csv files are composed of 3 columns:

<ul>
<li>question id: ID of the question answered (starts from 0).</li>
<li>user id: ID of the student who answered the question (starts from 0).</li>
<li>is correct: Binary indicator whether the student’s answer was correct (0 is incorrect, 1 is correct).
We also provide a sparse matrix, sparse_matrix.npz, where each row corresponds to the user id and each column corresponds to the question id. An illustration of the sparse matrix is shown in figure 2. The correct answer given a pair of (user id, question id) will have an entry 1 and an incorrect answer will have an entry 0. Answers with no observation and held-out data (that will be used for validation and test) will have an entry NaN (np.NaN).
</li>
</ul>
3.2 Question Metadata

We also provide the question metadata, question_meta.csv, which contains the following columns:

</div>
</div>
<div class="layoutArea">
<div class="column">
• •

3.3

</div>
<div class="column">
question id: ID of the question answered (starts from 0).

subject id: The subject of the question covered in an area of mathematics. The text de-

scription of each subject is provided in subject_meta.csv. Student Metadata

</div>
</div>
<div class="layoutArea">
<div class="column">
Lastly, we provide the student metadata, student_meta.csv, that is composed of the following columns:

</div>
</div>
<div class="layoutArea">
<div class="column">
3

</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
CSC311 Fall 2021 Final Project

</div>
</div>
<div class="layoutArea">
<div class="column">
<ul>
<li>user id: ID of the student who answered the question (starts from 0).</li>
<li>gender: Gender of the student, when available. 1 indicates a female, 2 indicates a male, and
0 indicates unspecified.
</li>
<li>data of birth: Birth date of the student, when available.</li>
<li>premium pupil: Student’s eligibility for free school meals or pupil premium due to being financially disadvantaged, when available.
4 PartA

In the first part of the project, you will implement and apply various machine learning algorithms you studied in the course to predict students’ correctness of a given diagnostic question. Review the course notes if you don’t recall the details of each algorithm. For this part, you will only be using the primary data: train_data.csv, sparse_matrix.npz, valid_data.csv, and test_data.csv. Moreover, you may use the helper functions provided in utils.py to load the dataset and evaluate your model. You may also use any functions from packages NumPy, Scipy, Pandas, and PyTorch. Make sure you understand the code instead of using it as a black box.
</li>
</ul>
<ol>
<li>[5pts] k-Nearest Neighbor. In this problem, using the provided code at part_a/knn.py, you will experiment with k-Nearest Neighbor (kNN) algorithm.
The provided kNN code performs collaborative filtering that uses other students’ answers to predict whether the specific student can correctly answer some diagnostic questions. In particular, the starter code implements user-based collaborative filtering: given a user, kNN finds the closest user that similarly answered other questions and predicts the correctness based on the closest student’s correctness. The core underlying assumption is that if student A has the same correct and incorrect answers on other diagnostic questions as student B, A’s correctness on specific diagnostic questions matches that of student B.

<ol>
<li>(a) &nbsp;Complete a function main located at knn.py that runs kNN for different values of k ∈ {1,6,11,16,21,26}. Plot and report the accuracy on the validation data as a function of k.</li>
<li>(b) &nbsp;Choose k∗ that has the highest performance on validation data. Report the chosen k∗ and the final test accuracy.</li>
<li>(c) &nbsp;Implement a function knn_impute_by_item on the same file that performs item-based collaborative filtering instead of user-based collaborative filtering. Given a question, kNN finds the closest question that was answered similarly, and predicts the correctness basted on the closest question’s correctness. State the underlying assumption on item- based collaborative filtering. Repeat part (a) and (b) with item-based collaborative filtering.</li>
<li>(d) &nbsp;Compare the test performance between user- and item- based collaborative filtering. State which method performs better.</li>
<li>(e) &nbsp;List at least two potential limitations of kNN for the task you are given.</li>
</ol>
</li>
<li>[15pts] Item Response Theory. In this problem, you will implement an Item-Response Theory (IRT) model to predict students’ correctness to diagnostic questions.</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
4

</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="layoutArea">
<div class="column">
CSC311 Fall 2021 Final Project

</div>
</div>
<div class="layoutArea">
<div class="column">
The IRT assigns each student an ability value and each question a difficulty value to formulate a probability distribution. In the one-parameter IRT model, βj represents the difficulty of the question j, and θi that represents the i-th students ability. Then, the probability that the question j is correctly answered by student i is formulated as:

p(cij=1|θi,βj)= exp(θi−βj) 1+exp(θi −βj)

We provide the starter code in part_a/item_response.py.

<ol>
<li>(a) &nbsp;Derive the log-likelihood logp(C|θ,β) for all students and questions. Here C is the sparse matrix. Also, show the derivative of the log-likelihood with respect to θi and βj (Hint: recall the derivative of the logistic model with respect to the parameters).</li>
<li>(b) &nbsp;Implement missing functions in item_response.py that performs alternating gradient descent on θ and β to maximize the log-likelihood. Report the hyperparameters you selected. With your chosen hyperparameters, report the training curve that shows the training and validation log-likelihoods as a function of iteration.</li>
<li>(c) &nbsp;With the implemented code, report the final validation and test accuracies.</li>
<li>(d) &nbsp;Select three questions j1,j2, and j3. Using the trained θ and β, plot three curves on the same plot that shows the probability of the correct response p(cij = 1) as a function of θ given a question j. Comment on the shape of the curves and briefly describe what these curves represent.</li>
</ol>
3. [15pts] Matrix Factorization OR Neural Networks. In this question, please read both option (i) and option (ii), but you only need to do one of the two.

(i) Option 1: Matrix Factorization. In this problem, you will be implementing matrix factorization methods. The starter code is located at part_a/matrix_factorization.

<ol>
<li>(a) &nbsp;Using a function svd_reconstruct that factorizes the sparse matrix using singular- value decomposition, try out at least 5 different k and select the best k using the validation set. Report the final validation and test performance with your chosen k.</li>
<li>(b) &nbsp;State one limitation of SVD in the task you are given. (Hint: how are you treating the missing entries?)</li>
<li>(c) &nbsp;Implement functions als and update_u_z located at the same file that performs alternating updates. You need to use SGD to update un ∈ Rk and zm ∈ Rk as described in the docstring. To be noted, this is different from the alternating least squares (ALS) we introduced in the course slides, where we used the direct solution. As a reminder, the objective is as follows:
1􏰃􏰆 ⊤􏰇2

Cnm − un zm ,

where C is the sparse matrix and O = {(n, m) : entry (n, m) of matrix C is observed}.
</li>
<li>(d) &nbsp;Learn the representations U and Z using ALS with SGD. Tune learning rate and number of iterations. Report your chosen hyperparameters. Try at least 5 different values of k and select the best k∗ that achieves the lowest validation accuracy.</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
min U,Z 2

</div>
<div class="column">
(n,m)∈O

</div>
</div>
<div class="layoutArea">
<div class="column">
5

</div>
</div>
</div>
<div class="page" title="Page 6">
<div class="layoutArea">
<div class="column">
CSC311 Fall 2021 Final Project

</div>
</div>
<div class="layoutArea">
<div class="column">
(e) With your chosen k∗, plot and report how the training and validation squared-error- losses change as a function of iteration. Also report final validation accuracy and test accuracy.

(ii) Option 2: Neural Networks. In this problem, you will implement neural networks to predict students’ correctness on a diagnostic question. Specifically, you will design an autoencoder model. Given a user v ∈ RNquestions from a set of users S, our objective is:

min 􏰃 ∥v − f(v; θ)∥2,

θ

v∈S

where f is the reconstruction of the input v. The network computes the following

function:

f(v; θ) = h(W(2)g(W(1)v + b(1)) + b(2)) ∈ RNquestions

for some activation functions h and g. In this question, you will be using sigmoid activation functions for both. Here, W(1) ∈ Rk×Nquestions and W(2) ∈ RNquestions×k, where k ∈ N is the latent dimension. We provide the starter code written in PyTorch at part_a/neural_network.

<ol>
<li>(a) &nbsp;Describe at least three differences between ALS and neural networks.</li>
<li>(b) &nbsp;Implement a class AutoEncoder that performs a forward pass of the autoencoder following the instructions in the docstring.</li>
<li>(c) &nbsp;Train the autoencoder using latent dimensions of k ∈ {10, 50, 100, 200, 500}. Also, tune optimization hyperparameters such as learning rate and number of iterations. Select k∗ that has the highest validation accuracy.</li>
<li>(d) &nbsp;With your chosen k∗, plot and report how the training and validation objectives changes as a function of epoch. Also, report the final test accuracy.</li>
<li>(e) &nbsp;Modify a function train so that the objective adds the L2 regularization. The objective is as follows:
min 􏰃 ∥v − f(v; θ)∥2 + λ(∥W(1)∥2F + ∥W(2)∥2F ) θ v∈S 2

You may use a method get_weight_norm to obtain the regularization term. Using the k and other hyperparameters selected from part (d), tune the regularization penalty λ ∈ {0.001, 0.01, 0.1, 1}. With your chosen λ, report the final validation and test accuracy. Does your model perform better with the regularization penalty?
</li>
</ol>
4. [15pts] Ensemble. In this problem, you will be implementing bagging ensemble to im- prove the stability and accuracy of your base models. Select and train 3 base models with bootstrapping the training set. You may use the same or different base models. Your imple- mentation should be completed in part_a/ensemble.py. To predict the correctness, generate 3 predictions by using the base model and average the predicted correctness. Report the final validation and test accuracy. Explain the ensemble process you implemented. Do you obtain better performance using the ensemble? Why or why not?

</div>
</div>
<div class="layoutArea">
<div class="column">
6

</div>
</div>
</div>
<div class="page" title="Page 7">
<div class="layoutArea">
<div class="column">
CSC311 Fall 2021 Final Project

</div>
</div>
<div class="layoutArea">
<div class="column">
5 PartB

In the second part of the project, you will modify one of the algorithms you implemented in part A to hopefully predict students’ answers to the diagnostic question with higher accuracy. In particular, consider the results obtained in part A, reason about what factors are limiting the performance of one of the methods (e.g. overfitting? underfitting? optimization difficulties?) and come up with a proposed modification to the algorithm which could help address this problem. Rigorously test the performance of your modified algorithm, and write up a report summarizing your results as described below.

You will not be graded on how well the algorithm performs (i.e. its accuracy); rather, your grade will be based on the quality of your analysis. Try to be creative! You may also optionally use the provided metadata (question_meta.csv and student_meta.csv) to improve the accuracy of the model. At last, you are free to use any third-party ideas or code as long as it is publicly available. You must properly provide references to any work that is not your own in the write-up.

The length of your report for part B should be 3-4 pages. Don’t be afraid to keep the text short and to include large illustrative figures. The guidelines and marking schemes are as follows:

</div>
</div>
<div class="layoutArea">
<div class="column">
6

</div>
</div>
<div class="layoutArea">
<div class="column">
1.

2. 3.

4.

</div>
<div class="column">
[15pts] Formal Description: Precisely define the way in which you are extending the algorithm. You should provide equations and possibly an algorithm box. Describe way your proposed method should be expected to perform better. For instance, are you intending it to improve the optimization, reduce overfitting, etc.?

[10pts] Figure or Diagram: that shows the overall model or idea. The idea is to make your report more accessible, especially to readers who are starting by skimming your report.

[15pts] Comparison or Demonstration: Include:

<ul>
<li>A comparison of the accuracies obtained by your model to those from baseline models.
Include a table or a plot for an illustrative comparison.
</li>
<li>Based on the argument you gave for why your extension should help, design and carry out an experiment to test your hypothesis. E.g., consider how you would disentangle whether the benefits are due to optimization or regularization.
[15pts] Limitations: of your approach.
</li>
</ul>
<ul>
<li>Describe some settings in which we’d expect your approach to perform poorly, or where
all existing models fail.
</li>
<li>Try to guess or explain why these limitations are the way they are.</li>
<li>Give some examples of possible extensions, ways to address these limitations, or open problems.
Optional: Competition
</li>
</ul>
</div>
</div>
<div class="layoutArea">
<div class="column">
Optionally, you will take part in a competition where you will submit the predictions of your model on a private test dataset. We will be using a platform called Kaggle3, which is a popular online community of data scientists and machine learning practitioners to solve data science challenges.

3 https://www.kaggle.com/

</div>
</div>
<div class="layoutArea">
<div class="column">
7

</div>
</div>
</div>
<div class="page" title="Page 8">
<div class="layoutArea">
<div class="column">
CSC311 Fall 2021 Final Project

</div>
</div>
<div class="layoutArea">
<div class="column">
You will first form a team on Kaggle and submit your prediction as a csv file on the project website, which can be found here:

<pre>            https://www.kaggle.com/t/51c62bffb0f348008333b29ab4385b23
</pre>
Your submission should be a csv file with two columns id and is correct. You should use the provided functions load_private_test_csv to load the dataset and save_private_test_csv to save the predictions as a csv file both located at utils.py. Once you save the csv file, you can directly upload it to the competition page. After the submission, you can find your model’s performance on the public leaderboard, which displays an accuracy of your predictions on 70% of private data. The full leaderboard that uses all private data will be released after the competition.

You are not required to use your model from Part B for this part.

Performance on the competition will not affect your grade, not even as extra credit. The top 3 teams with the highest performance on all private data will receive a special gift from the instructors.

</div>
</div>
<div class="layoutArea">
<div class="column">
References

[1] Wang, Zichao, et al. Diagnostic Questions: The NeurIPS 2020 Education Challenge. arXiv preprint arXiv:2007.12061 (2020)

</div>
</div>
</div>
