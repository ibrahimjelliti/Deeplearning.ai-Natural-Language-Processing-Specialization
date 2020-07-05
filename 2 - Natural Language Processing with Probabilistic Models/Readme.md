# Natural Language Processing with Probabilistic Model
Welcome to the [second course](https://www.coursera.org/learn/probabilistic-models-in-nlp) of the Natural Language Processing Specialization at [Coursera](https://www.coursera.org/specializations/natural-language-processing) which is moderated by [DeepLearning.ai](http://deeplearning.ai/). The course is taught by Younes Bensouda Mourri, Łukasz Kaiser and Eddy Shyu.

## Table of contents
- [Natural Language Processing with Probabilistic Model](#natural-language-processing-with-probabilistic-model)
  - [Table of contents](#table-of-contents)
  - [Course summary](#course-summary)
  - [Autocorrect and Dynamic Programming](#autocorrect-and-dynamic-programming)
    - [Autocorrect](#autocorrect)
    - [Building the model](#building-the-model)
    - [Minimum edit distance](#minimum-edit-distance)
    - [Minimum edit distance algorithm](#minimum-edit-distance-algorithm)
  - [Part of Speech Tagging and Hidden Markov Models](#part-of-speech-tagging-and-hidden-markov-models)
    - [Part of Speech Tagging](#part-of-speech-tagging)
    - [Markov Chains](#markov-chains)
    - [Markov Chains and POS Tags](#markov-chains-and-pos-tags)
    - [Hidden Markov Chains models](#hidden-markov-chains-models)
    - [The Transition Matrix](#the-transition-matrix)
    - [The Emission  Matrix](#the-emission-matrix)
    - [The Viterbi Algorithm](#the-viterbi-algorithm)
      - [Initialization](#initialization)
      - [Forward Pass](#forward-pass)
      - [Backward Pass](#backward-pass)

## Course summary
This is the  course summary as its given on the course [link] (https://www.coursera.org/learn/probabilistic-models-in-nlp):

In Course 2 of the Natural Language Processing Specialization, offered by deeplearning.ai, you will:

> Create a simple auto-correct algorithm using minimum edit distance and dynamic programming,
> Apply the Viterbi Algorithm for part-of-speech (POS) tagging, which is important for computational linguistics,
> Write a better auto-complete algorithm using an N-gram language model, and 
> Write your own Word2Vec model that uses a neural network to compute word embeddings using a continuous bag-of-words model.
 
> Please make sure that you’re comfortable programming in Python and have a basic knowledge of machine learning, matrix multiplications, and conditional probability.

> By the end of this Specialization, you will have designed NLP applications that perform question-answering and sentiment analysis, created tools to translate languages and summarize text, and even built a chatbot!

> This Specialization is designed and taught by two experts in NLP, machine learning, and deep learning. Younes Bensouda Mourri is an Instructor of AI at Stanford University who also helped build the Deep Learning Specialization. Łukasz Kaiser is a Staff Research Scientist at Google Brain and the co-author of Tensorflow, the Tensor2Tensor and Trax libraries, and the Transformer paper.

## Autocorrect and Dynamic Programming
### Autocorrect
- Autocorrect is an application that changes misspelled words into the correct ones.
  - Example: Happy birthday *deah* friend! ==> dear
- How it works:
  1. Identify a misspelled word
  2. Find strings n edit distance away
  3. Filter candidates
  4. Calculate word probabilities 
### Building the model
- Identify a misspelled word
  - If word not in vocabulary then its misspedlled 
- Find strings n edit distance away
  - Edit: an operation performed on a string to change it
    - how many operations away one string is from another
      - Insert (add a letter)
         - Add a letter to a string at any position: to ==> top,two,...
      -  Delete (remove a letter)
         - Rmove a letter  from a string : hat ==> ha, at, ht
      -  Switch (swap 2 adjacent letters)
         - Exmaple: eta=> eat,tea
      -  Replace (change 1 letter to another)
         - Example: jaw ==> jar,paw,saw,...
 -  By combining the 4 edit operations, we get list of all possible strings that are n edit.
    - ![](Images/1.png)
- Filter candidates: 
  - From the list from step 2, consider only real and correct spelled word
  - if the edit word not in vocabualry ==> remove it from list of candidates
    - ![](Images/2.png)
- Calculate word probabilities: the word candidate is the one with the highest probability 
  - a word probablity in a corpus is: number of times the word appears divided by the total number of words.
    - ![](Images/3.png) 
### Minimum edit distance
- Evaluate the similarity between 2 strings
- Minimum number of edits needed to transform 1 string into another
- the algorithm try to minimize the edit cost  
  - ![](Images/4.png)
- Applications:
  -  Spelling correction
  -  document similarity
  -  machine translation
  -  DNA sequencing
  -  etc
### Minimum edit distance algorithm
- The source word layed on the column
- The target word layed on the row
- Empty string at the start of each word at (0,0)
- D[i,j] is the minimum editing distance between the beginning of the source word to index i and the beginning of the target word to index j 
  - ![](Images/5.png)
- To fillout the rest of the table we can use this formulaic approach:
  - ![](Images/6.png)
## Part of Speech Tagging and Hidden Markov Models
### Part of Speech Tagging
- Part-of-speech refers to the category of words or the lexical terms in the language
  - tags can be: Noun, Verb, adjective, preposition, adverb,...
  - Example for the sentence: why not learn something ?
    - ![](Images/7.png)
- Applications:
  - Named entities
  - Co-reference resolution
  - Speech recognition 
### Markov Chains
- Markov chain can be depicted as a directed graph
  - a graph is a kind of data structure that is visually represented as a set of circles connected by lines.
- The circles of the graph represents states of our model
- The arraows from state s1 to s2 represents the transition probabilities, the likelihood to move from s1 to s2 
  - ![](Images/8.png) 
### Markov Chains and POS Tags
- Think about a sentence as a sequence of words with associated parts of speech tags
  - you can represent that sequence with a graph 
  - where the parts of speech tags are events that can occur depicted by the states of the model graph.
  - the weights on the arrows between the states define the probability of going from one state to another
    - ![](Images/9.png)  
- The probability of the next event only depends on the current events
- The model graph can be represented as a Transition matrix with dimension n+1 by n
  - when no previous state, we introduce an initial state π.
  - The sum of all transition from a state should always be 1.
    - ![](Images/10.png)   
### Hidden Markov Chains models
- The hidden Markov model implies that states are hidden or not directly observable
- The hidden Markov model have a transition probability matrix A of dimensions (N+1,N) where N is number of hidden states
- The hidden Markov model have emission probabilities matrix B describe the transition from the hidden states to the observables(the words of your corpus)
  - the row sum of emission probabity for a hidden state is 1 
  - ![](Images/11.png)
### The Transition Matrix
- Transition matrix holds all the transition probabilities between states of the Markov model
- C(t<sub>i-1</sub>,t<sub>i</sub>) count all occurrences of tag pairs in your training corpus
- C(t<sub>i-1</sub>,t<sub>j</sub>) count all occurrences of tag t<sub>i-1</sub>
  - ![](Images/12.png)
- ![](Images/13.png)
- to avoid division by zero and lot of entries in the transition matrix are zero, we apply smoothing to the probability formula 
  - ![](Images/14.png)
### The Emission  Matrix
- Count the co-occurrences of a part of speech tag with a specific word.
  - ![](Images/15.png)
### The Viterbi Algorithm
- The Viterbi algorithm is actually a graph algorithm
- The goal is to to find the sequence of hidden states or parts of speech tags that have the highest probability for a sequence
  - ![](Images/16.png)
- The algorithm can be split into three main steps: the initialization step, the forward pass, and the backward pass.
- Given your transition and emission probabilities, you first populates and then use the auxiliary matrices C and D
  - matrix C holds the intermediate optimal probabilities
  - matrix D holds the indices of the visited states as we are traversing the model graph to find the most likely sequence of parts of speech tags for the given sequence of words, W<sub>1</sub> all the way to W<sub>k</sub>.
  - C and D matrix have n rows (number of parts of speech tags) and k comlumns (number of words in the given sequence)
#### Initialization
- The initialization of matrix C tell the probability of every word belongs to a certain part of speech.
  - ![](Images/17.png)
- in D matrix, we store the labels that represent the different states we are traversing when finding the most likely sequence of parts of speech tags for the given sequence of words W<sub>1</sub> all the way to W<sub>k</sub>.
#### Forward Pass
- For the C matrix, the entries are calculated by this formula:
  - ![](Images/18.png)
- For matrix D, save the k, which maximizes the entry in c<sub>i,j</sub>.
  - ![](Images/19.png)
#### Backward Pass
- The backward pass help retrieve the most likely sequence of parts of speech tags for your given sequence of words.
- First calculate the index of the entry, C<sub>i,K</sub>, with the highest probability in the last column of C
  - represents the last hidden state you traversed when you observe the word w<sub>i</sub>
- Use this index to traverse back through the matrix D to reconstruct the sequence of parts of speech tags
-  multiply many very small numbers like probabilities leads to numerical issues
   - Use log probabilities instead where numbers are summed instead of multiplied.
   - ![](Images/20.png)