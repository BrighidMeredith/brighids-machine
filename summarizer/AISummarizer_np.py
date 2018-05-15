# -*- coding: utf-8 -*-
"""
Written By: Brighid Meredith
Top Level Function: Take text input (sententious) and return summary
Definitions: 
    sententious input is a complete thought, typically marked by '.' or '." or conjunction
    summary is a list of words in each category: Object, Action, and Subject
Technique: Neural Net using 3 different types of neurons
    TypeGate is a type of neuron that only permits words through that are of a specified type (i.e. an noun-gate will only allow nouns to pass through)
    DominantPick is a type of neuron that will return the words that are the most prolific or dominant in an input. Dominance defined by number of repitions
    Exploder is a type of neuron that puts out the sententious definition of each word provided (i.e. call the Language Processor and interpret the word's definition)
Optimization Required:
    How many layers to use, and which layers are devoted to what (i.e. subject, object, action, all, subject & object, subject & action, object & action)
    Should the network be feed forward or recursive
    How many levels should the network have
    Should all layers have the same number of levels
Stage 1: Unsupervised Learning Metric:
    If subject is present in input: Does the summary typically have a subject that matches the sententious input?
    If object is present in input: See above, but for objects
    If action is present in input: See above, but for actions
Stage 2: Unsupervised Learning Metric:
    Arrange the Summary into Sententious Input and feed back through the algorithm. The Summary should match.

References:  
    Packages Used: Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.
"""

import nltk
import math
import copy
#import string #for printing
import random #for generating intitial brain and mutating existing brain
import csv
import multiprocessing
#import os

#from urllib import request

#nltk.download()

"""
str1 = 'The Project Gutenberg EBook of Crime and Punishment, by Fyodor Dostoevsky\r\n'
tokens = nltk.word_tokenize(str1)
print(nltk.pos_tag(tokens))

str1 = 'I swallowed a fly, but don\'t ask me why!'
tokens = nltk.word_tokenize(str1)
print(nltk.pos_tag(tokens))

str1 = 'The bank was robbed.'
tokens = nltk.word_tokenize(str1)
print(nltk.pos_tag(tokens))
line_input = nltk.pos_tag(tokens)
tup1 = line_input[0]
w1 = tup1[0]
t1 = tup1[1]

print("nltk proofed")

print("action item: program to identify subject in any type of sentence")
print("action item: use subject and predicate to feed summarizing AI")



print("Completed.")
"""
class Utility:
    
    def accumulator(self, width, N, incoming_index):
        """ returns list of indexes to be combined.
            parameters: width of accumulator (number of streams combined)
            N (number of streams possible)
            incoming_index (index to build list around)"""
        #return the index above and the index below
        accumulate_indexes = []
        to_each_side = math.ceil(width/2)
        min_i = incoming_index - to_each_side;
        if 0 > min_i:
            min_i = 0
        max_i = incoming_index + to_each_side;
        if N <= max_i:
            max_i = N - 1
        for i in range(min_i, max_i):
            accumulate_indexes.append(i)
        return accumulate_indexes;
    
    #assume input of tagged tokens
    #for triggers: control parameter to be {__WORD__,__TAG___} type
    def filter(self, feed, control, word_or_type, filterOut_filterIn_flood):
        """ returns filtered stream
            parameters: feed and control, these are tagged tokenized lists
            word_or_type: 1 = word, 1 = type
            filterOut_filterIn_flood: 2 = filter_out, 1 = filter_in, 0 = flood"""
        passthrough = []
        check_feed = []
        check_control = []
        left1,right1 = zip(*feed)
        if word_or_type == 1: #then words
            check_feed = left1
        else: #then tags
            check_feed = right1
        left2,right2 = zip(*control)
        if word_or_type == 1:
            check_control = left2
        else:
            check_control = right2
        print(check_feed)
        print(check_control)
        if filterOut_filterIn_flood == 2: #filter out (evaluated on a trigger by trigger basis)
            for i in range(len(check_feed)):
                on_do_not_add_list = False
                for j in range(len(check_control)):
                    if check_feed[i] == check_control[j]:
                        on_do_not_add_list = True
                        break;
                if not on_do_not_add_list:
                    passthrough.append(feed[i])
        elif filterOut_filterIn_flood == 1: #filter in (evaulated wholistically. triggers must be found in order)
            met_criteria = True
            i_min = 0
            temp_passthrough = []
            j = 0
            while j < (len(check_control)):
                trigger_found = False
                for i in range(i_min,len(check_feed)):
                    i_min += 1
                    if check_feed[i] == check_control[j]:
                        trigger_found = True
                        temp_passthrough.append(feed[i])
                        break;
                if not trigger_found:
                    met_criteria = False
                    #temp_passthrough.clear()
                    break;
                else:
                    if j + 1 == len(check_control) and met_criteria:
                        j = 0
                        passthrough = [temp_passthrough[k] for k in range(len(temp_passthrough))] 
                    else:
                        j += 1
            #if met_criteria:
            #    passthrough = temp_passthrough
        elif filterOut_filterIn_flood == 0: #flood filter
            key_found = False
            i_min = 0
            for j in range(len(check_control)):
                for i in range(i_min,len(check_feed)):
                    i_min += 1
                    if check_control[j] == check_feed[i]:
                        key_found = True
                        break;
                if not key_found:
                    break;
            if key_found:
                passthrough = feed
        else:
            passthrough = feed        
        
        return passthrough;



class IdentifySubject:
    """Program is meant to learn to identify subject of a sentence based off word tags"""
    
    def randWeight(self,last_row=1,this_row=1):
        if last_row == this_row:
            return 1
        else:
            return 0
    
    def randType(self):
        
        key1 = self.possibilities[math.floor(random.random()*len(self.possibilities))]
        key2 = self.possibilities[math.floor(random.random()*len(self.possibilities))]
        index_key = 0
        if random.random() > .5:
            index_key = key1 * 36 + key2
        else:
            index_key = key1
        
        #print(index_key.__str__()+"\t"+key1.__str__()+"\t"+key2.__str__()+"\t")
        #print(self.translate_index_to_key_to_tags(index_key))
        return index_key;

    def initialize_gray_matter_from_scratch(self):
        self.weights = [[[1 for k in range(self.rows)] for j in range(self.rows)] for i in range(self.columns)]
        self.types = [[self.randType() for j in range(self.rows)] for i in range(self.columns)]               
        self.initialized = True
                       
    def initialize_gray_matter(self, new = 0):
        if new != 0:
            self.initialize_gray_matter_from_scratch()
        else:
            self.types = []
            try:
                
                #Load Gates
                with open(self.types_filename) as csvfile:
                    gray_matter_reader = csv.reader(csvfile, delimiter = ',')
                    i = 0
                    rows = 0
                    for row in gray_matter_reader:
                        types_in_row = []
                        j = 0
                        for gate in row:
                            #print(gate)
                            j += 1
                            rows = j
                            types_in_row.append(int(gate))
                        if len(types_in_row)>0:
                            self.types.append(types_in_row)
                            #print(types_in_row)
                            i += 1
                self.columns = i
                self.rows = rows
                print("columns:"+self.columns.__str__())
                print("rows:"+self.rows.__str__())
                
                self.weights = [[[1 for k in range(self.rows)] for j in range(self.rows)] for i in range(self.columns)]

                #Load Weights
                with open(self.weights_filename) as csvfile:
                    gray_matter_reader = csv.reader(csvfile, delimiter=',')
                    i_max = self.columns
                    j_max = self.rows
                    k_max = self.rows
                    k = 0
                    j = 0
                    i = 0
                    for row in gray_matter_reader:
                        if k == k_max:
                            k = 0
                            j += 1
                        if j == j_max:
                            j = 0
                            i += 1
                            if i >= i_max:
                                print("i is out of range")
                                break;
                        k_weights = []
                        for w in row:
                            k_weights.append((int(w)))
                        if len(k_weights) == k_max:
                            self.weights[i][j] = k_weights
                        #else:
                        #    print("row was of invalid length" + str(len(k_weights)))
                        k += 1
                    if i >= i_max:
                        print(self.weights_filename + "file corrupted")
                        print("rows expected: "+i_max.__str__())
                        print("rows found: "+i.__str__())
                        answer = int(input("continue with possible corruption? (Y = 1 / N = 0)"))
                        if answer != 1:
                            exit;
                self.initialized = True
                print("Gray matter loaded:")
                print(self.weights_filename)
                print(self.types_filename)
            except IOError:
                print("Unable to load Gray Matter for IdentifySubject")
                print(self.weights_filename)
                print(self.types_filename)
                answer = int(input("Try again? (Y = 1 / N = 0)"))
                if answer == 1:
                    self.initialize_gray_matter(self.columns, self.rows)
                else:
                    print("Initializing from scratch...")   

    def reset_TypeColumn(self,col_i):
        for j in range(len(self.types[col_i])):
            self.types[col_i][j] = self.randType()
    
    def reset_WeightColumn(self,col_i):
        for j in range(len(self.weights[col_i])):
            for k in range(len(self.weights[col_i][j])):
                self.weights[col_i][j][k] = 1

    def add_column(self):
        if self.columns > 0:
            self.columns += 1
            new_weights = [[self.randWeight(k,j) for k in range(self.rows)] for j in range(self.rows)]
            self.weights.append(new_weights)
            temp_types = [self.types[0][j] for j in range(self.rows)]            
            self.types.append(temp_types)            
        else:
            print("initialize columns before adding a new one...");
                    
    #initialize
    def __init__(self, weights_filename, gates_filename, columns = 3, rows = 10):
        self.subject = []
        self.columns = columns
        self.rows = rows
        self.weights_filename = weights_filename#'GrayMatter_IdentifySubjectWeights.csv'
        self.types_filename = gates_filename#'GrayMatter_IdentifySubjectGates.csv'
        self.possibilities = [3,12,13,14,15,16,17,18,19,26,27,28,29,30,31,33,34];
        self.subject_possibilities = [3,12,13,14,15,16,17,18,19,26,27,28,29,30,31,33,34];
        self.all_possibilities = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]        
        self.weights = []
        self.types = []
        self.initialized = False
        self.utility = Utility();
        #self.initialize_gray_matter()
        print("remember to initialize grey matter!")
    
    def set_columns(self, columns):
        self.columns = columns;
        self.initialize_gray_matter(new = 1)
    
    def set_rows(self, rows):
        self.rows = rows;
        self.initialize_gray_matter(new = 1)
        
    def save_gray_matter(self):
        print("saving gray matter...")
        print(self.weights_filename)
        print(self.types_filename)
        with open(self.weights_filename, 'w') as csvfile:
            gray_matter_writer = csv.writer(csvfile, delimiter=',')
            i_max = len(self.weights)
            i=0
            while i < i_max:
                j_max = len(self.weights[i])
                j=0
                while j < j_max:
                    gray_matter_writer.writerow(self.weights[i][j])
                    j = j + 1
                i = i + 1
        
        with open(self.types_filename,'w') as csvfile:
            gray_matter_writer = csv.writer(csvfile, delimiter=',')
            i_max = len(self.types)
            i = 0
            while i < i_max:
                gray_matter_writer.writerow(self.types[i])
                i += 1
        
    def translate_key_to_tag(self, key):
        tag = ''
        if key == -1:
            tag = 'NULLTAG'
        elif key == 1:
            tag = 'CC'  #CC coordinating conjunction
        elif key == 2:
            tag = 'CD'  #CD cardinal digit 
        elif key == 3:
            tag = 'DT'  #DT determiner 
        elif key == 4:
            tag = 'EX'  #EX existential there (like: "there is" ... think of it like "there exists")
        elif key == 5:
            tag = 'FW'  #FW foreign word 
        elif key == 6:
            tag = 'IN'  #IN preposition/subordinating conjunction
        elif key == 7:
            tag = 'JJ'  #JJ adjective 'big'
        elif key == 8:
            tag = 'JJR' #JJR adjective, comparative 'bigger'
        elif key == 9:
            tag = 'JJS' #JJS adjective, superlative 'biggest'
        elif key == 10:
            tag = 'LS'  #LS list marker 1)
        elif key == 11:
            tag = 'MD'  #MD modal could, will
        elif key == 12:
            tag = 'NN'  #NN noun, singular 'desk'
        elif key == 13:
            tag = 'NNS' #NNS noun plural 'desks'
        elif key == 14:
            tag = 'NNP' #NNP proper noun, singular 'Harrison'
        elif key == 15:
            tag = 'NNPS'    #NNPS proper noun, plural 'Americans'
        elif key == 16:
            tag = 'PDT' #PDT predeterminer 'all the kids'
        elif key == 17:
            tag = 'POS' #POS possessive ending parent's
        elif key == 18:
            tag = 'PRP' #PRP personal pronoun I, he, she
        elif key == 19:
            tag = 'PRP$'   #PRP$ possessive pronoun my, his, hers
        elif key == 20:
            tag = 'RB'  #RB adverb very, silently, 
        elif key == 21:
            tag = 'RBR' #RBR adverb, comparative better
        elif key == 22:
            tag = 'RBS' #RBS adverb, superlative best
        elif key == 23:
            tag = 'RP'  #RP particle give up
        elif key == 24:
            tag = 'TO'  #TO to go 'to' the store.
        elif key == 25:
            tag = 'UH'  #UH interjection errrrrrrrm
        elif key == 26:
            tag = 'VB'  #VB verb, base form take
        elif key == 27:
            tag = 'VBD' #VBD verb, past tense took
        elif key == 28:
            tag = 'VBG' #VBG verb, gerund/present participle taking
        elif key == 29:
            tag = 'VBN' #VBN verb, past participle taken
        elif key == 30:
            tag = 'VBP' #VBP verb, sing. present, non-3d take
        elif key == 31:
            tag = 'VBZ' #VBZ verb, 3rd person sing. present takes
        elif key == 32:
            tag = 'WDT' #WDT wh-determiner which
        elif key == 33:
            tag = 'WP'  #WP wh-pronoun who, what
        elif key == 34:
            tag = 'WP$' #WP$ possessive wh-pronoun whose
        elif key == 35:
            tag = 'WRB' #WRB wh-abverb where, when
        return tag;
        
    def translate_index_to_key_to_tags(self, index):
        #for system of 2 keys per index, 36 ints are permitted per spot (0 returns no tag)
        #36 * 36 = 1296 + 36 options
        first_key = math.floor(index/36)
        second_key = index - first_key*36
        first_tag = self.translate_key_to_tag(first_key)
        second_tag = self.translate_key_to_tag(second_key)
        if first_key == 0:
            return [second_tag]
        else:
            return [first_tag,second_tag]
    
    def test_arrays(self):
        for i in range(len(self.types)):
            print(self.types[i])
        
        for i in range(len(self.types)):
            tags = []
            for j in range(len(self.types[i])):
                tags.append(self.translate_index_to_key_to_tags(self.types[i][j]))
            print(tags)
    
    def permitted_types(self, gate_type, line_input):
        reduced_line_input = []
        allowed_types = self.translate_index_to_key_to_tags(gate_type)#['DT','NN']
        #print(allowed_types)
        len_allowed_types = len(allowed_types)        
        for i in range(len(line_input) - len_allowed_types + 1):
            #print(i)
            allowed = 1
            for j in range( len_allowed_types ):
                #print(j)
                type_at_iplusj = line_input[i+j][1]
                allowed_type_j = allowed_types[j]
                if allowed_type_j != type_at_iplusj:
                    allowed = 0
                    #print("breaking...")
                    break;
            if allowed == 1:
                #print("match")
                for j in range( len_allowed_types ):
                    reduced_line_input.append(line_input[i+j])
        #print(reduced_line_input)
        return reduced_line_input;
    
    #get line of input and provide subject
    #input is an array of pairs {word, type}
    def get_subject(self, line_input):
        #print(line_input)
        #calculate initial activations for first column
        init_activation = [0 for k in range(self.rows)]
        activation = init_activation
        for k in range (0,self.rows):
            #print("-")
            #print(self.types[0][k])
            init_activation[k] = self.permitted_types(self.types[0][k],line_input)
            
        
        #print(init_activation)
        
        for column in range(1,self.columns):
            for row in range(self.rows):
                this_row_activation = []
                for last_row in range(self.rows):
                    use_last_row = self.weights[column][row][last_row]
                    #print("use_last_row: "+use_last_row.__str__())
                    if use_last_row == 1:
                        #print("use_last_row")
                        #print(self.types[column][row])
                        #print(init_activation[row])
                        #print("call")
                        temp = self.permitted_types(self.types[column][row],init_activation[row])
                        #print(temp)
                        for t_j in range(len(temp)):
                            #print(temp[t_j])
                            this_row_activation.append(temp[t_j])
                activation[row] = this_row_activation
                #
                #if column == self.columns-1: #changed from -1
                #    #print(activation[row])
                #    #print(column.__str__()+"\t"+row.__str__())
                #    break;
                #
            #print(column)
            #print(activation)
            init_activation = activation
        
        final_activation = []
        for k in range (0,self.rows):
            if len(activation[k]) != 0:
                final_activation.append(activation[k])
        #print(init_activation)
        #print(activation)
        #print(final_activation)
        return final_activation;
    
    def get_cleanedSubject(self, line_input):
        final_activation = self.get_subject(line_input)
        unique_words = []
        unique_tokens = []
        num_of_tokens = 0;
        for ba in final_activation:
            for bae in ba:
                num_of_tokens += 1
                word_e = bae[0];
                unique = 1
                for word in unique_words:
                    if word_e == word:
                        unique = 0
                if unique == 1:
                    unique_words.append(word_e)
                    unique_tokens.append(bae)
        return unique_tokens;
        

class combine_svo:
    """ This class will take N number mini-brains which are optimized via IdentifySubject
        This class will take input, pass through N mini-brains and create an N array of outputs
        The outputs will then be fielded through a switch-brain before the final answers are put to an N array
        if a subject, verb, and object mini-brain are provided, then a subject, verb, and object answer array will be returned"""
    def __init__(self,weight_filenames,type_filenames):
        self.N = len(weight_filenames)
        self.N_Brains = []
        if len(weight_filenames) != len(type_filenames):
            print("ERROR! Mismatch between weight_filenames and type_filenames")
        else:
            for i in range(self.N):
                self.N_Brains.append(IdentifySubject(weight_filenames[i], type_filenames[i]))
                self.N_Brains[i].initialize_gray_matter()
                if self.N_Brains[i].initialized == False:
                    print("ERROR! brain "+i.__str__()+" is not initialized")
                
    def get_non_filtered_answers(self, line_input):
        non_filtered_answers = []
        for i in range(self.N):
            non_filtered_answers.append(self.N_Brains[i].get_cleanedSubject(line_input))
        return non_filtered_answers;
        
    """
    On Filters:
        Each column receives N filters
        each filter sits on input stream n (out of N)
        each filter may receive controller from any other stream != to n
        each filter may receive controller from any previous column (feed forward)
        each filter is identified as follows
        [ input feed = {n within N ; c within Columns} ; 
          word_or_wordType = 1/0 ; 
          static_or_regular_or_trigger = 2/1/0 ;
          trigger_index = j within Triggers]
    """
    def list_trigger_words_index(self,trigger_word_index):
        trigger_words = ['was','by','I','You','you','can\'t','will','should','could','shall','is','be','to','he','He','she','She','his','His','him','Him','hers','Hers','her','Her']
        return trigger_words[trigger_word_index];

    def list_trigger_types_index(self,trigger_type_index):
        trigger_types = ['noun','pronoun','name','verb_i','verb_t','adj','adv']
        return trigger_types[trigger_type_index];

    def accumulator(self, width, N, incoming_index):
        """ returns list of indexes to be combined.
            parameters: width of accumulator (number of streams combined)
            N (number of streams possible)
            incoming_index (index to build list around)"""
        #return the index above and the index below
        accumulate_indexes = []
        to_each_side = math.ceil(width/2)
        min_i = incoming_index - to_each_side;
        if 0 > min_i:
            min_i = 0
        max_i = incoming_index + to_each_side;
        if N <= max_i:
            max_i = N - 1
        for i in range(min_i, max_i):
            accumulate_indexes.append(i)
        return accumulate_indexes;
    
    #assume input of tagged tokens
    #for triggers: control parameter to be {__WORD__,__TAG___} type
    def filter(self, feed, control, word_or_type, filterOut_filterIn_flood):
        """ returns filtered stream
            parameters: feed and control, these are tagged tokenized lists
            word_or_type: 1 = word, 1 = type
            filterOut_filterIn_flood: 2 = filter_out, 1 = filter_in, 0 = flood"""
        passthrough = []
        check_feed = []
        check_control = []
        left1,right1 = zip(*feed)
        if word_or_type == 1: #then words
            check_feed = left1
        else: #then tags
            check_feed = right1
        left2,right2 = zip(*control)
        if word_or_type == 1:
            check_control = left2
        else:
            check_control = right2
        print(check_feed)
        print(check_control)
        if filterOut_filterIn_flood == 2: #filter out (evaluated on a trigger by trigger basis)
            for i in range(len(check_feed)):
                on_do_not_add_list = False
                for j in range(len(check_control)):
                    if check_feed[i] == check_control[j]:
                        on_do_not_add_list = True
                        break;
                if not on_do_not_add_list:
                    passthrough.append(feed[i])
        elif filterOut_filterIn_flood == 1: #filter in (evaulated wholistically. triggers must be found in order)
            met_criteria = True
            i_min = 0
            temp_passthrough = []
            j = 0
            while j < (len(check_control)):
                trigger_found = False
                for i in range(i_min,len(check_feed)):
                    i_min += 1
                    if check_feed[i] == check_control[j]:
                        trigger_found = True
                        temp_passthrough.append(feed[i])
                        break;
                if not trigger_found:
                    met_criteria = False
                    #temp_passthrough.clear()
                    break;
                else:
                    if j + 1 == len(check_control) and met_criteria:
                        j = 0
                        passthrough = [temp_passthrough[k] for k in range(len(temp_passthrough))] 
                    else:
                        j += 1
            #if met_criteria:
            #    passthrough = temp_passthrough
        elif filterOut_filterIn_flood == 0: #flood filter
            key_found = False
            i_min = 0
            for j in range(len(check_control)):
                for i in range(i_min,len(check_feed)):
                    i_min += 1
                    if check_control[j] == check_feed[i]:
                        key_found = True
                        break;
                if not key_found:
                    break;
            if key_found:
                passthrough = feed
        else:
            passthrough = feed        
        
        return passthrough;

    def convert_unfiltered_types_to_broad_categories(self):
        print("in work")
            
""" Test Environment """
###########################

weight_filenames = ['object_weights.csv','object_weights.csv'];
type_filenames = ['object_gates.csv','object_gates.csv'];
print(weight_filenames)
print(type_filenames)
test_brain = combine_svo(weight_filenames, type_filenames)
test_string = "Bill conquered trigonometry. Bill conquered all. Did Bill conquer?"
test_tokens = nltk.word_tokenize(test_string)
test_tags = nltk.pos_tag(test_tokens)
test_tags2 = nltk.pos_tag(nltk.word_tokenize("conquer"))
print(test_tags)
unfiltered_answers = test_brain.get_non_filtered_answers(test_tags)
print(unfiltered_answers)
print("testing filters")
print(test_tags2)
fake_control = test_tags2#test_tags[0]
print(fake_control)
print(test_brain.filter(test_tags,fake_control,1,1))#feed, control, word_or_type, filterOut_filterIn_flood))

############################
""" End Testing """


class GradeSubject:
    """This will compare the output of the IdentifySubject to a loaded list of sentences and answers and return a grade."""
    def load_file(self, filename):
        questions = []
        answers = []
        try:
            with open(filename) as f:
                lines = [line.rstrip('\n') for line in f]
                i = 0            
                for line in lines:
                    if i % 2 == 0:
                        questions.append(line)
                    else:
                        answers.append(line)
                    i += 1
            print("Loaded answer key.")
            print(filename)
        except IOError:
            print("Failed to load answer key.")
            print(filename)
        return questions, answers;
    
    def __init__(self, filename):
        self.answer_file_name = filename
        self.grade = 0
        self.questions, self.answers = self.load_file(filename)
        self.penalty_used = 0 #turn to 1 (on) so that penalties are applied to score
        #for i in range(len(self.questions)):
        #    print(self.questions[i] + "\t" + self.answers[i])
    
    def view_performance(self, brain):
        for i in range(len(self.questions)):
            q = self.questions[i]
            a = self.answers[i]
            q_tokens = nltk.word_tokenize(q)
            q_input = nltk.pos_tag(q_tokens)
            brains_answer = brain.get_cleanedSubject(q_input)
            
            print("line: "+q + "\t" + a)
            print(brains_answer)
    
            
    def get_grade(self, brain):
        grade = 0
        for i in range(len(self.questions)):
            q = self.questions[i]
            a = self.answers[i]
            q_tokens = nltk.word_tokenize(q)
            a_tokens = nltk.word_tokenize(a)
            a_w_tags = nltk.pos_tag(a_tokens)
            q_input = nltk.pos_tag(q_tokens)
            brains_answer = brain.get_cleanedSubject(q_input)
            #answer_found = grade;
            delta = 0
            penalty = 0
            
            for brains_answer_token in brains_answer:
                found_in_answer = 0
                for a_token in a_w_tags:
                    ba_token_word = brains_answer_token[0]
                    answer_word = a_token[0]
                    if ba_token_word == answer_word:
                        found_in_answer = 1
                        break;
                if found_in_answer == 1:
                    delta += 1
                elif found_in_answer == 0:
                    penalty += .1
            
            grade = grade + delta
            
            if self.penalty_used == 1:
                grade = grade - penalty

        return grade;
    
    def max_score(self):
        grade = 0
        for i in range(len(self.questions)):
            a = self.answers[i]
            a_tokens = nltk.word_tokenize(a)
            a_w_tags = nltk.pos_tag(a_tokens)
            brains_answer = a_w_tags
            delta = 0
            penalty = 0
            
            for brains_answer_token in brains_answer:
                found_in_answer = 0
                for a_token in a_w_tags:
                    ba_token_word = brains_answer_token[0]
                    answer_word = a_token[0]
                    if ba_token_word == answer_word:
                        found_in_answer = 1
                        break;
                if found_in_answer == 1:
                    delta += 1
                elif found_in_answer == 0:
                    penalty += .1
            
            grade = grade + delta
            
            if self.penalty_used == 1:
                grade = grade - penalty

        return grade;
        
    def get_grade_v1(self, brain):
        grade = 0
        for i in range(len(self.questions)):
            q = self.questions[i]
            a = self.answers[i]
            q_tokens = nltk.word_tokenize(q)
            a_tokens = nltk.word_tokenize(a)
            q_input = nltk.pos_tag(q_tokens)
            brains_answer = brain.get_subject(q_input)
            
            
            #print(brains_answer)
            #assign partial credits based on progress. if all partials accomplished, sum to whole
            tally = 0
            for a_item in nltk.pos_tag(a_tokens):
                for item in brains_answer:
                    if item[0] == a_item[0]:
                        tally += 1
                        break;
            correct_words_found = tally/len(a_tokens)
            correct_number = 0
            diff = abs(len(brains_answer) - len(a_tokens))
            if diff < .1:
                correct_number = 1
            elif diff < 1.1:
                correct_number = .75
            elif diff < 2.1:
                correct_number = .5
            elif diff < 3.1:
                correct_number = .25
            
            if len(brains_answer) > 0:
                grade += 1;
            
            for item in brains_answer:
                if item[1][0]== 'N':
                    grade += 1.0/len(brains_answer)
            
            
            grade += correct_number + correct_words_found
        return grade;

#y = GradeSubject("subject_answer_key.txt")
#y.get_grade(x)

class GreedyOptimization:
    """This calss will take a brain and a grade rubric and will optimize the brain through random changes"""
    
    def __init__(self, brain, grader, mutations_per_grade = 10, generations_before_wipe = 1, chance_to_change_weight = .1, save = 1):
        self.brain = brain
        self.grader = grader
        self.generations_before_wipe = int(generations_before_wipe)
        
        self.mutations_per_grade = mutations_per_grade
        self.largest_type_change = 5
        self.chance_to_change_weight_vs_type = chance_to_change_weight
        
        self.last_grade = grader.get_grade(self.brain)
        self.current_grade = 0
        self.delta = 0
        self.wiper = 0
        self.save = save
        self.skip_if_no_improvement = 1
        self.skip_if_more_than_this_invariability = .95
        self.loops_without_improvement_cap = 3
        self.column_range = range(0,len(self.brain.types))
        self.row_range = range(0,len(self.brain.types[0]))
    
        
    def super_greedy_weight(self,col,row):
        start_grade = self.grader.get_grade(self.brain)
        sum_grade1 = start_grade
        sum_grade2 = 0.0
        sum_grade3 = 0.0
        
        for going_into in range(0,self.brain.rows):
            weight = self.brain.weights[col][row][going_into]
            new_weight = abs(weight - 1) #if was 1, now zero. if was zero, now 1
            self.brain.weights[col][row][going_into] = new_weight
            new_grade = self.grader.get_grade(self.brain)
            sum_grade3 = sum_grade2
            sum_grade2 = sum_grade1
            sum_grade1 = new_grade
            if new_grade > start_grade:
                start_grade = new_grade
                weight = new_weight
            else:
                self.brain.weights[col][row][going_into] = weight
            if (sum_grade1+sum_grade2+sum_grade3)/3 - start_grade < .1 and self.skip_if_no_improvement == 1:
                #no change at this neuron. skip.
                break;
         
    def super_greedy_weight_no_cap(self,col,row):
        start_grade = self.grader.get_grade(self.brain)
        self.current_grade = start_grade        
        for going_into in range(0,self.brain.rows):
            weight = self.brain.weights[col][row][going_into]
            new_weight = abs(weight - 1) #if was 1, now zero. if was zero, now 1
            self.brain.weights[col][row][going_into] = new_weight
            new_grade = self.grader.get_grade(self.brain)
            if new_grade > start_grade:
                start_grade = new_grade
                self.current_grade = start_grade
                weight = new_weight
            else:
                self.brain.weights[col][row][going_into] = weight
    
    def super_greedy(self,col,row):
          
        possible_mutationsi = self.brain.subject_possibilities
        possible_mutationsj = possible_mutationsi
        random.shuffle(possible_mutationsi)
        gate = self.brain.types[col][row]
        start_grade = self.grader.get_grade(self.brain) 
        grade_store = start_grade
        #Below helps to prevent needless activity for low-impact neurons
        loops_without_improvement = 0
        loops_of_loops_without_improvement = 0
        #Randomize 
        #print(self.brain.possibilities)
        for i in range(-1,len(possible_mutationsi)):
        #    if exit_loops or no_variation/loop_count > self.skip_if_more_than_this_invariability:
        #        break;
            if loops_of_loops_without_improvement > self.loops_without_improvement_cap:
                break;
            random.shuffle(possible_mutationsj)
            for j in range(len(possible_mutationsj)):
                new_gate = gate
                if i == -1:
                    new_gate = possible_mutationsi[j]
                else:
                    new_gate = possible_mutationsi[i]*36+possible_mutationsj[j]
                #print(self.brain.translate_index_to_key_to_tags(new_gate))
                self.brain.types[col][row] = new_gate
                new_grade = self.grader.get_grade(self.brain)
                print(new_gate.__str__()+"\t"+new_grade.__str__()+"\t"+start_grade.__str__()+"\t"+gate.__str__());
                if new_grade > start_grade:
                    start_grade = new_grade
                    gate = new_gate
                if new_grade != grade_store:
                    loops_without_improvement = 0
                    loops_of_loops_without_improvement = 0
                    grade_store = new_grade
                else:
                    loops_without_improvement += 1
                if loops_without_improvement > self.loops_without_improvement_cap:
                    loops_of_loops_without_improvement += 1
                    break;

        self.brain.types[col][row] = gate
                    
    def super_greedy_no_cap(self,col,row):
        gate = self.brain.types[col][row]
        start_grade = self.grader.get_grade(self.brain) 
        self.current_grade = start_grade
        for i in range(-1,len(self.brain.possibilities)):
            for j in range(0,len(self.brain.possibilities)):
                new_gate = gate
                if i == -1:
                    new_gate = self.brain.possibilities[j]
                else:
                    new_gate = self.brain.possibilities[i]*36+self.brain.possibilities[j]
                self.brain.types[col][row] = new_gate
                new_grade = self.grader.get_grade(self.brain)
                if new_grade > start_grade:
                    start_grade = new_grade
                    gate = new_gate
                    self.current_grade = start_grade
        self.brain.types[col][row] = gate

    def identify_important_neurons_by_type(self):
        important_neurons = [[0 for j in range(self.brain.rows)] for i in range(self.brain.columns)]
        for i in range(self.brain.columns):
            for j in range(self.brain.rows):
                gate = self.brain.types[i][j]
                old_grade = self.grader.get_grade(self.brain)                
                self.brain.types[i][j] = -1
                new_grade = self.grader.get_grade(self.brain)
                self.brain.types[i][j] = gate
                if abs(old_grade - new_grade) >= .1:
                    important_neurons[i][j] = 1        
        return important_neurons;
                

    def mutate(self, temp_brain):
        brain_i = temp_brain.columns
        brain_j = temp_brain.rows
        brain_k = brain_j
        for i in range(self.mutations_per_grade):
            switcher = random.random()
            i = math.floor(brain_i * random.random())
            j = math.floor(brain_j * random.random())
            k = math.floor(brain_k * random.random())
            
            if switcher < self.chance_to_change_weight_vs_type:
                temp_brain.weights[i][j][k] = abs(self.brain.weights[i][j][k] - 1)
            else:
                #gate = self.brain.types[i][j]
                gate = self.brain.randType() #gate + round((random.random() - .5)*2*self.largest_type_change)
                #gate = max(gate,0)
                temp_brain.types[i][j] = gate
                #print(gate)
        return temp_brain
        
    def mutatation_by_range(self, temp_brain):
        for i in self.column_range:
            for j in self.row_range:
                switcher = random.random()
                if(switcher > 1 / self.mutations_per_grade):
                    gate = self.brain.randType();
                    temp_brain.types[i][j] = gate
                    for k in self.row_range:
                        switcher = random.random()*2
                        if math.floor(switcher) == 1:
                            temp_brain.weights[i][j][k] = 0
                        else:
                            temp_brain.weights[i][j][k] = 1
        return temp_brain
            
    def mutate_and_test(self):
        temp_brain = copy.deepcopy(self.brain)
        #last_grade = self.grader.get_grade(temp_brain)
        #temp_brain = self.mutate(temp_brain)
        temp_brain = self.mutatation_by_range(temp_brain)
        current_grade = self.grader.get_grade(temp_brain)
        #print(current_grade.__str__() + "\t" + last_grade.__str__())
        if current_grade - self.last_grade > -.0001:
            return {current_grade, temp_brain};
        else:
            return 0, self.brain;
            #print("new brain")
            #self.brain = temp_brain
            #self.last_grade = current_grade
            
    
    def generation_controller(self):
        first_grade = self.last_grade;#self.grader.get_grade(self.brain)
        #self.last_grade = first_grade
        for i in range(int(self.generations_before_wipe)):
            score,temp_brain = self.mutate_and_test()
            if score > first_grade:
                print("mutate_and_test was success: old score = "+first_grade.__str__()+" new score = "+score.__str__())
                first_grade = score
                self.brain = temp_brain
        second_grade = self.last_grade;#grader.get_grade(self.brain)
        self.delta = second_grade - first_grade
        #print("improvement to score: "+ self.delta.__str__())
        self.current_grade = second_grade
        if self.save == 1:
            self.brain.weights_filename = "v"+self.wiper.__str__()+self.brain.weights_filename
            self.brain.types_filename = "v"+self.wiper.__str__()+self.brain.types_filename
            self.brain.save_gray_matter()
            self.wiper += 1
        return self.delta

            

class create_svo_identifier:
    """Takes a filename for solutions answer set. Runs through the below steps. Returns 2 files (at filedirectory) for types and weights.
        step 1: create small single column array
        step 2: optimize small array without penalties applied (or with penalties, depending on severity)
        step 3: add columns
        step 3.1: verify that weights work as intended (FW to FW, 1 to 1, not all rows to 1)
        step 4: locate the neurons that have largest affect on score
        step 5: optimize important neurons
        step 6: repeat 4-6 until no delta is found
        step 7: if required: random permutations until no delta is found
        step 8: if required: add another column and repeat 4-7"""
    def __init__(self,solution_set_filename, types_filename, weights_filename, generations_before_wipe=100, goal_percent = .95, rows=10, from_scratch=True, optimize=True, add_columns=True, max_number_columns = 2, infinity=False):
        self.solutions = solution_set_filename
        self.types_filename = types_filename
        self.weights_filename = weights_filename
        initial_columns = 1 #only start with 1 column. can add others later.
        self.brain = IdentifySubject(weights_filename,types_filename,initial_columns,rows) #initialize the brain
        self.brain.possibilities = self.brain.all_possibilities #broaden the scope to all possibilities. more time, but very thourough, especially when not all combos are known before hand
        mutations_per_grade = 5 #inverse is used as switch threshold
        chance_to_change_weight = .9 #no longer used
        #generations_before_wipe = 1000 #number of generations to perform before exiting
        save = 0 #if 1, then generation controller will save to a unique filename. irritating. not recommended.
        if from_scratch:
            self.brain.initialize_gray_matter_from_scratch() #will create a new brain, per columns and rows specified above
        else:
            self.brain.initialize_gray_matter() #will attempt loading brain from given filename
        self.brain.save_gray_matter() #will save initial brain
        print("brain initialized and saved. view of types array:")
        self.brain.test_arrays() #will print initial brain
        self.grader = GradeSubject(solution_set_filename) #initialize the grader to be used
        self.z = GreedyOptimization(self.brain,self.grader,mutations_per_grade,generations_before_wipe,chance_to_change_weight,save) #initialize the optimizing engine
        if optimize and from_scratch:
            self.optimize_types() #optimze column
        self.z.grader.penalty_used = 1 #initialize penalties for incorrect words
        #if add_columns and self.z.brain.columns < max_number_columns:
            #self.z.brain.add_column() #duplicate first column and add to array
            #if optimize:
                #self.optimize_weights() #optimize weights of newest column
        #if optimize:    
            #skipped super greedy entire organism optimization. gains were marginal and problems were encountered.
            #self.evo_optimization() #optimize entire brain through evo
        score = self.z.grader.get_grade(self.z.brain)
        goal_score = goal_percent * self.z.grader.max_score()
        while(self.z.brain.columns < max_number_columns and score < goal_score and optimize):
            print("score: "+score.__str__()+" goal: "+goal_score.__str__())
            self.z.brain.add_column()
            self.optimize_weights()
            self.evo_optimization()
            score = self.z.grader.get_grade(self.z.brain)
        if infinity:
            while True:
                score = self.z.grader.get_grade(self.z.brain)
                goal_score = goal_percent * self.z.grader.max_score()
                print("to infinity: current score: "+score.__str__()+" goal score: "+goal_score.__str__())
                if goal_score-score < .0001:
                    break;
                self.evo_optimization()
        self.z.brain.save_gray_matter()
        print("COMPLETE")
        
        
    
    def optimize_types(self):
        print("optimizing types of all neurons all columns. greedy.")
        curr_score = self.z.current_grade
        no_improvement_cap = 2        
        no_improvement = 0        
        for col in range (len(self.z.brain.types)):
            if no_improvement > no_improvement_cap:
                break;
            for row in range (len(self.z.brain.types[0])):
                print(self.z.grader.get_grade(self.z.brain))
                self.z.super_greedy_no_cap(col,row)
                self.z.brain.save_gray_matter()
                if curr_score - self.z.current_grade < .001:
                    no_improvement += 1
                if no_improvement > no_improvement_cap:
                    break;
        print("finished greedy type optimization:"+self.z.grader.get_grade(self.z.brain).__str__())
        self.brain.save_gray_matter()
        
    def optimize_weights(self):
        print("optimizing weights of largest column. greedy.")
        for col in range (len(self.z.brain.types)-1,len(self.z.brain.types)):
            for row in range (len(self.brain.types[0])):
                print(self.z.grader.get_grade(self.z.brain))
                self.z.super_greedy_weight_no_cap(col,row)
                self.z.brain.save_gray_matter()
        print("finished greedy weight optimization:"+self.z.grader.get_grade(self.z.brain).__str__())
        self.z.brain.save_gray_matter()
        
    def evo_optimization(self):
        print("optimizing entire brain through random permutations. greedy.")
        improving = 0
        gen_to_run = self.z.generations_before_wipe;
        while(improving > -5):    
            grade_1 = self.z.grader.get_grade(self.z.brain)
            delta = self.z.generation_controller()
            self.z.brain.save_gray_matter()
            print("delta: "+delta.__str__()+" old: "+grade_1.__str__())
            if delta <= .001:
                improving -= 1
                print("decreasing number of gerneations by 10%")
                self.z.generations_before_wipe = .9 * self.z.generations_before_wipe
            else:
                print("increasing number of gerneations by 50%")
                self.z.generations_before_wipe = 1.5 * self.z.generations_before_wipe
        print("finished greedy random permutations.")
        self.z.generations_before_wipe = gen_to_run

        
#create_brain = create_svo_identifier("object_answer_key.txt", "object_gates.csv", "object_weights.csv", goal_percent = .95, rows=10, from_scratch=True, optimize=True, add_columns=True, max_number_columns = 3)
#create_brain = create_svo_identifier("object_answer_key.txt", "object_gates.csv", "object_weights.csv", generations_before_wipe=1, goal_percent = .95, rows=10, from_scratch=False, optimize=True, add_columns=True, max_number_columns = 3, infinity=True)

from multiprocessing import Process

def call_create_svo_identifier(which_one):
    if which_one == 0:
        print(0)        
        create_brain = create_svo_identifier("object_answer_key.txt", "object_gates.csv", "object_weights.csv", generations_before_wipe=10000, goal_percent = .95, rows=50, from_scratch=False, optimize=True, add_columns=True, max_number_columns = 10, infinity=False)
        
    if which_one == 1:
        print(1)
        create_brain = create_svo_identifier("subject_answer_key.txt", "subject_gates.csv", "subject_weights.csv", generations_before_wipe=10000, goal_percent = .95, rows=50, from_scratch=False, optimize=True, add_columns=True, max_number_columns = 10, infinity=False)

    if which_one == 2:
        print(2)
        create_brain = create_svo_identifier("action_answer_key.txt", "action_gates.csv", "action_weights.csv", generations_before_wipe=1000, goal_percent = .95, rows=50, from_scratch=False, optimize=True, add_columns=True, max_number_columns = 10, infinity=False)
    
    
    
def dummy(faker,faker2):
    print(faker)


def main():
    #worker1 = multiprocessing.Process(target = call_create_svo_identifier, args = ("object_answer_key.txt", "object_gates.csv", "object_weights.csv", generations_before_wipe=100, goal_percent = .95, rows=20, from_scratch=True, optimize=True, add_columns=True, max_number_columns = 10, infinity=False),)
    #worker1 = Process(target = dummy, args = ('arg','arg2',))
    worker1 = Process(target = call_create_svo_identifier, args = (0,))    
    worker1.start()
    worker2 = Process(target = call_create_svo_identifier, args = (1,))    
    worker2.start()
    worker1.join()
    worker1.close()
    worker2.join()    
    worker3 = Process(target = call_create_svo_identifier, args = (2,))    
    worker3.start()
    worker2.join()    
    worker3.join()
    print("done")

#call_create_svo_identifier(2)
#call_create_svo_identifier(1)
#call_create_svo_identifier(0)

#if __name__ == '__main__':
#    main()    
    #worker1 = multiprocessing.Process(target = call_create_svo_identifier, args = ("object_answer_key.txt", "object_gates.csv", "object_weights.csv", generations_before_wipe=100, goal_percent = .95, rows=20, from_scratch=True, optimize=True, add_columns=True, max_number_columns = 10, infinity=False))
    #worker2 = pool.apply_async(create_svo_identifier,("subject_answer_key.txt", "subject_gates.csv", "subject_weights.csv", generations_before_wipe=100, goal_percent = .95, rows=20, from_scratch=True, optimize=True, add_columns=True, max_number_columns = 10, infinity=False))
    #worker3 = pool.apply_async(create_svo_identifier,("action_answer_key.txt", "action_gates.csv", "action_weights.csv", generations_before_wipe=100, goal_percent = .95, rows=20, from_scratch=True, optimize=True, add_columns=True, max_number_columns = 10, infinity=False))
 