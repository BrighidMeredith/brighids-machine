# -*- coding: utf-8 -*-
"""
Written By: Brighid Meredith
Top Level Function: Take text input (sententious) and return summary
Definitions:
    sententious input is a complete thought, typically marked by '.' or '." or
    conjunction
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
import random #for generating intitial brain and mutating existing brain
import pickle
import statistics
import time


class CoreUtility:

    #assumed neuron_array is N x Columns array of tuples
    def __init__(self, neuron_array):
        self.neuron_array = neuron_array
        self.N = len(neuron_array)
        self.Columns = len(neuron_array[0])

    def append_neuron_array(self, appending_neuron_array, direction):
        """ Add annother neuron array to the self.neuron_array. add to the left, right, up, or down.
            Strongly recommended: Use one primary class when adding multiple arrays
            Else risk losing partition data. It is assumed that N = N for left and right, that Columns = Columns for up and down"""
        if direction == 'down':
            for line in appending_neuron_array:
                self.neuron_array.append(line)
        if direction == 'up':
            for line in self.neuron_array:
                appending_neuron_array.append(line)
            self.neuron_array = appending_neuron_array
        if direction == 'right':
            for i in range(len(self.neuron_array)):
                self.neuron_array[i] = self.neuron_array[i] + appending_neuron_array[i]
        if direction == 'left':
            for i in range(len(appending_neuron_array)):
                self.neuron_array[i] = appending_neuron_array[i] + self.neuron_array[i]
        self.N = len(self.neuron_array)
        self.Columns = len(self.neuron_array[0])


    def accumulator(self, width, incoming_index):
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
        if self.N <= max_i:
            max_i = self.N - 1
        for i in range(min_i, max_i):
            accumulate_indexes.append(i)
        return accumulate_indexes;

    def splicer(self, *incoming_feed):
        passthrough = []
        for line in incoming_feed:
            for token_tag in line:
                #print(token_tag)
                #print("printed")
                passthrough.append(token_tag)
        return passthrough;

    #assume input of tagged tokens
    #for triggers: control parameter to be {__WORD__,__TAG___} type
    def nfilter(self, feed, control, word_or_type, filterOut_filterIn_flood):
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



    #input must be an N by 1 array of strings
    def process_input(self, N_line_input):
        if len(N_line_input) != self.N:
            print("number of input lines does not match neuron rows")
            return None;
        else:
            for col in range(self.Columns):
                next_N_line_input = [None] * self.N#[[('','')] for i in range(self.N)]
                for row in range(self.N):
                    neuron = self.neuron_array[row][col]
                    neuronType = neuron[0]
                    if neuronType == -1:
                        #passthrough empty column
                        next_N_line_input[row] = N_line_input[row]
                    if neuronType == 0:
                        #accumulator
                        width = neuron[1]
                        accum_indexes = self.accumulator(width,row)
                        for i in accum_indexes:
                            temp = []
                            for tagged_word in N_line_input[i]:
                                temp.append(tagged_word)
                                #print('tagged word')
                                #print(tagged_word)
                            next_N_line_input[row] = temp
                    if neuronType == 1:
                        #filter
                        if N_line_input[row]:
                            word_or_type = neuron[1]
                            filterOut_filterIn_flood = neuron[2]
                            static_filter_control = neuron[3]
                            if static_filter_control == 1:
                                static_filter_control = neuron[4]
                            else:
                                static_filter_control = N_line_input[neuron[4]]
                            #print(N_line_input[row])
                            next_N_line_input[row] = self.nfilter(N_line_input[row], static_filter_control, word_or_type, filterOut_filterIn_flood)
                            #print('filtered word')
                            #print(next_N_line_input[row])
                            #print('end')
                    if neuronType == 2:
                        #splicer
                        distance_to_travel = row + neuron[1]
                        while (distance_to_travel >= self.N):
                            distance_to_travel = distance_to_travel - self.N
                        next_N_line_input[row] = self.splicer(N_line_input[row],N_line_input[distance_to_travel])
                        print(N_line_input[row])
                        print(N_line_input[distance_to_travel])
                        print(next_N_line_input[row])
                        print(distance_to_travel)
                #end of neurons in column, save over previous input
                N_line_input = next_N_line_input
                return N_line_input;
    # Future Expansion: Exchange x by y pockets of data between parents & child instead of just rows
    def child_maker(self, neuron_array1, neuron_array2):
        child_neuron_array = []
        for i in range(len(neuron_array1)):
            donated_row = []
            if random.random() < .5 or len(neuron_array2) <= i:
                donated_row = neuron_array1[i]
            else:
                donated_row = neuron_array2[i]
            child_neuron_array.append(donated_row)
            l = len(child_neuron_array)
        for i in range(len(neuron_array2) - l):
            child_neuron_array.append(neuron_array2[i + l])
        brain = CoreUtility(child_neuron_array)
        return brain;

    # Pickle files end with .p
    def save_cortex(self, filename):
        pickle.dump (self.neuron_array, open(filename, "wb"))

    def load_cortex(self, filename):
        return pickle.load(open(filename, "rb"))

class IdentifyWordFunction:
    """Program is meant to learn to identify subject of a sentence based off word tags"""

    #initialize
    def __init__(self):
        self.possibilities = [3,12,13,14,15,16,17,18,19,26,27,28,29,30,31,33,34];
        self.subject_possibilities = [3,12,13,14,15,16,17,18,19,26,27,28,29,30,31,33,34];
        self.all_possibilities = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]

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

    def randType(self, size):

        word_tags = [0] * size
        for i in range(size):
            key = self.possibilities[math.floor(random.random()*len(self.possibilities))]
            tag = self.translate_key_to_tag(key)
            word_tags[i] = ('',tag)
        return word_tags;

    def randWord(self, size):
        return nltk.pos_tag(nltk.word_tokenize('was by'));

    def basic_accumulator(self, width):
        return (0,width)

    def basic_splicer(self,distance):
        return (2, distance)

    def basic_filter(self, word_or_type, filterOut_filterIn_flood, static_filter_control, control):
        return (1, word_or_type, filterOut_filterIn_flood, static_filter_control, control);

    def basic_cortex(self, rows,columns,col_directions):
        cortex = [None] * rows
        for row in range(rows):
            row_of_neurons = []
            for column in range(columns):
                if col_directions[column] == 0:
                    neuron = self.basic_accumulator(3)
                    row_of_neurons.append(neuron)
                if col_directions[column] == 1:
                    x = math.ceil(random.random()*2)
                    neuron = self.basic_filter(0,1,1,self.randType(x))
                    row_of_neurons.append(neuron)
            cortex[row] = row_of_neurons
        return cortex;

    # Allowable mutations: filter by static word_type, accumulator
    def mutator_for_region_type1(self, neuron):
        max_width = 10
        prob_of_accumulator = .5
        max_tags = 2
        if random.random() < prob_of_accumulator:
            return self.basic_accumulator(math.ceil(random.random()*max_width))
        else:
            filter_type = random.randint(0,2)
            return self.basic_filter(0,filter_type,1,self.randType(math.ceil(random.random()*max_tags)))

    # Allowable mutations: splice, filter by static word or word_type, accumulator
    def mutator_for_region_type2(self, neuron):
        max_width = 10
        prob_of_accumulator = .5
        max_tags = 2
        prob_of_splicer = .5
        splicer_footprint = 300

        if random.random() < prob_of_accumulator:
            if random.random() < prob_of_splicer:
                return self.basic_splicer(math.ceil((random.random())*splicer_footprint))
            else:
                return self.basic_accumulator(math.ceil(random.random()*max_width))
        else:
            filter_type = random.randint(0,2)
            filter_word_or_type = random.randint(0,1)
            if filter_word_or_type == 0:
                return self.basic_filter(0,filter_type,1,self.randType(math.ceil(random.random()*max_tags)))
            else:
                return self.basic_filter(1,filter_type,1,self.randWord(math.ceil(random.random()*max_tags)))

    def mutator(self, neuron, region_type = 1):
        if region_type == 1:
            return self.mutator_for_region_type1(neuron)
        elif region_type == 2:  # region_type = 2:
            return self.mutator_for_region_type2(neuron)
        elif region_type == -1:
            return (-1)



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
        self.penalty_used = 1 #turn to 1 (on) so that penalties are applied to score
        #for i in range(len(self.questions)):
        #    print(self.questions[i] + "\t" + self.answers[i])

    def quick_expand_input_to_fit_brain_N(self, input, N):
        return [nltk.pos_tag(nltk.word_tokenize(input)) for i in range(N)]

    def view_performance(self, brain):
        N = brain.N
        for i in range(len(self.questions)):
            a = self.answers[i]
            brains_answer = brain.process_input(self.quick_expand_input_to_fit_brain_N(self.questions[i],N))
            print("line: "+self.questions[i] + "\t" + a)
            print(brains_answer)


    def get_grade(self, brain):
        grade = 0
        N = 0
        try:
            N = brain.N
        except AttributeError:
            N = len(brain)
        for i in range(len(self.questions)):
            a = self.answers[i]
            a_tokens = nltk.word_tokenize(a)
            a_w_tags = nltk.pos_tag(a_tokens)
            brains_answer = brain.process_input(self.quick_expand_input_to_fit_brain_N(self.questions[i],N))
            delta = 0
            penalty = 0

            #print(brains_answer)

            for row_answer in brains_answer:
                if row_answer:
                    for brains_answer_token in row_answer:
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
                    penalty += .3

            grade = grade + delta

            if self.penalty_used == 1:
                grade = grade - penalty

        return grade;


class organism:
    def __init__(self, brain):
        self.brain = brain

class population:
    def __init__(self, organism):
        self.organism = organism

class evolution_optimization:
    def __init__(self, population, mutator, fitness, child_maker,
                 rule_mutations = None):
        self.population = population
        self.mutator = mutator
        self.fitness = fitness
        self.child_maker = child_maker
        self.hall_of_fame = []  # (Max_score, Organism, Mean_score_of_pop)
        self.rule_mutations = rule_mutations

    def apply_mutations(self, prob_of_mutation):
        #prob_of_mutation = .05;
        for organism in self.population.organism:
            if random.random() < prob_of_mutation:
                N = organism.brain.N
                L = organism.brain.Columns
                neuron_n = random.randint(0,N-1)
                neuron_l = random.randint(0,L-1)
                neuron = organism.brain.neuron_array[neuron_n][neuron_l]
                # print(neuron)
                if self.rule_mutations:
                    pass_result = self.rule_mutations(neuron_n, neuron_l, neuron)
                    neuron = self.mutator(neuron, pass_result)
                else:
                    neuron = self.mutator(neuron)
                # print(neuron)
                organism.brain.neuron_array[neuron_n][neuron_l] = neuron

    def apply_fitness(self, num_questions = 0):
        questions_to_use = num_questions
        max_gradepoints = len(self.fitness.questions)
        # Store questions in copy as fitness is truncated
        questions = copy.deepcopy(self.fitness.questions)
        if num_questions != 0:
            index_qs = [random.randint(0,max_gradepoints-1) for i in range(questions_to_use)]
            self.fitness.questions = [self.fitness.questions[index_qs[i]] for i in range(len(index_qs))]
        # Get Score for each organism
        population_score = []
        for organism in self.population.organism:
            population_score.append(self.fitness.get_grade(organism.brain))
            #print(max_gradepoints)

        # Reset questions to original
        self.fitness.questions = questions
        return population_score;

    def roulette_wheel(self, pop_score):
        pop_mean = statistics.mean(pop_score)
        pop_stdv = statistics.stdev(pop_score)

        # Initialize wheel, every child is gauranteed at least one spot
        wheel = [i for i in range(len(self.population.organism))]
        for i in range(len(self.population.organism)):
            score = pop_score[i]
            possible_additions = 1
            if pop_stdv > .01:
                while(score >= pop_mean):
                    possible_additions += 4
                    score -= pop_stdv
            for j in range(possible_additions):
                wheel.append(i)

        # Shuffle and return wheel
        random.shuffle(wheel)
        return wheel;

    def mate(self,organism1,organism2):
        brain1 = organism1.brain.neuron_array
        brain2 = organism2.brain.neuron_array
        child_brain = self.child_maker(brain1, brain2)
        entity = organism(child_brain)
        return entity;

    def next_generation(self, wheel, bonus = 1):
        size_of_generation = len(self.population.organism)
        size_of_wheel = len(wheel)
        children_per_couple = math.ceil(2 * size_of_generation / size_of_wheel)
        new_generation = []
        # Since wheel is already shuffled, iterate until size of generation
        no_back_tracks_j = 0
        for i in range(size_of_generation):
            organism_j1 = self.population.organism[wheel[no_back_tracks_j]]
            organism_j2 = self.population.organism[wheel[no_back_tracks_j+1]]
            no_back_tracks_j += 2 * bonus
            for k in range(children_per_couple):
                child = self.mate(organism_j1, organism_j2)
                new_generation.append(child)
        return new_generation;

    def run_evolution_simulation(self, generations = 100, questions_per_cycle = 5, prob_of_mutation = 0.05, output = False):
        for i in range(generations):
            self.apply_mutations(prob_of_mutation)
            pop_score = self.apply_fitness(questions_per_cycle)
            pop_mean = statistics.mean(pop_score)
            pop_stdv = statistics.stdev(pop_score)
            pop_max = max(pop_score)
            pop_min = min(pop_score)
            if output:
                print("generation statistics: mean: "+pop_mean.__str__()+" stdv: "+pop_stdv.__str__()+" max: "+pop_max.__str__()+" min: "+pop_min.__str__())
            # Save best performer in hall of fame
            for j in range(len(pop_score)):
                if abs(pop_max - pop_score[j]) < 0.001:
                    entry_hall_of_fame = (pop_max, self.population.organism[j],
                                          pop_mean)
                    self.hall_of_fame.append(entry_hall_of_fame)
                    break;
            wheel = self.roulette_wheel(pop_score)
            #print(wheel)
            nextgen = self.next_generation(wheel)
            #print(len(nextgen))
            self.population.organism = nextgen
    #  Intensive round that both finds the best performance and delivers the true performance of population
    def exam_round(self, output = True):
        if True:
            pop_score = self.apply_fitness()  # All Questions
            pop_mean = statistics.mean(pop_score)
            pop_stdv = statistics.stdev(pop_score)
            pop_max = max(pop_score)
            pop_min = min(pop_score)
            if output:
                print("generation statistics: mean: "+pop_mean.__str__()+" stdv: "+pop_stdv.__str__()+" max: "+pop_max.__str__()+" min: "+pop_min.__str__())
            # Save best performer in hall of fame
            best_organism = self.population.organism[0]
            for j in range(len(pop_score)):
                if abs(pop_max - pop_score[j]) < 0.001:
                    best_organism = self.population.organism[j]
                    entry_hall_of_fame = (pop_max, best_organism,
                                          pop_mean)
                    self.hall_of_fame.append(entry_hall_of_fame)
                    break;
            wheel = self.roulette_wheel(pop_score)
            #print(wheel)
            nextgen = self.next_generation(wheel, bonus = 100)
            #print(len(nextgen))
            self.population.organism = nextgen
            return pop_max, pop_mean, pop_stdv, best_organism


optimizing = False
testing = False
orthogonal_testing = True

if orthogonal_testing:
    """ Use L4 table to test and find ideal parameters to increase pop mean
        and decrease convergence time """
    #  L4 Orthogonal Matrix
    L4_Orth = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]]
    L4_Orth_results = []  # List of tuples (elapsed time, mean score, max score)
    #  Initial values
    population_sizes = [100, 1000]
    mutation_factor = [.0025, .01]
    questions_per_test = [10, 0]  # Note: 0 => All questions available
    columns_test = 5
    rows_test = 100
    column_instructions_test = [1, 0, 1, 0, 1]  #Filter, Pool, Filter, Pool, Filter
    initial_test_brain_structure_filename = 'orth_test_results/test_brain_start.p'
    test_against_questions_answers = "subject_answer_key.txt"
    base_filename_for_in_test_saves = "orth_test_results/L4_subject_test_"
    time_limit_for_test = 10 * 60 * 60  # * 60 * 60 converts to seconds
    generations_before_exam = 10
    allowable_error = 0.001  # Essentially what the program considers 0
    all_time_best_organism = None
    all_time_best_score = 0

    # Construct initial test individual if not already available
    brain_structure = IdentifyWordFunction()
    brain_structure_test = brain_structure.basic_cortex(rows_test,columns_test,
                                                      column_instructions_test)
    y = GradeSubject(test_against_questions_answers)
    blankUtility = CoreUtility([[]])
    try:
        brain_structure_test = blankUtility.load_cortex(
                                         initial_test_brain_structure_filename)
    except:
        print("could not load "+initial_test_brain_structure_filename)
    # Create brain (structure + utility)
    brain_test = CoreUtility(brain_structure_test)
    brain_test.save_cortex(initial_test_brain_structure_filename)
    # Construct base organism
    organism_test = organism(copy.deepcopy(brain_test))
    # Perform Orthogonal Test
    for ft in L4_Orth:
        pop_size = population_sizes[ft[0]]
        mut_factor = population_sizes[ft[1]]
        qs_test = questions_per_test[ft[2]]

        organisms_test = [copy.deepcopy(organism_test) for i in range(pop_size)]
        pop_test = population(organisms_test)
        evo = evolution_optimization(pop_test, brain_structure.mutator, y,
                                     blankUtility.child_maker)
        start_time = time.monotonic()  # Time run starts
        not_converged = True
        elapsed_time = 0
        last_three_means = [0,0,0]
        pop_mean = 0  # Temporary hold for Population Mean
        pop_max = 0  # Temporary hold for Max Score in Iteration
        while not_converged and elapsed_time < time_limit_for_test:
            evo.run_evolution_simulation(generations_before_exam,
                                         qs_test, mut_factor, output = False)
            #  Run Exam
            pop_max, pop_mean, pop_stdev, best_org = evo.exam_round()
            #  test_scores = evo.apply_fitness()
            #  pop_mean = statistics.mean(test_scores)
            #  pop_stdev = statistics.stdev(test_scores)
            #  pop_max = max(test_scores)
            elapsed_time = time.monotonic() - start_time
            #  print("population mean\t\t"+pop_mean.__str__() +
            #      "\tstdev\t" + pop_stdev.__str__() +
            #      "\telapsed time\t\t" + elapsed_time.__str__())

            # Record winner
            best_org = evo.hall_of_fame[-1][1]
            t_fname = (base_filename_for_in_test_saves +
                       time.monotonic().__str__() +
                       "_" +
                       pop_max.__str__() +
                       ".p")
            best_org.brain.save_cortex(t_fname)
            if pop_max > all_time_best_score:
                all_time_best_organism = best_org
            # Breakout Logic
            previous_average = sum(last_three_means)/len(last_three_means)
            if abs( previous_average - pop_mean ) < allowable_error:
                not_converged = False
            for i in range(len(last_three_means) - 1):
                last_three_means[i + 1] = last_three_means[i]
            last_three_means[i] = pop_mean
        print("completed trial: total time:\t" + elapsed_time.__str__() + "\tpop_mean:\t" + pop_mean)
        L4_Orth_results.append((elapsed_time, pop_mean, pop_max))  # List of tuples (elapsed time, mean score, max score)
    #  Finished all trials
    print("completed all trials: all_time_best_score:\t"+all_time_best_score.__str__())
    all_time_best_organism.brain.save_cortex(base_filename_for_in_test_saves + "all_time_best.p")
    print(L4_Orth_results)
    pickle.dump (L4_Orth_results, open(base_filename_for_in_test_saves + "_L4_results.p", "wb"))





if testing:
    id_brain = IdentifyWordFunction()
    y = GradeSubject("subject_answer_key.txt")
    blankUtility = CoreUtility([[]])

    string1 = "this is a test"
    string2 = "of the emergency"
    string3 = "broadcasting signal"

    tag1 = nltk.pos_tag(nltk.word_tokenize(string1))
    tag2 = nltk.pos_tag(nltk.word_tokenize(string2))
    tag3 = nltk.pos_tag(nltk.word_tokenize(string3))

    print(blankUtility.splicer(tag1,tag2,tag3))

    fake_neuron = id_brain.basic_splicer(10)

    for i in range(100):
        print(id_brain.mutator(fake_neuron,2))

    fake_neuron_array = []#[[(1, 1, 2, 1, [('was', 'VBD'), ('by', 'IN')]) * 10 ]*10]
    for i in range(10):
        fake_neuron_array.append((1, 1, 2, 1, [('was', 'VBD'), ('by', 'IN')]))
    print(fake_neuron_array)
    print("??")
    fakeBrain = id_brain.basic_cortex(10,1,[1])
    print(fakeBrain)
    for i in range(len(fakeBrain)):
        fakeBrain[i][0] = (2, 206)
    print(fakeBrain)
    fakeUtility = CoreUtility(fakeBrain)
    fake_input_row = nltk.pos_tag(nltk.word_tokenize("this was done by the baron"))
    fake_input = [fake_input_row for i in range(10)]
    print(len(fake_input))
    print(fakeUtility.N)
    print(fakeUtility.Columns)
    fakeUtility.process_input(fake_input)

    print("\n\n\n\n\n\n\n\n\n\n")
    fakeBrain1 = id_brain.basic_cortex(5,2,[0,1])
    fakeBrain2 = id_brain.basic_cortex(5,1,[0,0])
    fakeUtilityBrain = CoreUtility(fakeBrain1)
    print(fakeBrain1)
    print()
    print(fakeBrain2)
    print()
    fakeUtilityBrain.append_neuron_array(fakeBrain2, 'left')
    print(fakeUtilityBrain.neuron_array)

    # Initialize rule function
    def rule_for_region1_region2(row, column, neuron):
        if neuron[0] == -1:
            return -1
        elif column < 10:
            return 1
        else:
            return 2

    # Test evo optimization
    pop_size = 10
    rows_A = 10
    rows_B = 10
    rows_C = 10
    rows_D = 30

#    don't forget where you are!
    organisms = [organism(CoreUtility(id_brain.basic_cortex(10,1,[1]))) for i in range(5)]
    pop = population(organisms)
    evo = evolution_optimization(pop, id_brain.mutator, y, blankUtility.child_maker)
    print(evo.apply_fitness())
    evo.apply_mutations(1)
    print(evo.apply_fitness())
    evo.apply_mutations(1)
    print(evo.apply_fitness())

if optimizing:


    # Initialize tools that will create IdentifyWordFunction Brain
    id_brain = IdentifyWordFunction()
    y = GradeSubject("subject_answer_key.txt")
    blankUtility = CoreUtility([[]])
    filename = "subject.p"
    pop_size = 1000
    rows_organism = 100 #number of rows per brain per organism

    organisms = [organism(CoreUtility(id_brain.basic_cortex(rows_organism,1,[1]))) for i in range(pop_size)]
    try:
        brain_i = CoreUtility(blankUtility.load_cortex(filename))
        organisms = [organism(copy.deepcopy(brain_i)) for i in range(pop_size)]
    except FileNotFoundError:
        print(FileNotFoundError)
    pop = population(organisms)
    evo = evolution_optimization(pop, id_brain.mutator, y, blankUtility.child_maker)

    columns_can_add = 20
    score_column_add = 0
    score_improvement_diff_columns = 5
    all_time_best_organism = organisms[0]
    all_time_best_score = 0

    cycles = 100
    questions_cycle = 5
    mutation_factor = .005
    score_i = 0
    all_time_best_organism = evo.population.organism[0]
    columns_can_add = 10


    i = 0
    while(i < columns_can_add):
        j = 0
        while j < i+2:
            evo.run_evolution_simulation(cycles,questions_cycle,mutation_factor)
            evo.hall_of_fame.clear()
            evo.run_evolution_simulation(1,round(len(y.questions)/2),0)
            score, organisms = zip(*evo.hall_of_fame)
            #print(score)
            max_score = max(score)
            print("max_score:"+max_score.__str__())
            max_index = 0
            for i in range(len(score)):
                if max_score - score[i] < .001:
                    max_index = i
                    break;
            organisms[max_index].brain.save_cortex(filename)
            if max_score - score_i > .001:
                all_time_best_organism = organisms[max_index]
                print("new best score! "+max_score.__str__())
            j += 1
        print("adding column columns left to add: "+columns_can_add.__str__())
        new_column = id_brain.basic_cortex(rows_organism,1,[1])

        neural_net_of_organism = all_time_best_organism.brain.neuron_array
        for i in range(len(neural_net_of_organism)):
            neural_net_of_organism[i].append(new_column[i])

        organisms = [organism(CoreUtility(neural_net_of_organism)) for i in range(pop_size)]
        pop = population(organisms)
        evo.population = pop
        i += 1

    all_time_best_organism.brain.save_cortex("all_time_best_"+filename)
