#! /usr/bin/env python

import os
import numpy as np

""" This program verifies the Law of Conservation of Generalization or """
""" Part ii of NFL Theorem 2 (i.e., E1(D|C)=E2(C|D)) for the space of  """
""" binary functions on NUM_INPUTS Boolean variables  """

""" Supporting functions  """


def binary_sequence(int_to_code, code_length, comma=','):
    """ converts an integer 0 < n < 2^l-1 to a binary string of length l """
    bin_eq = ""
    while int_to_code != 0:
        n_div_2 = divmod(int_to_code, 2)
        bin_eq = comma + repr(n_div_2[1]) + bin_eq
        int_to_code = n_div_2[0]
    for _ in range(len(bin_eq)//2, code_length):
        bin_eq = comma + repr(0) + bin_eq
    return bin_eq[len(comma):]


def make_base_arff_headers():
    """ creates the headers of the base ARFF files """
    train_set = open(BASE_TRAIN_FILE_NAME, 'w')
    test_set = open(BASE_TEST_FILE_NAME, 'w')
    train_set.write('@relation BoolTrain')
    test_set.write('@relation BoolTest')
    for i in range(0, NUM_INPUTS):
        train_set.write('\n@attribute A' + repr(i) + ' {0,1}')
        test_set.write('\n@attribute A' + repr(i) + ' {0,1}')
    train_set.write('\n@attribute class {0,1}\n')
    test_set.write('\n@attribute class {0,1}\n')
    train_set.write('@data\n')
    test_set.write('@data\n')
    train_set.close()
    test_set.close()


def make_base_arff_contents(task_index):
    """ creates the contents of the base ARFF files for task of index """
    """ task_index, with NUM_EXAMPLES total, NUM_TEST examples for    """
    """ testing (OTS) [NUM_EXAMPLES - NUM_TEST for training] and      """
    """ NUM_INPUTS                                                    """
    """ NOTE: Since we are in the Boolean case, we assume that        """
    """       the total number of tasks is pow(2, NUM_EXAMPLES)       """
    """       and hence task_index is coded on NUM_EXAMPLES bits      """
    train_set = open(BASE_TRAIN_FILE_NAME, 'a')
    test_set = open(BASE_TEST_FILE_NAME, 'a')
    vector_c = binary_sequence(task_index, NUM_EXAMPLES)
    for i in range(0, NUM_EXAMPLES - NUM_TEST):
        train_set.write(binary_sequence(i, NUM_INPUTS) + ',' + vector_c[2*i] + '\n')
    for i in range(NUM_EXAMPLES - NUM_TEST, NUM_EXAMPLES):
        test_set.write(binary_sequence(i, NUM_INPUTS) + ',' + vector_c[2*i] + '\n')
    train_set.close()
    test_set.close()


""" Main program                                """

""" Get parameter values  """
NUM_INPUTS = input("Number of binary inputs: ")
NUM_TEST = input("Number of test instances: ")
classifier_key = raw_input("Classifier (be sure to write it as \'class.name\', e.g., \'trees.J48\'): ")
# The minority class algorithm.
MINORITY = 'minority'
# A shorthand mapping of algorithm abbreviations to their full names.
classifiers = {
    'j48': 'trees.J48',
    'mlp': 'functions.MultilayerPerceptron',
    'nb': 'bayes.NaiveBayes',
    'z': 'rules.ZeroR',
    'min': MINORITY
}
classifier_name = classifiers[classifier_key]

""" Initialize global variables  """
NUM_EXAMPLES = pow(2, NUM_INPUTS)
NUM_TASKS = pow(2, NUM_EXAMPLES)

BASE_TRAIN_FILE_NAME = "train.arff"
BASE_TEST_FILE_NAME = "test.arff"
OUTPUT_FILE_NAME = "task-gp.out"
ACC_START = 'Correctly Classified Instances'

""" Sets things up for the external call to WEKA """
""" Result is stored in OUTPUT_FILE_NAME """
calling_stem = 'java --illegal-access=warn -cp /Applications/WEKA/weka-3-8-2/weka.jar weka.classifiers.'
base_options = ' -t ' + BASE_TRAIN_FILE_NAME + ' -T ' + BASE_TEST_FILE_NAME + ' -o > ' + OUTPUT_FILE_NAME

""" Run classifier on all tasks and print result """

accuracies = [[], []]
if classifier_name is MINORITY:
    # Return the minority class in the training data.
    for index in range(NUM_TASKS):
        # The entirety of the task.
        full_sequence = binary_sequence(index, NUM_EXAMPLES, '')
        # The part of the task which we see.
        training_labels = full_sequence[:-NUM_TEST]
        # The portion of the task on which we will be tested.
        test_labels = full_sequence[-NUM_TEST:]
        # Train.
        counts = [0, 0]
        for c in training_labels:
            counts[int(c)] += 1
        prediction = np.argmin(counts)
        # Test.
        test_correct = 0
        for c in test_labels:
            if int(c) == prediction:
                test_correct += 1
        accuracy = test_correct / NUM_TEST
        accuracies[0] += [accuracy]
        accuracies[1] += [accuracy]
else:
    for index in range(NUM_TASKS):
        make_base_arff_headers()
        make_base_arff_contents(index)
        os.system(calling_stem + classifier_name + base_options)
        # Read output
        with open(OUTPUT_FILE_NAME) as fd:
            output_lines = fd.readlines()
            paradigm_index = 0
            for line in output_lines:
                if not line.startswith(ACC_START):
                    continue
                parts = line[len(ACC_START):].split()
                acc = parts[1]
                accuracies[paradigm_index] += [float(acc)/100.]
                paradigm_index += 1
                if paradigm_index == 2:
                    break

accuracies = np.array(accuracies)
training_perf = np.sum(accuracies[0] - .5)
test_perf = np.sum(accuracies[1] - .5)
print('Training generalization performance: {}'.format(training_perf))
print('Test generalization performance: {}'.format(test_perf))
training_acc = np.mean(accuracies[0])
test_acc = np.mean(accuracies[1])
print('Training average accuracy: {}'.format(training_acc))
print('Test average accuracy: {}'.format(test_acc))
