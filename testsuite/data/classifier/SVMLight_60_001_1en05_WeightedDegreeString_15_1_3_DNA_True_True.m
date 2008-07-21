classifier_accuracy = 0.0001;
init_random = 42;
accuracy = 1e-09;
classifier_labels = [-1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1];
seqlen = 60;
classifier_tube_epsilon = 0.01;
classifier_epsilon = 1e-05;
data_test = {'ATTGCTAAGTACCGGTTAGCAAACTATCCGCCAGGCCACGGCTCCAGGCATGGAATTTTT', 'ATTTGCTGATACCACGCGCCGACGAGTAACGGGGTGAGCACTTGCTGTGCTTGAGCGACA'};
data_type = '';
kernel_name = 'WeightedDegreeString';
classifier_type = 'kernel';
classifier_classified = [-0.0485163335, 0.251237645];
feature_type = 'Char';
classifier_C = 1.5;
name = 'SVMLight';
classifier_num_threads = 1;
data_train = {'TGAGTACTGAGTTTTCCCGTCTCCGCACTGCCGTTATTTACCCCTCTTTCGATTTTCCTT', 'TCAGGTACAATTAGCCCTTGAAGAGTTTTACGACCGTATTTATGCACAGCCTATTGTTCG', 'TACCCCTTACTGTTTGCACGTTGGAACCAGCAGAATAACTCTGGAAACCTCCTTTGCTCC', 'CGGGAATGACCGCGACTTCCCTAGGTTAGCCTGCCGGACAGGCAGCTCTAAACTTGGTCA', 'ACAAATGTATCATGAATAACAAGGCGGAGGGCGAATTGCCCTGCAACTCATCGCTCAGTC', 'TGCGATCATCCGGGTTGGACCACCGTACGCTCTATCCCTCGCCCGTCAAGCTGACACCCG', 'TTGACGCTGTTAATACCAACAAGCGGATCGCGCACAAAACTTGGGCAACAGACTTTGAAA', 'AGTAGACAACAGGCGGGTGGAACAGAGGTTACCTACCGTGCATCGAAATGACCTGCTCAA', 'GGGATAAGAGGCAAAAAAACAGAAGAGCCCAATCGCGTCGATGCTACGGTAGTTGGGCCC', 'CAGTCTATGACATGCTCCCCCCCTTGGGCTGGTTAGATATTGTTCGCGTTGGGTCTCATG', 'GGGCCCGGCTCATCGAGGGCAGGGATGTGGCACTTGTTCCCAGCTGGTTCATTCTACGGC'};
feature_class = 'string';
classifier_bias = 0.0441333252026;
kernel_arg0_degree = 3;
classifier_alphas = [-1.04910913, -1.0741882, -1.43228255, -1.24645838, -1.1588481, 1.21976246, 0.867857318, 1.05330266, 0.930352543, 0.961149019, 0.928462366];
alphabet = 'DNA';
classifier_support_vectors = [0, 5, 6, 7, 9, 1, 2, 3, 4, 8, 10];
classifier_linadd_enabled = 'True';
classifier_batch_enabled = 'True';
data_class = 'dna';
