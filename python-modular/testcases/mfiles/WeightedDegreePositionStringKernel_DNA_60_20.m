traindat = ['CCCGGATGCGTAAATCACATAGTTATTGTATATTGAAATACTCTGAACCTGTGCACGAGG', 'GCAACAGGAGAACTGGCCAATAGGTTTAGACCATCTAACTCCTGTGTCCGCGGTCGCGAA', 'AATGTGGTAACGAGCGAGCAACGCAAGACGGGAAACACTTTATCTGAAGATCGACGTGCT', 'GTAACATCATTTTGGCGTTGATGGAATGGATGCCTACGTAAGGCCAGATAGCCTCTGCGA', 'TATTGTCCGGTGTATCTACCTTTTGGCACCTCGCTAACAACACCCCTTCTGGCACAAGAC', 'GAGCTATTGCGAATAATGATTACCATATGTAATACAAACATCGTGCTTAAACTACCCTCG', 'CCGGTTACCGTGTGAAGACAGACTCGCCCCTCAGACCTTCCAGTTTATTCCTGCAGTGTA', 'TTCACGTGGATGGTTTTTGTGAACACTGACACACTCACGAGATCTTATATAGATTGCGTG', 'CATATGCCCTGGCAAAGCATGCGTATAAACAGATTCATCGCGCTCCACGTGCTATTCCGT', 'GCGGCATGTGGTGCTGCTCTACGCAAACCATGTCCGGGTACACCAGAACTTTGAACGTCT', 'AATCTCCGACTATGTCTACTTGCCAGTACGAAGCTTCGTCATCATATCGGCACAGACACC']
testdat = ['CTTGGGCGTAGTCCGTATTCGGGGTAGTCCAATTCGCTCTCGGCGGTTGAAATTCGCGCG', 'GTGTAACTACAGAGCCTAAATGTAGCGTCAGCCGAAAGATGAAACCGCGAGGGAGAGCAA', 'AACGCGTTGTCAGGTGCTCGCTCGCCGTTAAAGTTGCGTTACATCCGCACTACAGCTCAA', 'AGAGAAGACGCGCGATCCGTGTAGAGGGAGTTTCCCAACGCGGTGAGCATAGTAGCACCT', 'ACTCCTCTCGACATTGCTAGTCTGCATCCATTAACGATCTGTACAAAACACTCATAATAG', 'CCCGTAGTCCATTAGTACTATTAAGGATTGCACGCGCCGACACACCCTCCAACCGACGTC', 'ATAGGATAAATCACTCAGGAGTGCGATGTGTCGAAAACCGGCACAGCCATATAAATTAGG', 'GAAGAAAAAGCTACACGGATTCAATTCTAAATGTGCTACTCAGCTATCTGATGCTTGCTG', 'GCCCCGATACTAGTAACGCTAAACTAGCCGCATCCATATTAGAAAAGAAGTAACGATTGA', 'CTCCCTCGCCAATACCCCTCGCTGATGGTTAGAGACGGTCGTGGTTAGTCAAAAAGGCCC', 'TAGTGACATACCTAAAGCCGTCAGGACACATCCTTTCCAGACATCTGCAACTAGGGTCCC', 'GTCACGTAGGCTTTCGTACTTAGCGTCTGCCGATAGGTTCTGATTGCCTCCGCAGTTGCT', 'ACTCAAGAACCGGGCAAATGATAATAGACAATCGCTTTCGCGTCCCGGCAGGTCTTCTTA', 'CGGCCGCGTGGGGATCAAAGTATGAGGGAGTTGGGACGAGCTTCATAGGGAGTGTTAACC', 'GACTCCCCCCATCTGAGAGGATACAAATTTGCGAAGTTGAAGCTTGAGTGCGAGTGAGCC', 'TATGCGTCGAGATTCCGAGACAGCTTACTACCCACGAGTCTGTTTTGAGCGATGGTTCTG', 'GGCGATTATCAGAGCTGGTCCGAGGCGCGTGAGCCTAGCTCAGATTTTTGGCTTGACTTG']
km_train = [1, 0.0369121562, 0.015971606, 0.0369121562, 0.0468500444, 0.0470275067, 0.044809228, 0.028216504, 0.0487133984, 0.0660159716, 0.0334516415;0.0369121562, 1, 0.0317657498, 0.0414374445, 0.0229813665, 0.036823425, 0.0247559894, 0.0367346939, 0.028216504, 0.0316770186, 0.0298136646;0.015971606, 0.0317657498, 1, 0.0247559894, 0.0298136646, 0.0299911269, 0.0419698314, 0.044809228, 0.0403726708, 0.0708961846, 0.0417923691;0.0369121562, 0.0414374445, 0.0247559894, 1, 0.0332741792, 0.0212067436, 0.0212067436, 0.0299023957, 0.0299911269, 0.0520851819, 0.034960071;0.0468500444, 0.0229813665, 0.0298136646, 0.0332741792, 1, 0.0296362023, 0.058296362, 0.0387755102, 0.0386867791, 0.0425022183, 0.0668145519;0.0470275067, 0.036823425, 0.0299911269, 0.0212067436, 0.0296362023, 1, 0.0210292813, 0.0317657498, 0.0470275067, 0.0334516415, 0.0401952085;0.044809228, 0.0247559894, 0.0419698314, 0.0212067436, 0.058296362, 0.0210292813, 1, 0.0448979592, 0.036823425, 0.0366459627, 0.0299023957;0.028216504, 0.0367346939, 0.044809228, 0.0299023957, 0.0387755102, 0.0317657498, 0.0448979592, 1, 0.0417923691, 0.0299911269, 0.0264418811;0.0487133984, 0.028216504, 0.0403726708, 0.0299911269, 0.0386867791, 0.0470275067, 0.036823425, 0.0417923691, 1, 0.0247559894, 0.028216504;0.0660159716, 0.0316770186, 0.0708961846, 0.0520851819, 0.0425022183, 0.0334516415, 0.0366459627, 0.0299911269, 0.0247559894, 1, 0.0281277728;0.0334516415, 0.0298136646, 0.0417923691, 0.034960071, 0.0668145519, 0.0401952085, 0.0299023957, 0.0264418811, 0.028216504, 0.0281277728, 1]
km_test = [0.0279503106, 0.0264418811, 0.0264418811, 0.0420585626, 0.0352262644, 0.0415261757, 0.0653948536, 0.0438331854, 0.0332741792, 0.0370008873, 0.0211180124, 0.015971606, 0.015971606, 0.0450754215, 0.0176574978, 0.0229813665, 0.0194321207;0.0612244898, 0.0367346939, 0.0246672582, 0.034960071, 0.0405501331, 0.0299023957, 0.0212954747, 0.050310559, 0.0283052351, 0.02635315, 0.0230700976, 0.0472049689, 0.0317657498, 0.0212067436, 0.0364685004, 0.034960071, 0.0365572316;0.0403726708, 0.0467613132, 0.0299911269, 0.0299023957, 0.0335403727, 0.0211180124, 0.0439219166, 0.0247559894, 0.036823425, 0.028216504, 0.0281277728, 0.0502218279, 0.0449866903, 0.0212067436, 0.0381543922, 0.0402839397, 0.0328305235;0.0366459627, 0.0352262644, 0.0386867791, 0.0299023957, 0.0264418811, 0.0212067436, 0.0401952085, 0.034960071, 0.0367346939, 0.0316770186, 0.030079858, 0.0316770186, 0.0450754215, 0.0279503106, 0.0382431233, 0.0369121562, 0.0265306122;0.0246672582, 0.049068323, 0.0246672582, 0.0212954747, 0.0352262644, 0.0519077196, 0.0382431233, 0.0194321207, 0.0106477374, 0.0193433895, 0.0488908607, 0.0364685004, 0.0314108252, 0.0386867791, 0.0299023957, 0.0261756877, 0.0281277728;0.0419698314, 0.0283052351, 0.0332741792, 0.043566992, 0.028216504, 0.028216504, 0.028216504, 0.0450754215, 0.0420585626, 0.0228926353, 0.0245785271, 0.0350488021, 0.0229813665, 0.0228926353, 0.0299023957, 0.0352262644, 0.040461402;0.0246672582, 0.0229813665, 0.0195208518, 0.0281277728, 0.0281277728, 0.0369121562, 0.0211180124, 0.0383318545, 0.0264418811, 0.0535048802, 0.0401952085, 0.0487133984, 0.0229813665, 0.0247559894, 0.034960071, 0.0385093168, 0.0331854481;0.036823425, 0.0176574978, 0.0262644188, 0.0464951198, 0.0229813665, 0.0312333629, 0.0455190772, 0.0246672582, 0.0314108252, 0.036823425, 0.0177462289, 0.0536823425, 0.0211180124, 0.0402839397, 0.0401952085, 0.0401952085, 0.0317657498;0.0386867791, 0.0228926353, 0.0279503106, 0.0555456965, 0.0279503106, 0.0212067436, 0.0194321207, 0.0335403727, 0.0123336291, 0.0367346939, 0.0477373558, 0.0333629104, 0.0527950311, 0.0351375333, 0.0382431233, 0.0335403727, 0.0209405501;0.0352262644, 0.0211180124, 0.0477373558, 0.0350488021, 0.0686779059, 0.0327417924, 0.0401064774, 0.0299911269, 0.0488021295, 0.0314995563, 0.0244897959, 0.0386867791, 0.0229813665, 0.0398402839, 0.0362910382, 0.0332741792, 0.0228926353;0.0264418811, 0.0451641526, 0.0585625555, 0.0246672582, 0.028216504, 0.0367346939, 0.0229813665, 0.0327417924, 0.050310559, 0.0419698314, 0.0367346939, 0.0510204082, 0.02635315, 0.0471162378, 0.0333629104, 0.0331854481, 0.0382431233]
functionname = 'test_wdps_kernel'
kernelname = 'WeightedDegreePositionStringKernel'
alphabet= 'DNA' 
seqlen= 60 
degree= 20 
