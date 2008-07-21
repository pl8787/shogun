distance_name = 'EuclidianDistance';
accuracy = 1e-08;
init_random = 42;
clustering_k = 3;
name = 'KMeans';
data_type = 'double';
data_train = [1.06132836, 1.5614079, 0.882579114, 0.910679008, 1.60798054, 1.04214731, 1.3933516, 0.659771425, 0.591369008, 0.329410691, 0.393923553, 1.43365247, 0.18372773, 0.021593512, 0.45718976, 1.14182191, 0.285816555, 1.11331849, 0.636367607, 0.301794044, 0.578010078, 0.393700539, 1.8730616, 0.325415104, 0.179327586, 1.37346864, 0.620791698, 0.934356317, 0.589758859, 0.65518408, 0.824867767, 0.815233546, 1.19725891;1.38787951, 0.744691172, 0.0849341304, 0.716975076, 0.677161306, 0.173745223, 0.764503129, 1.19832385, 0.807304461, 0.816049192, 1.43889462, 1.49535131, 1.17743918, 1.98898811, 0.184291886, 1.45569815, 0.0392505629, 0.506577483, 0.88069051, 0.623707284, 0.0244315996, 0.959113902, 0.243992954, 0.743960697, 1.93392153, 0.00232902252, 0.438669959, 0.981324973, 1.38727432, 1.20393331, 0.114967106, 0.0322006606, 1.49312669;0.708242927, 0.443712049, 0.00993714066, 1.55416733, 0.520357598, 1.79908118, 0.603931873, 0.0277461705, 0.246571415, 1.72773196, 1.97577364, 1.96050889, 0.273439922, 1.43440281, 0.26954975, 1.20177041, 0.387169028, 0.768537998, 0.363211976, 0.211166745, 0.253331974, 0.552855556, 1.85107706, 1.80609899, 0.552279515, 0.514426976, 0.659127346, 0.91803426, 1.89389971, 0.674208602, 1.02667461, 0.740423864, 0.182990978;0.742495564, 0.522553332, 1.8634851, 0.980822852, 0.310211854, 1.1624969, 0.707781323, 0.91514162, 0.236244638, 0.0797969155, 0.803477403, 0.816454456, 1.16685229, 0.663784527, 1.18693614, 0.278425943, 1.24074919, 0.269110938, 0.854943325, 0.945902459, 0.592819835, 0.278045994, 1.61653104, 1.89885413, 0.1275357, 0.624749297, 0.860334611, 1.63717886, 0.239333708, 1.5091094, 1.76240482, 1.78748155, 0.578683912;1.99384208, 1.35231871, 0.60316138, 0.174927507, 0.208017831, 0.899525938, 1.34218057, 1.97355236, 1.38329568, 0.790663521, 1.10280218, 0.662661714, 1.80406914, 1.59771313, 0.285054769, 0.117851421, 0.36004373, 0.705123373, 0.703217232, 0.0886944751, 1.27724492, 0.326437876, 1.28407267, 0.606290314, 0.620304831, 0.76933608, 0.0745303068, 0.307318797, 0.253944014, 0.156141272, 1.77132667, 0.952040823, 0.870305581];
feature_class = 'simple';
data_test = [0.690168753, 0.772318679, 0.926018169, 0.326245624, 0.683416544, 0.971250831, 0.67367974, 0.383951971, 0.507610098, 1.27540124, 1.52312696, 0.417638592, 0.318143218, 0.882950448, 0.738026042, 0.629901111, 0.635375995, 0.385352865, 0.449143926, 1.49462279, 0.471729886, 1.04133907, 1.79905365, 0.34800174, 1.14903795, 1.44472494, 0.874609193, 1.22152813, 1.46742538, 0.3599067, 0.340955425, 1.99423574, 1.6351752, 0.378085727, 1.43723097, 0.0896112516, 0.106708629, 1.60489894, 0.539964657, 0.573059183, 0.19054037, 0.767290979, 1.32542257, 0.585810517, 0.183560716, 1.4239828, 0.443055722, 0.405976613, 1.53841735, 0.871775055, 1.51889287;0.641226298, 0.0257568597, 0.664784546, 0.436191358, 0.344712244, 1.90286147, 0.164044936, 0.871320455, 0.0124298696, 1.68852056, 0.733169395, 0.257393981, 0.246772054, 0.183983758, 0.0576767828, 0.233418903, 1.69296676, 0.15820421, 1.1369485, 0.253344805, 0.668454243, 0.907555366, 0.775693794, 0.0637726912, 1.2577465, 0.152887706, 1.61752266, 1.89776237, 1.83901518, 1.99900044, 1.70819969, 0.00632852432, 0.683929839, 1.98649715, 0.960335354, 1.27695783, 0.587147519, 1.67360388, 0.664095951, 1.7531653, 1.61808053, 0.0899622151, 0.348619051, 0.624572286, 0.893454759, 0.41065223, 0.454126325, 0.533322367, 0.0989213091, 1.91284036, 1.68978587;0.37364426, 0.370595434, 1.41412121, 0.779165356, 0.434698471, 0.55336531, 0.879590548, 0.30527679, 0.702925569, 0.880669614, 0.232628388, 0.999737926, 0.231227003, 1.1867904, 1.40049148, 1.74322057, 0.584275796, 0.995135302, 0.230837827, 1.28276381, 0.385226332, 0.815103884, 0.631041585, 1.60258805, 0.25350012, 0.718261869, 1.3424061, 1.57140652, 0.966276806, 1.05317233, 0.655092618, 1.5899541, 0.231265336, 1.23694497, 0.578685965, 1.09386726, 1.88214038, 0.981717055, 0.566885912, 0.226041477, 0.172315565, 0.51013704, 1.74143414, 0.900392761, 0.744251034, 0.677140325, 1.06556943, 1.8205273, 0.220626287, 1.32629408, 0.816192285;1.78248478, 0.0324467348, 0.617357518, 1.87750903, 0.366138211, 1.03049257, 0.495030811, 0.540623554, 1.16191523, 1.06782287, 0.804028268, 0.751979968, 1.45295076, 1.2734332, 0.0637216077, 0.311028868, 0.832302945, 0.616034603, 0.208242929, 0.130998677, 0.422969615, 0.0609081872, 1.26225842, 0.0347362473, 1.249054, 0.52110971, 1.56190171, 0.966034605, 0.664985265, 0.706904727, 0.703194952, 0.504774024, 1.30665311, 1.4984445, 1.56358164, 1.10048428, 0.514405395, 0.689570187, 0.400417737, 0.699779025, 0.0819951431, 0.32733538, 0.908973156, 1.86533816, 0.511977634, 0.126188875, 1.89235637, 0.427459173, 0.504463485, 0.00110830044, 1.00105266;0.137581572, 0.0280943727, 1.06180395, 0.0259809812, 0.457808553, 0.970360716, 0.905992934, 0.479243063, 0.27842833, 1.51118486, 1.11739128, 1.72935803, 0.916717996, 0.0153701984, 1.31561732, 0.203223322, 1.80758227, 0.258640478, 1.68661291, 0.804038425, 0.134071967, 0.347967591, 1.77817575, 1.97164831, 0.876098532, 1.94360584, 0.664694107, 0.999468086, 0.125983721, 1.80345553, 0.215260166, 1.32154189, 1.11096845, 0.80623682, 0.0177522046, 1.12376055, 0.460866582, 1.33824638, 0.501714484, 0.31784281, 0.0663070076, 0.460272766, 0.897766468, 0.273529678, 1.82889565, 0.630129726, 1.86271324, 0.0153948481, 0.63364452, 0.514649698, 0.42707364];
clustering_radi = [1.22836972, 1.22836972, 1.36937937];
feature_type = 'Real';
clustering_centers = [0.761127929, 0.767400379, 0.893318944;0.474186899, 1.16111448, 0.941326837;0.488738487, 0.549999227, 1.75223435;0.997218467, 0.742347773, 0.875132595;0.486883062, 1.47089088, 0.654748809];
data_class = 'rand';
clustering_accuracy = 1e-08;
