data_type = 'double';
classifier_C = 30;
classifier_accuracy = 0.001;
init_random = 42;
name = 'SVMSGD';
classifier_num_threads = 1;
data_train = [0.909809563, 0.587420499, 0.71275604, 0.416237142, 0.144368542, 0.29495884, 0.661871597, 0.374279396, 0.487827499, 0.971312272, 0.130971299, 0.989836112, 0.224842831, 0.904883696, 0.365526021, 0.819653041, 0.333415127, 0.356884461, 0.00216546219, 0.949216437, 0.213708848, 0.469088375;0.920556407, 0.679476478, 0.5063059, 0.371486541, 0.787602028, 0.365500882, 0.992237568, 0.739118936, 0.105528629, 0.975495216, 0.161291487, 0.00433617062, 0.513190251, 0.834691675, 0.715759056, 0.450417138, 0.547903832, 0.174723074, 0.835495375, 0.183146805, 0.443140842, 0.243585488;0.0450454418, 0.895065199, 0.186699727, 0.626577192, 0.976316683, 0.547555753, 0.314924065, 0.861140608, 0.593344517, 0.256546299, 0.717171499, 0.180645383, 0.46380024, 0.972748777, 0.955438814, 0.0278084452, 0.207971351, 0.445328034, 0.557419411, 0.130130073, 0.247648604, 0.0463267981;0.561016922, 0.0325777345, 0.0418814181, 0.935396526, 0.340696251, 0.384366138, 0.300975977, 0.0898533434, 0.240068998, 0.552037955, 0.363196303, 0.0288213104, 0.714773477, 0.888584176, 0.0794423468, 0.808683087, 0.939236209, 0.528945708, 0.720993009, 0.263865271, 0.428558605, 0.34309135;0.184159277, 0.423715932, 0.294134775, 0.753741396, 0.791978484, 0.00667400541, 0.996586216, 0.633501751, 0.517829471, 0.893105432, 0.716288552, 0.388882023, 0.64149921, 0.604318315, 0.259864349, 0.412000047, 0.100538374, 0.943151536, 0.730988711, 0.276599136, 0.951155176, 0.94884773;0.0141228138, 0.828441387, 0.260363396, 0.744574184, 0.72907607, 0.0242528503, 0.273115346, 0.358245494, 0.0776034029, 0.215485021, 0.667204169, 0.836402253, 0.763665152, 0.171685743, 0.603955451, 0.708225601, 0.649319837, 0.899070894, 0.973048101, 0.580418025, 0.572344231, 0.459953357;0.0524526171, 0.0660546467, 0.152718269, 0.642542395, 0.403854248, 0.330913956, 0.969878762, 0.824528981, 0.339501777, 0.38876976, 0.637439791, 0.143001354, 0.89135849, 0.31013589, 0.429855595, 0.535506332, 0.385326463, 0.996192177, 0.766382341, 0.260688214, 0.033435138, 0.191576695;0.750608232, 0.452545687, 0.723743484, 0.528602632, 0.572726455, 0.797728351, 0.898015936, 0.0987281614, 0.956760064, 0.113531171, 0.18409346, 0.609782082, 0.481032224, 0.779809517, 0.150270804, 0.639475123, 0.187334157, 0.338130841, 0.0919768187, 0.957883592, 0.390510578, 0.68062379;0.791455422, 0.0293030677, 0.746881187, 0.985900268, 0.428172542, 0.196457953, 0.678512639, 0.600627514, 0.345069112, 0.913161291, 0.516374211, 0.0808168173, 0.0421336964, 0.2185442, 0.0119418727, 0.669680218, 0.209850118, 0.780031461, 0.481807122, 0.37877357, 0.297842399, 0.619214459;0.914233071, 0.488505564, 0.698535882, 0.253585041, 0.323512574, 0.319010295, 0.615655209, 0.465541423, 0.308809477, 0.884883119, 0.240725512, 0.504842817, 0.0361888535, 0.673471729, 0.542644907, 0.646730953, 0.677010818, 0.53219517, 0.364637398, 0.352078303, 0.683397457, 0.423376073;0.553632429, 0.4468172, 0.628228939, 0.226871833, 0.868035887, 0.180505551, 0.243770083, 0.0392598666, 0.566929681, 0.54680526, 0.646203648, 0.0792966528, 0.392670715, 0.328323899, 0.383716949, 0.24702123, 0.060400309, 0.43276716, 0.600571606, 0.318043768, 0.197084355, 0.520591154];
classifier_labels = [-1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1];
classifier_epsilon = 0.0001;
data_test = [0.437990129, 0.786128659, 0.636865919, 0.57457102, 0.739807004, 0.0977297092, 0.200166987, 0.785701837, 0.909721083, 0.860806901, 0.987843662, 0.389894036, 0.911037807, 0.921970427, 0.322857258, 0.651589023, 0.584168693, 0.263824199, 0.980013502, 0.614128735, 0.302710325, 0.460737347, 0.0109790564, 0.380235164, 0.7191138, 0.862638767, 0.578798297, 0.877067904, 0.836751506, 0.287648467, 0.899078213, 0.599743998, 0.29606019, 0.291000787;0.697110532, 0.693907951, 0.142496365, 0.795254389, 0.531953766, 0.428452889, 0.948634797, 0.218979467, 0.192774567, 0.548822654, 0.536948935, 0.389358711, 0.122773878, 0.35753484, 0.285822107, 0.967341836, 0.530272741, 0.740742878, 0.805453681, 0.610128909, 0.112698819, 0.149000482, 0.0996466191, 0.810599013, 0.537951211, 0.0639255326, 0.103873979, 0.882932933, 0.45174353, 0.20018458, 0.743833645, 0.410563473, 0.0191978316, 0.0396541126;0.406918123, 0.090056096, 0.342565246, 0.144695522, 0.958287924, 0.0653836788, 0.930717261, 0.652191039, 0.358952504, 0.848071738, 0.0885281208, 0.977684198, 0.932598603, 0.917362496, 0.0512504987, 0.341510711, 0.220938488, 0.685171295, 0.0393613653, 0.67748324, 0.100204061, 0.643554465, 0.354941785, 0.231937092, 0.671657216, 0.674868332, 0.666578641, 0.379336306, 0.274377266, 0.465611936, 0.967587239, 0.193683386, 0.493876757, 0.164176518;0.660513715, 0.994343297, 0.224047958, 0.195999693, 0.0915032554, 0.451699075, 0.104299616, 0.252397501, 0.574332408, 0.339449197, 0.558114396, 0.184312046, 0.81505849, 0.0232452549, 0.390476369, 0.283694773, 0.25636469, 0.281649948, 0.350801751, 0.688102717, 0.458527294, 0.429717688, 0.423213816, 0.557114455, 0.544578079, 0.453127484, 0.924436834, 0.540908391, 0.263603253, 0.907794813, 0.612468144, 0.739345702, 0.475087679, 0.703076202;0.751469981, 0.000276547913, 0.100786153, 0.139122473, 0.682920438, 0.0574608813, 0.793016804, 0.748877046, 0.313785035, 0.560211885, 0.166609189, 0.648056999, 0.929981304, 0.894392671, 0.0974704504, 0.878246965, 0.73602114, 0.632887177, 0.945627529, 0.285987449, 0.832489195, 0.980437453, 0.823965039, 0.862521191, 0.793530162, 0.491647562, 0.145668261, 0.0595975898, 0.960901817, 0.951377684, 0.322985497, 0.234868263, 0.167655963, 0.675173543;0.144171965, 0.544976672, 0.525062269, 0.432531376, 0.495425128, 0.943288118, 0.37956954, 0.53006784, 0.92153395, 0.385325152, 0.160166947, 0.201928322, 0.574536644, 0.512938364, 0.294315541, 0.885858183, 0.8302951, 0.0104388609, 0.0607578971, 0.3186034, 0.611835506, 0.571640073, 0.413652519, 0.133330933, 0.766487091, 0.366470882, 0.341689146, 0.366506845, 0.502876663, 0.19654233, 0.0631212632, 0.0198295853, 0.435908128, 0.596982213;0.0451655581, 0.352851625, 0.627030213, 0.378055676, 0.412271468, 0.447044344, 0.233532167, 0.576703643, 0.147913362, 0.308515293, 0.992430811, 0.438164767, 0.837943081, 0.0879886619, 0.457354896, 0.417885728, 0.139490039, 0.706340215, 0.343981457, 0.955118765, 0.46710989, 0.302483519, 0.885456299, 0.653795694, 0.592321679, 0.213532876, 0.374583413, 0.0102179995, 0.588337259, 0.831747762, 0.200349179, 0.673541954, 0.0638222219, 0.677508988;0.841533361, 0.554816293, 0.816558471, 0.581009513, 0.401858076, 0.326288107, 0.221855604, 0.452329511, 0.793502087, 0.481471763, 0.661302626, 0.727798829, 0.0304716137, 0.441680728, 0.883065679, 0.19641177, 0.798048264, 0.0151446975, 0.0839865447, 0.424943507, 0.1089071, 0.599108958, 0.346084376, 0.332485476, 0.654133557, 0.623735187, 0.810231192, 0.281280047, 0.802438886, 0.0917089506, 0.31661214, 0.0327264221, 0.276762395, 0.897598424;0.985955534, 0.668892024, 0.488966859, 0.455458104, 0.144203482, 0.438393873, 0.225128033, 0.620860879, 0.782616954, 0.59845908, 0.742017071, 0.619419701, 0.474415087, 0.780157164, 0.60164761, 0.395186351, 0.595730185, 0.766138094, 0.902681533, 0.776998775, 0.408161343, 0.9854942, 0.433755925, 0.411534684, 0.376605111, 0.0389518438, 0.349513817, 0.452484619, 0.265469537, 0.0996755879, 0.999071956, 0.993647067, 0.914958578, 0.0363511983;0.607051172, 0.956796411, 0.903147017, 0.114486512, 0.281320647, 0.208513967, 0.811337298, 0.26389034, 0.103266798, 0.643933629, 0.493854987, 0.0400099417, 0.41129315, 0.0041667288, 0.479309906, 0.82105078, 0.347383155, 0.882294913, 0.873087278, 0.765470218, 0.839825997, 0.926459324, 0.319849666, 0.597366087, 0.447307805, 0.0484780378, 0.873402497, 0.705370508, 0.0917276822, 0.995119719, 0.225508172, 0.954973043, 0.615341641, 0.938967125;0.651292562, 0.90184094, 0.842028784, 0.807585952, 0.704753099, 0.020790591, 0.512739738, 0.888361542, 0.426647722, 0.920064363, 0.299884428, 0.775844451, 0.605014515, 0.0784656553, 0.755724773, 0.934306884, 0.0718510968, 0.728424188, 0.274176022, 0.140368719, 0.33645941, 0.366977236, 0.44366866, 0.989880201, 0.651509233, 0.533183867, 0.0509691618, 0.80504416, 0.0260827244, 0.321003214, 0.224003857, 0.245443157, 0.901630442, 0.894083084];
classifier_bias = 0.812066006574;
classifier_bias_enabled = 'True';
classifier_type = 'linear';
feature_class = 'simple';
classifier_classified = [-5.85048805, -0.0898007538, -4.11542196, -2.17887363, -5.43666669, 3.37685711, -3.00532516, -5.92723384, 0.074257452, -5.64726237, -21.5186421, -7.35724163, -10.0684514, -11.0399506, -2.56625926, 1.4971724, -3.49429085, -11.3911955, -17.530754, -18.0206263, -1.21939273, -6.09723227, -6.11017187, -6.80859138, -4.9789014, -4.20066201, -8.46891137, -0.40616273, -15.0892816, -10.1034903, -15.0760273, -15.1611872, 6.23506101, 1.16009854];
feature_type = 'Real';
