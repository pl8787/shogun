distance_name = 'EuclidianDistance';
classifier_accuracy = 1e-08;
init_random = 42;
name = 'KNN';
classifier_num_threads = 1;
data_train = [0.319394898, 0.331296514, 0.547299402, 0.0169629411, 0.486775898, 0.688017491, 0.629623079, 0.618388264, 0.450626817, 0.348459937, 0.674683267, 0.448596971, 0.248647315, 0.763084198, 0.15782382, 0.229891184, 0.506629903, 0.882991327, 0.198525303, 0.894816825, 0.0284521506, 0.0634051156;0.21708455, 0.805398327, 0.513326089, 0.576164071, 0.958186379, 0.396252569, 0.267618682, 0.294420742, 0.022967254, 0.337841668, 0.996342021, 0.306169632, 0.988877925, 0.988145128, 0.247572817, 0.0751092151, 0.287184303, 0.808883029, 0.171743638, 0.793821806, 0.486222521, 0.469340323;0.535921098, 0.29076578, 0.685143369, 0.695798256, 0.319653805, 0.666588559, 0.312142245, 0.556671399, 0.494217821, 0.210123293, 0.0964021506, 0.247003835, 0.256523592, 0.472223086, 0.589318059, 0.0585253021, 0.693991279, 0.269754762, 0.638649919, 0.147937112, 0.518219046, 0.0740226691;0.908939725, 0.113334688, 0.673785901, 0.204777919, 0.904260737, 0.678988491, 0.70879753, 0.123781061, 0.732771116, 0.73540916, 0.118826157, 0.221792782, 0.97121706, 0.835328863, 0.445922508, 0.357911711, 0.592575726, 0.21018223, 0.659389733, 0.0578158893, 0.562475752, 0.548425936;0.3557628, 0.666094198, 0.448934355, 0.3627193, 0.989462743, 0.776694941, 0.700180067, 0.537803226, 0.687892109, 0.757108848, 0.494481578, 0.0911934538, 0.092717778, 0.408196564, 0.117762138, 0.125203941, 0.255455475, 0.761808211, 0.715466979, 0.31192412, 0.310710105, 0.106715414;0.760455126, 0.597666821, 0.385298553, 0.0929071759, 0.0246635563, 0.544324151, 0.482092399, 0.74045143, 0.683862153, 0.165076817, 0.772680522, 0.974709895, 0.910754985, 0.576786015, 0.620537837, 0.790951136, 0.423483982, 0.833856866, 0.674638176, 0.945892846, 0.120774375, 0.747105063;0.779064613, 0.16678056, 0.171096471, 0.210110284, 0.417183377, 0.0746988292, 0.412147142, 0.0264644097, 0.653565214, 0.684218716, 0.58263318, 0.224843666, 0.178019314, 0.924121249, 0.201283185, 0.671774059, 0.456704212, 0.0482690007, 0.583303382, 0.886859733, 0.348519053, 0.215321574;0.0632723209, 0.585692939, 0.817287123, 0.676065433, 0.0384381673, 0.717089913, 0.743663572, 0.834818912, 0.653907085, 0.158407749, 0.236436971, 0.306054071, 0.617399664, 0.468892421, 0.158358373, 0.449481926, 0.000404516664, 0.46670564, 0.712263292, 0.802954958, 0.024297778, 0.877118999;0.782277821, 0.307684745, 0.701301453, 0.906126613, 0.416442285, 0.0517301643, 0.812559157, 0.126242227, 0.117363365, 0.051659623, 0.263181866, 0.929496034, 0.759452608, 0.728365894, 0.950337548, 0.262723931, 0.658864417, 0.46883351, 0.0438246869, 0.950996525, 0.36254739, 0.951016106;0.132189448, 0.394383509, 0.79924684, 0.159388947, 0.67916535, 0.542060915, 0.655190857, 0.0630935646, 0.491234535, 0.231784233, 0.595477742, 0.711964043, 0.35698138, 0.843902283, 0.45484393, 0.705996674, 0.43961438, 0.788857608, 0.727956572, 0.814054725, 0.615987248, 0.650061232;0.439074765, 0.751729094, 0.438845177, 0.0802902479, 0.947662473, 0.484092466, 0.630056356, 0.930257243, 0.585721903, 0.765460624, 0.436998587, 0.447763751, 0.741524045, 0.353379395, 0.177703906, 0.847593967, 0.34891931, 0.154220008, 0.885482091, 0.537511934, 0.57514407, 0.521016398];
classifier_labels = [1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1];
data_test = [0.277310268, 0.50320802, 0.439247632, 0.34546993, 0.188667918, 0.567539289, 0.1165126, 0.940408294, 0.818513451, 0.166707317, 0.803915469, 0.788075436, 0.106317179, 0.501962369, 0.723424887, 0.919466792, 0.303094782, 0.0871082164, 0.00329009273, 0.635513413, 0.0201000536, 0.517270919, 0.367806855, 0.0461362157, 0.865122633, 0.185935876, 0.857363245, 0.915753864, 0.681059755, 0.406077033, 0.169725438, 0.512997539, 0.0826753236, 0.589557016;0.921956018, 0.360802331, 0.462403495, 0.715944547, 0.899472478, 0.939894755, 0.495974639, 0.912128059, 0.317965533, 0.220768145, 0.994259297, 0.246688118, 0.645375236, 0.587057685, 0.287486376, 0.523675321, 0.575929755, 0.572763215, 0.473402055, 0.785107805, 0.672643866, 0.368873535, 0.545680163, 0.903245199, 0.362891067, 0.271607869, 0.110644239, 0.851981782, 0.544819129, 0.944122452, 0.196225658, 0.0681113707, 0.563116092, 0.0501077966;0.441576683, 0.861161768, 0.51897803, 0.186118964, 0.839325315, 0.296926946, 0.174147251, 0.294144613, 0.345587107, 0.0614799107, 0.375523564, 0.317471075, 0.995794256, 0.432094955, 0.333232386, 0.69139543, 0.82932794, 0.928026378, 0.812242661, 0.539399491, 0.0326965402, 0.936283379, 0.200482688, 0.687154523, 0.36480964, 0.650422768, 0.139317312, 0.52524461, 0.104954262, 0.49142593, 0.975387047, 0.0714726762, 0.400904467, 0.0908674776;0.848109705, 0.256677199, 0.308407819, 0.234400558, 0.904831087, 0.22295632, 0.671065631, 0.470560693, 0.966273031, 0.939202442, 0.845815369, 0.82145792, 0.791758822, 0.309997015, 0.310625945, 0.604952993, 0.380363664, 0.845870559, 0.336967996, 0.937997989, 0.392541223, 0.748821514, 0.189420382, 0.0200196392, 0.927365539, 0.107746388, 0.512916299, 0.244746325, 0.232438984, 0.00296148933, 0.0668432144, 0.725345482, 0.263019141, 0.723636526;0.598804881, 0.392542726, 0.809991088, 0.258831956, 0.655171466, 0.0870459278, 0.354110743, 0.556539876, 0.85486558, 0.348842976, 0.287527913, 0.478236521, 0.650177458, 0.27924574, 0.25700359, 0.730738053, 0.550976406, 0.573324523, 0.72339433, 0.718235061, 0.129696446, 0.526763546, 0.712371919, 0.0687314592, 0.979307175, 0.709724408, 0.0747153305, 0.966315492, 0.437297904, 0.896780475, 0.569417128, 0.546505001, 0.375288085, 0.615719258;0.853868064, 0.816066066, 0.532852607, 0.0573533505, 0.850809979, 0.302968743, 0.843407117, 0.89428483, 0.377387182, 0.936252241, 0.376735199, 0.939132443, 0.867099184, 0.747900473, 0.233720615, 0.436920761, 0.928144989, 0.233189362, 0.951181637, 0.962410124, 0.363358556, 0.764643106, 0.155077238, 0.850669649, 0.0604295812, 0.325178747, 0.395166944, 0.70797749, 0.508936074, 0.819631003, 0.695239001, 0.165853044, 0.946110184, 0.0620529906;0.699073798, 0.303914815, 0.606926962, 0.44954532, 0.102587711, 0.547004507, 0.954537836, 0.373558011, 0.160891492, 0.332227525, 0.850552814, 0.965276284, 0.101851556, 0.357739049, 0.392130729, 0.0572909079, 0.358689041, 0.353774884, 0.688641665, 0.116033081, 0.0730725245, 0.708616263, 0.128743504, 0.386088404, 0.222197424, 0.124508628, 0.954183207, 0.243567193, 0.190320886, 0.265257892, 0.772441185, 0.334479898, 0.689683899, 0.280144177;0.0716283647, 0.696737199, 0.329280761, 0.766879435, 0.190832977, 0.265794638, 0.0600708867, 0.119665528, 0.611745933, 0.667959403, 0.260060739, 0.985091749, 0.646366272, 0.298390974, 0.485440171, 0.234478051, 0.412947394, 0.440985503, 0.879842258, 0.44784078, 0.0216088013, 0.494935223, 0.710771773, 0.437069637, 0.792855609, 0.677317359, 0.37353078, 0.0692673882, 0.7983323, 0.657658427, 0.874772975, 0.534764506, 0.92803764, 0.430038627;0.280068433, 0.330786857, 0.872421355, 0.0470715179, 0.437732473, 0.877587674, 0.97410621, 0.607623418, 0.559945867, 0.379490051, 0.362413465, 0.893773617, 0.511717454, 0.561799807, 0.840167852, 0.0766164612, 0.118451023, 0.376252371, 0.193758407, 0.416920915, 0.362774467, 0.0867660302, 0.233019384, 0.0187815469, 0.284748695, 0.184742896, 0.292819038, 0.326155433, 0.166306439, 0.956842249, 0.469891392, 0.0355117167, 0.261682873, 0.111061208;0.773310778, 0.0277116783, 0.595183805, 0.535015628, 0.0868717329, 0.0789379226, 0.585765473, 0.62029537, 0.132813428, 0.290421127, 0.311832955, 0.41102226, 0.270133768, 0.0737627817, 0.162715072, 0.311956322, 0.227597443, 0.961278397, 0.522447598, 0.15132826, 0.519600402, 0.191054521, 0.389732224, 0.491340289, 0.446148096, 0.714548634, 0.58488883, 0.983201002, 0.629042739, 0.362559143, 0.0410204045, 0.315639474, 0.142566501, 0.161392203;0.0640531149, 0.159426734, 0.785289893, 0.390465156, 0.554240825, 0.262481609, 0.99817301, 0.169694572, 0.190239361, 0.232859021, 0.287711202, 0.692155166, 0.418602843, 0.149964952, 0.478940957, 0.438615879, 0.0508521426, 0.50917173, 0.612649933, 0.315568256, 0.180315054, 0.268974369, 0.095281109, 0.937058243, 0.0755929346, 0.754938826, 0.539830704, 0.65410662, 0.855707482, 0.247810775, 0.880722105, 0.995158297, 0.051364144, 0.881456218];
data_type = 'double';
accuracy = 1e-08;
classifier_type = 'knn';
classifier_k = 3;
data_class = 'rand';
feature_class = 'simple';
classifier_classified = [-1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1];
feature_type = 'Real';
