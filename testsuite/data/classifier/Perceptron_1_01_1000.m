classifier_accuracy = 1e-06;
init_random = 42;
name = 'Perceptron';
classifier_num_threads = 1;
data_train = [0.798631369, 0.953067261, 0.228109759, 0.66331808, 0.727314817, 0.0838561413, 0.120813896, 0.711150463, 0.50293506, 0.0717371218, 0.379240303, 0.884283192, 0.234483261, 0.511822712, 0.953198229, 0.383448384, 0.0423183058, 0.037921391, 0.332610992, 0.100661009, 0.273968617, 0.214968869;0.948499565, 0.282050782, 0.328040242, 0.597924333, 0.348850424, 0.600718512, 0.619367085, 0.0716287524, 0.753669715, 0.945739352, 0.123182894, 0.0562409389, 0.109085335, 0.869057069, 0.574573422, 0.330190231, 0.524460057, 0.872422584, 0.615482615, 0.154031832, 0.25403996, 0.429256284;0.326168215, 0.122179265, 0.176616382, 0.793042927, 0.0150865997, 0.93861455, 0.659743667, 0.0415413468, 0.630047348, 0.621638005, 0.626183281, 0.144329294, 0.115897541, 0.355277632, 0.409508207, 0.0401294208, 0.302047265, 0.556888742, 0.245412458, 0.697403574, 0.730478794, 0.048540279;0.384754371, 0.596362318, 0.0303225969, 0.423709894, 0.779930123, 0.513431873, 0.881412071, 0.311357988, 0.580262485, 0.27013654, 0.716788742, 0.50595862, 0.968048616, 0.144870953, 0.477513975, 0.0376355934, 0.462212927, 0.848919791, 0.630497613, 0.293183462, 0.189298851, 0.569867697;0.927429523, 0.628615198, 0.36775326, 0.334376762, 0.924691183, 0.893433934, 0.203835206, 0.576028438, 0.930542655, 0.341008166, 0.582076099, 0.193857455, 0.41352626, 0.574513597, 0.0311949276, 0.251458637, 0.423974626, 0.304299261, 0.0118191998, 0.0606822333, 0.638108539, 0.0634219805;0.685305472, 0.17218921, 0.362337798, 0.753245127, 0.420214084, 0.712394897, 0.692842991, 0.970847017, 0.183494612, 0.527030716, 0.284792915, 0.252183103, 0.451380654, 0.450378326, 0.29081347, 0.397549001, 0.70783126, 0.0230573646, 0.333450948, 0.687364286, 0.559339512, 0.732861296;0.257612953, 0.907433847, 0.230094062, 0.181648387, 0.864397489, 0.0749978614, 0.957474381, 0.101015638, 0.704270761, 0.165868469, 0.455854686, 0.855399628, 0.554328662, 0.0686223343, 0.453777875, 0.242193467, 0.316742966, 0.649848761, 0.748223416, 0.230193998, 0.873950782, 0.157327899;0.962177232, 0.810627051, 0.529274385, 0.272967985, 0.813163718, 0.633402327, 0.498178894, 0.645574231, 0.0390284326, 0.426084218, 0.558962095, 0.446276788, 0.0607126863, 0.950809427, 0.786892616, 0.99847424, 0.392552228, 0.212784439, 0.49725254, 0.535343319, 0.79694779, 0.27280174;0.77760819, 0.301793235, 0.49346788, 0.411915248, 0.976112058, 0.978583917, 0.52917419, 0.998713669, 0.513557099, 0.492708608, 0.965133215, 0.868874302, 0.172116579, 0.14835797, 0.593327133, 0.369722947, 0.58628549, 0.933313302, 0.209799088, 0.370804073, 0.255548488, 0.290740735;0.000231389991, 0.710748914, 0.403221842, 0.169261018, 0.34017294, 0.193037001, 0.354218699, 0.30573094, 0.983073226, 0.729485133, 0.581043522, 0.314428081, 0.343375942, 0.647500901, 0.943605597, 0.395826093, 0.0710950081, 0.230500231, 0.581472329, 0.5374008, 0.444128045, 0.590465951;0.851364034, 0.554121987, 0.42872766, 0.251524449, 0.865502939, 0.795196237, 0.475404874, 0.834918552, 0.312817577, 0.492490835, 0.714286459, 0.444067709, 0.872704273, 0.477192314, 0.255142325, 0.547725843, 0.582908241, 0.617216954, 0.339668397, 0.533403626, 0.407768678, 0.850333434];
classifier_labels = [1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1];
data_type = 'double';
data_test = [0.527054626, 0.350279911, 0.351291146, 0.808505329, 0.355749673, 0.88229743, 0.755955143, 0.588957123, 0.712898214, 0.520163416, 0.300212796, 0.476366312, 0.758999155, 0.269589015, 0.725179209, 0.3058006, 0.0047297619, 0.692976063, 0.649130032, 0.120562511, 0.0816963298, 0.0566346556, 0.362892543, 0.920731239, 0.682919144, 0.113928452, 0.749815963, 0.606414055, 0.0651056568, 0.622765003, 0.568966856, 0.774704835, 0.993462926, 0.157183881;0.285627736, 0.240565358, 0.0456415456, 0.848310234, 0.594332693, 0.641819155, 0.709209788, 0.478581068, 0.899417213, 0.504457925, 0.552537011, 0.539218436, 0.560710156, 0.774549595, 0.118704253, 0.611218977, 0.270838074, 0.402447451, 0.0251924986, 0.337075527, 0.21462882, 0.52986321, 0.594959341, 0.708292321, 0.597612045, 0.694636965, 0.283583838, 0.193876735, 0.519203913, 0.925794128, 0.218519149, 0.760299924, 0.294411892, 0.293452212;0.0474391852, 0.972085047, 0.535925367, 0.722983923, 0.151369034, 0.859628149, 0.190803463, 0.312403893, 0.95286514, 0.896367816, 0.44535067, 0.488700554, 0.744397025, 0.885170026, 0.0329126393, 0.979177876, 0.675190566, 0.0842245555, 0.660272593, 0.849656249, 0.259794781, 0.462441697, 0.343246681, 0.557815536, 0.0517632596, 0.802475923, 0.0241047743, 0.538174728, 0.586364889, 0.119784724, 0.537789693, 0.190045027, 0.563413012, 0.570113779;0.239626962, 0.593687379, 0.807629462, 0.44078279, 0.16420389, 0.247621905, 0.56719333, 0.793930898, 0.765031816, 0.382572842, 0.434550352, 0.244642216, 0.0899269102, 0.801235908, 0.327124369, 0.215428951, 0.340599192, 0.0305447873, 0.400984675, 0.794541496, 0.822082226, 0.73624743, 0.130351354, 0.96415111, 0.476195294, 0.74590629, 0.907489772, 0.0241286975, 0.857589876, 0.3928534, 0.730421241, 0.838346498, 0.706793899, 0.276223701;0.538454279, 0.176891674, 0.88887469, 0.255663608, 0.400228495, 0.349559889, 0.568874979, 0.491983149, 0.104461357, 0.935456238, 0.332012067, 0.802647291, 0.98679062, 0.602227266, 0.831182273, 0.28771086, 0.586622164, 0.603397457, 0.149916075, 0.967543701, 0.954135394, 0.313363798, 0.213297465, 0.0365533893, 0.328498053, 0.146548163, 0.793505309, 0.874098283, 0.390558934, 0.920869344, 0.509239395, 0.2541197, 0.390988478, 0.820776604;0.653614984, 0.802570666, 0.679808948, 0.306524451, 0.639622627, 0.00789648549, 0.712907035, 0.945979791, 0.437525128, 0.456430629, 0.484302581, 0.203361006, 0.179330722, 0.791202746, 0.183194347, 0.187572885, 0.625856187, 0.129704869, 0.595332683, 0.608878091, 0.796424414, 0.489729819, 0.225228378, 0.79521522, 0.937565764, 0.442129295, 0.513494196, 0.436086721, 0.236679101, 0.226094454, 0.272236453, 0.363887659, 0.259720543, 0.0499813651;0.971749634, 0.60377845, 0.120631805, 0.388526196, 0.77281597, 0.678647616, 0.659403048, 0.1963374, 0.85467542, 0.648332383, 5.46735585e-05, 0.658192869, 0.861623672, 0.508870778, 0.401069076, 0.205546419, 0.77297342, 0.321516087, 0.594098012, 0.786282966, 0.657152487, 0.927571744, 0.629675754, 0.73964866, 0.798007303, 0.035289898, 0.980423617, 0.418291602, 0.166116536, 0.282076965, 0.238636075, 0.707041637, 0.748236553, 0.132494679;0.295534218, 0.617690722, 0.0032999928, 0.73469007, 0.881225365, 0.140274981, 0.780792138, 0.729349431, 0.687202976, 0.128606424, 0.0225467587, 0.688562964, 0.154500919, 0.597445358, 0.0677335691, 0.09215329, 0.0583047163, 0.0198642899, 0.0646143435, 0.931729126, 0.801089318, 0.360888658, 0.545375548, 0.834074051, 0.378778855, 0.00204782261, 0.71549172, 0.671137958, 0.444874881, 0.0303445642, 0.299608854, 0.974429521, 0.0412554709, 0.336864216;0.0750296219, 0.741611576, 0.597884237, 0.911290831, 0.00670453652, 0.818064256, 0.782385389, 0.535186525, 0.654814896, 0.0488658463, 0.912418331, 0.656169918, 0.295993515, 0.94222824, 0.901188295, 0.989784942, 0.0852019017, 0.416970306, 0.403156286, 0.102065444, 0.826680019, 0.0647020233, 0.539125907, 0.110712777, 0.408724551, 0.898641926, 0.0922589638, 0.538316831, 0.657538593, 0.198382875, 0.406586004, 0.221935929, 0.317060275, 0.351068698;0.532687321, 0.347133431, 0.0822104018, 0.288901696, 0.352192895, 0.134002465, 0.718349076, 0.453475672, 0.464944866, 0.612435655, 0.496208913, 0.0532170948, 0.0989826745, 0.727542788, 0.708485702, 0.98301459, 0.39056932, 0.000676347275, 0.260770159, 0.693887983, 0.648439118, 0.101990628, 0.751982429, 0.228798497, 0.430408694, 0.283684403, 0.136596487, 0.195236899, 0.819496326, 0.665560973, 0.466047737, 0.846923713, 0.0672160153, 0.736619136;0.993991082, 0.702283731, 0.30375012, 0.959573164, 0.647807882, 0.375121941, 0.273438862, 0.82856953, 0.761915349, 0.804934323, 0.871284983, 0.9757469, 0.0799311071, 0.840853547, 0.453985165, 0.572264248, 0.15497502, 0.499357362, 0.525838135, 0.843085752, 0.829287688, 0.588617165, 0.387364937, 0.522651594, 0.729606795, 0.0915755123, 0.897111408, 0.661503974, 0.236691782, 0.995851589, 0.0482455184, 0.843321523, 0.864134521, 0.820063736];
classifier_bias = -1.1;
classifier_learn_rate = 0.1;
classifier_type = 'perceptron';
classifier_max_iter = 1000;
feature_class = 'simple';
classifier_classified = [0.514112527, -0.872111463, 0.924628482, -2.86199683, 0.442797709, -2.39696316, 0.0795386782, 0.340306285, -0.66875775, 0.407377687, -2.44401264, -1.83530997, -0.00181245236, -0.588541855, -1.79442954, -3.61854069, 1.45921101, -1.60500685, -0.509120288, 1.60620815, 0.552835624, 1.88763839, -1.75521869, 1.95799702, 0.76718361, -0.883760679, 2.3080046, -1.79698736, -0.36782834, -0.425669887, 0.323922903, 0.46947174, 0.0282965228, -1.84729933];
feature_type = 'Real';
