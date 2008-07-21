accuracy = 1e-07;
dm_train = [7.66752689e-09, 0.887528233, 1.11407735, 1.5797209, 1.48348982, 1.69882908, 1.29587395, 1.601842, 0.998156051, 1.45575657, 1.32933079;0.887528233, 6.39013023e-09, 1.46985722, 1.62016495, 1.6578923, 1.9134687, 1.33352434, 1.47176788, 0.916563565, 1.55884437, 1.42776602;1.11407735, 1.46985722, 2.00124338e-08, 1.74380881, 1.58579815, 1.8459899, 1.18421686, 1.66225183, 1.12614071, 1.482816, 1.51246495;1.5797209, 1.62016495, 1.74380881, 1.79958659e-08, 1.46102182, 1.25339884, 1.70251358, 1.63682311, 1.68398853, 1.38628746, 1.02286802;1.48348982, 1.6578923, 1.58579815, 1.46102182, 1.37665733e-08, 1.36027189, 1.53148275, 1.41449014, 1.38833019, 0.704892429, 1.05111144;1.69882908, 1.9134687, 1.8459899, 1.25339884, 1.36027189, 2.60791109e-08, 1.71522672, 1.38276854, 1.88984411, 1.3914418, 0.797344463;1.29587395, 1.33352434, 1.18421686, 1.70251358, 1.53148275, 1.71522672, 1.48355331e-08, 1.8377829, 1.19777874, 1.57184834, 1.24422783;1.601842, 1.47176788, 1.66225183, 1.63682311, 1.41449014, 1.38276854, 1.8377829, 2.6374687e-08, 1.38496734, 1.22468965, 1.21412752;0.998156051, 0.916563565, 1.12614071, 1.68398853, 1.38833019, 1.88984411, 1.19777874, 1.38496734, 1.5528905e-08, 1.48564233, 1.39599696;1.45575657, 1.55884437, 1.482816, 1.38628746, 0.704892429, 1.3914418, 1.57184834, 1.22468965, 1.48564233, 2.59133289e-08, 1.08609279;1.32933079, 1.42776602, 1.51246495, 1.02286802, 1.05111144, 0.797344463, 1.24422783, 1.21412752, 1.39599696, 1.08609279, 1.64438445e-08];
init_random = 42;
name = 'SparseEuclidianDistance';
data_type = 'double';
dm_test = [0.739073135, 1.26889938, 1.27220504, 1.17737539, 1.28379712, 1.11760105, 1.68162111, 1.49078471, 1.53142219, 1.18874066, 1.1625794, 1.328574, 1.4495979, 1.67434383, 1.32115495, 1.3783886, 1.26839785;0.958314467, 1.42821462, 1.04733801, 1.29629861, 1.21840077, 1.17357513, 1.67465062, 1.76516571, 1.62466242, 1.05856727, 1.36697705, 1.63500168, 1.5334775, 1.71298348, 1.64411304, 1.64606219, 1.48618778;1.28540226, 1.41269161, 1.64650845, 1.50145851, 1.72528896, 1.25267724, 1.7862868, 1.36162322, 1.61796176, 1.67402494, 1.5678835, 1.15970734, 1.73336724, 1.56485998, 1.43949533, 1.41745156, 1.83388109;1.31455673, 1.28604681, 1.25853033, 1.25030455, 1.1116409, 1.35180976, 1.4966088, 1.25981799, 0.89204746, 1.21018247, 1.82266599, 1.00187522, 1.33208173, 1.2979371, 1.74658181, 0.966695239, 1.62478273;0.988621014, 1.20021488, 1.55365338, 1.36357874, 1.5451491, 1.32688882, 1.61083733, 1.10726897, 1.40141052, 1.42272733, 1.01969072, 1.22786948, 1.52214749, 0.930815224, 1.40844091, 0.777395675, 1.31258338;1.29028919, 1.08211115, 1.60818483, 1.50650723, 1.75724549, 1.60481439, 0.954593536, 1.47053399, 1.23645387, 1.32546956, 1.58630895, 1.22544319, 1.34858122, 1.23532043, 1.16703637, 0.858721924, 1.14717063;1.23534566, 1.61448275, 1.60672911, 1.26797986, 1.87734493, 0.920236582, 1.6514219, 1.45882662, 1.54380496, 1.51978669, 1.54898502, 1.64806237, 1.99518941, 1.29445631, 1.27063442, 1.34958471, 1.6163044;1.16558819, 1.21402195, 1.27225625, 1.34820725, 1.60037419, 1.8091217, 1.34870095, 1.65906964, 1.56627971, 1.14334709, 1.44347139, 1.34561672, 1.16366532, 1.24192109, 1.41711435, 1.39334073, 1.46487864;0.851217204, 1.38654273, 1.25996361, 1.08938754, 1.36010603, 1.11460105, 1.78021386, 1.20474623, 1.80811711, 1.13829249, 1.12682919, 1.40494827, 1.36568828, 1.38270643, 1.42213059, 1.43770055, 1.40776875;1.09708261, 1.28940474, 1.33628245, 1.33830246, 1.48551781, 1.31057241, 1.3926511, 1.35252434, 1.02086511, 1.51494926, 1.28547619, 1.20728025, 1.57161899, 0.92849305, 1.53197496, 0.991302835, 1.55802837;0.833943408, 1.17616781, 1.23778033, 1.06863749, 1.47329265, 1.22232349, 1.20902733, 1.22231819, 1.04375596, 0.975792408, 1.39840075, 1.17984812, 1.30680439, 0.872561996, 1.07469708, 0.669135542, 1.13388541];
data_train = [0.00578225235, 0.137673426, 0.540058387, 0.745631208, 0.862694068, 0.770894469, 0.674216722, 0.260544239, 0.115671681, 0.863948784, 0.684229886;0.503495618, 0.0895350073, 0.661491936, 0.993042656, 0.595366674, 0.471684286, 0.168200015, 0.123245661, 0.352004049, 0.655159128, 0.539017359;0.537615948, 0.139573314, 0.030142644, 0.229629704, 0.951338943, 0.965778727, 0.0417576771, 0.561097126, 0.00351635234, 0.920606292, 0.613499723;0.46479492, 0.15507536, 0.931309644, 0.111796084, 0.913749448, 0.156725679, 0.910166384, 0.36401679, 0.766202914, 0.790361077, 0.296253419;0.8569265, 0.43918101, 0.921810562, 0.835617565, 0.36713935, 0.774310278, 0.636909704, 0.395835804, 0.259087048, 0.422246735, 0.684611368;0.285247748, 0.0378519406, 0.847832486, 0.182961292, 0.195659934, 0.752297421, 0.114232259, 0.824860804, 0.347431628, 0.32558655, 0.227375405;0.092141873, 0.049431421, 0.103286433, 0.615231393, 0.678243878, 0.895792842, 0.243856949, 0.766849343, 0.505617275, 0.30206984, 0.841881869;0.164978602, 0.401531596, 0.197806065, 0.737041684, 0.265389726, 0.52785462, 0.204153007, 0.903753853, 0.199291224, 0.797201755, 0.516419376;0.451112966, 0.444579969, 0.113663258, 0.932535955, 0.543618007, 0.519760752, 0.12009628, 0.230647891, 0.421303961, 0.368471454, 0.224367516;0.710282648, 0.879308733, 0.762532669, 0.398291152, 0.842536426, 0.032446128, 0.0574032121, 0.993493282, 0.89742388, 0.886265004, 0.319600328;0.0309812934, 0.180893173, 0.161647409, 0.771332184, 0.707526475, 0.775771186, 0.225405985, 0.874828126, 0.156887755, 0.690489936, 0.621484641];
feature_class = 'simple';
data_test = [0.171865675, 0.641425057, 0.0705731634, 0.0338511069, 0.149140138, 0.749160083, 0.733009377, 0.676684479, 0.80810868, 0.203685281, 0.417611292, 0.660320193, 0.0207840267, 0.726091062, 0.137990582, 0.768451447, 0.287321882;0.506470198, 0.224411277, 0.71209941, 0.478706341, 0.979042643, 0.635513708, 0.287329164, 0.912057381, 0.872927858, 0.12287943, 0.0457055772, 0.82718485, 0.803653553, 0.589057639, 0.35987935, 0.763835303, 0.0797286449;0.630619593, 0.622269411, 0.289086724, 0.302579646, 0.278116438, 0.26181934, 0.760390357, 0.283568529, 0.585709632, 0.332391861, 0.937654796, 0.373040469, 0.479326479, 0.478331426, 0.706230533, 0.66719421, 0.992809442;0.445758409, 0.340145557, 0.0752593871, 0.830930339, 0.025521495, 0.586512865, 0.0828444724, 0.997484515, 0.329633781, 0.1437432, 0.83360169, 0.570331135, 0.00827698484, 0.964858331, 0.643057785, 0.617959806, 0.350187554;0.476976203, 0.79600822, 0.143174135, 0.691311899, 0.30620251, 0.277302739, 0.23202724, 0.503510296, 0.789703558, 0.738298151, 0.417755805, 0.965059122, 0.263540692, 0.285875355, 0.576503249, 0.698881957, 0.548124938;0.243657003, 0.848543686, 0.260710864, 0.0880246031, 0.236656959, 0.1997833, 0.969305677, 0.52324032, 0.261976973, 0.203032653, 0.424377317, 0.869101752, 0.862690207, 0.319293343, 0.695699529, 0.403314426, 0.433896637;0.514993038, 0.511013975, 0.11525795, 0.723201796, 0.122879845, 0.0934949788, 0.280102489, 0.973331192, 0.0499350167, 0.927304832, 0.634517946, 0.560406309, 0.771486555, 0.810845371, 0.556876123, 0.708833969, 0.960938215;0.218490982, 0.229829925, 0.668761592, 0.832492986, 0.354641223, 0.290948764, 0.780142573, 0.346005617, 0.940981616, 0.517609617, 0.171841908, 0.437881741, 0.320575313, 0.770614875, 0.0162030953, 0.263160767, 0.234886917;0.379756753, 0.943758943, 0.409579079, 0.61871534, 0.948931198, 0.594255302, 0.522124849, 0.870649704, 0.48385057, 0.728263729, 0.813099748, 0.854907689, 0.74676302, 0.330997568, 0.0444069238, 0.574747985, 0.822779708;0.660425641, 0.735596653, 0.572911778, 0.389951721, 0.843012554, 0.253701128, 0.0084744958, 0.51245519, 0.259989221, 0.702797891, 0.900782942, 0.768304777, 0.742378934, 0.328593767, 0.0388367074, 0.365702799, 0.383474416;0.448459855, 0.632195388, 0.787496709, 0.424266848, 0.761133685, 0.0266095829, 0.566686549, 0.257150664, 0.886321017, 0.436360827, 0.0819208801, 0.64819166, 0.765489825, 0.839053641, 0.769674595, 0.835578385, 0.11795935];
data_class = 'rand';
feature_type = 'Real';
