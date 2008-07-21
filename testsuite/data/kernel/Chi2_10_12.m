accuracy = 1e-08;
init_random = 42;
name = 'Chi2';
data_type = 'double';
km_train = [1, 0.213062977, 0.198738422, 0.410895856, 0.174871121, 0.162043416, 0.407274983, 0.129652586, 0.379508352, 0.281238212, 0.477537054;0.213062977, 1, 0.0657371739, 0.2676331, 0.235590838, 0.177088651, 0.189246749, 0.211282421, 0.319781533, 0.315659114, 0.401105674;0.198738422, 0.0657371739, 1, 0.221052544, 0.0911614643, 0.0930668409, 0.269787395, 0.118923413, 0.124119595, 0.201018871, 0.201364659;0.410895856, 0.2676331, 0.221052544, 1, 0.373872854, 0.2937257, 0.285919153, 0.229993793, 0.376320666, 0.357659931, 0.560556198;0.174871121, 0.235590838, 0.0911614643, 0.373872854, 1, 0.107020042, 0.0730298844, 0.210565568, 0.309546721, 0.0903742121, 0.276107988;0.162043416, 0.177088651, 0.0930668409, 0.2937257, 0.107020042, 1, 0.127012751, 0.0608706944, 0.157804828, 0.160368665, 0.159531514;0.407274983, 0.189246749, 0.269787395, 0.285919153, 0.0730298844, 0.127012751, 1, 0.238104312, 0.209653773, 0.491849896, 0.361252484;0.129652586, 0.211282421, 0.118923413, 0.229993793, 0.210565568, 0.0608706944, 0.238104312, 1, 0.141914165, 0.224141641, 0.42495293;0.379508352, 0.319781533, 0.124119595, 0.376320666, 0.309546721, 0.157804828, 0.209653773, 0.141914165, 1, 0.269416231, 0.323637564;0.281238212, 0.315659114, 0.201018871, 0.357659931, 0.0903742121, 0.160368665, 0.491849896, 0.224141641, 0.269416231, 1, 0.527977883;0.477537054, 0.401105674, 0.201364659, 0.560556198, 0.276107988, 0.159531514, 0.361252484, 0.42495293, 0.323637564, 0.527977883, 1];
data_train = [0.649807286, 0.1639892, 0.987133009, 0.539696639, 0.557808138, 0.921124783, 0.360147054, 0.061591568, 0.478584349, 0.200313702, 0.268141651;0.260435648, 0.719236939, 0.94608732, 0.352625699, 0.00956585553, 0.992858812, 0.713925675, 0.469202972, 0.187033236, 0.820509276, 0.428645097;0.195168709, 0.936906849, 0.0587132344, 0.387396069, 0.766865038, 0.168198597, 0.218124876, 0.66341017, 0.673242002, 0.402456752, 0.461526833;0.426970978, 0.251190134, 0.25077858, 0.872379703, 0.463783719, 0.795620536, 0.433243149, 0.837802246, 0.1677085, 0.354721475, 0.642537168;0.518938144, 0.553354475, 0.331876467, 0.479591836, 0.389504568, 0.550604295, 0.0841794782, 0.143260764, 0.734582447, 0.552221819, 0.643920993;0.114936434, 0.545018825, 0.69834593, 0.970859772, 0.884431433, 0.788461368, 0.285592483, 0.484051839, 0.368156244, 0.777281893, 0.551282704;0.595323783, 0.11737322, 0.853555889, 0.678602672, 0.196732645, 0.425015142, 0.846939182, 0.307916559, 0.904223626, 0.944948437, 0.376614375;0.787848996, 0.789120168, 0.519268141, 0.9431184, 0.965815237, 0.748753589, 0.365448384, 0.180708281, 0.71663358, 0.228855415, 0.422597893;0.295276697, 0.411569653, 0.0961175136, 0.32008565, 0.660638037, 0.963074714, 0.0685383895, 0.339769592, 0.491674884, 0.24348578, 0.285322304;0.719397126, 0.724965849, 0.0109450612, 0.414559968, 0.0902110526, 0.740629756, 0.67327959, 0.0503255765, 0.330625472, 0.851442734, 0.457934082;0.785716175, 0.318294948, 0.924131565, 0.532730278, 0.522663291, 0.0268868531, 0.518142633, 0.62118506, 0.219714301, 0.616900294, 0.967322847];
feature_class = 'simple';
kernel_arg1_size = 10;
data_test = [0.17531088, 0.935396475, 0.013313455, 0.676474303, 0.479568895, 0.252440169, 0.244507802, 0.700054058, 0.945088, 0.312053482, 0.207639379, 0.0447296942, 0.326255873, 0.641124008, 0.717200229, 0.701411819, 0.335752438;0.581777565, 0.366137385, 0.687859861, 0.471234257, 0.0444145989, 0.516945859, 0.946376974, 0.311012238, 0.315782705, 0.710627053, 0.16391138, 0.661550193, 0.34892085, 0.365167064, 0.600421689, 0.219462296, 0.382066631;0.04741067, 0.352977371, 0.257485372, 0.752051882, 0.311691801, 0.076730295, 0.872630385, 0.793096044, 0.6336215, 0.955796247, 0.768301141, 0.796606113, 0.998724513, 0.306664013, 0.761698836, 0.0584160102, 0.592036005;0.134561488, 0.419614163, 0.781833833, 0.823057426, 0.517443789, 0.219583576, 0.23352486, 0.350917366, 0.641468916, 0.0346871731, 0.924894741, 0.60329286, 0.118715565, 0.540245933, 0.449273766, 0.681772887, 0.46183746;0.780953533, 0.575366733, 0.219016613, 0.252173656, 0.309208294, 0.0583499238, 0.359158759, 0.427318369, 0.885283267, 0.420005974, 0.391111662, 0.335096735, 0.0204881712, 0.886829871, 0.606959377, 0.200556124, 0.464317461;0.200736203, 0.852107351, 0.312437397, 0.985366241, 0.537830301, 0.681072431, 0.808189891, 0.978195213, 0.339099919, 0.397233094, 0.134444977, 0.63838246, 0.976342115, 0.0638991541, 0.299824273, 0.0192020438, 0.911778114;0.718411058, 0.650629129, 0.0619232157, 0.377676929, 0.659367927, 0.976804697, 0.998095652, 0.398793082, 0.342893286, 0.381099617, 0.620599361, 0.875909997, 0.936762219, 0.692665051, 0.92146512, 0.887857671, 0.982304067;0.528212729, 0.925206379, 0.136926047, 0.358306882, 0.151523562, 0.584888108, 0.488643775, 0.54429678, 0.213255151, 0.893319866, 0.496669379, 0.647092945, 0.0908647345, 0.234321629, 0.521137815, 0.102166918, 0.0693036669;0.71675202, 0.817263198, 0.50043892, 0.580687064, 0.348664339, 0.364326591, 0.196506214, 0.278711636, 0.722382723, 0.75737485, 0.365297197, 0.929381066, 0.106817437, 0.609719774, 0.292591349, 0.610310931, 0.684236679;0.71405868, 0.851959697, 0.174883241, 0.452900847, 0.720235346, 0.0673393502, 0.23675272, 0.703124907, 0.235181921, 0.418691235, 0.234693828, 0.5743297, 0.0582528956, 0.97121059, 0.959084545, 0.87874054, 0.636523913;0.587790616, 0.293021639, 0.0918957532, 0.763202695, 0.295777748, 0.305297719, 0.345612349, 0.416597092, 0.133910793, 0.811942983, 0.740258173, 0.0589915721, 0.520603112, 0.950341641, 0.184391191, 0.273320064, 0.941968126];
km_test = [0.441644105, 0.375868517, 0.085326204, 0.251032753, 0.367276651, 0.184262557, 0.157716973, 0.348916076, 0.218381306, 0.249555201, 0.394772303, 0.120736232, 0.0634623759, 0.58863042, 0.401743167, 0.336682496, 0.220001663;0.242841314, 0.273218516, 0.198231188, 0.286158504, 0.182387507, 0.129377953, 0.35779741, 0.472510501, 0.203132177, 0.514438517, 0.211545989, 0.345035689, 0.0941789151, 0.137710435, 0.34667243, 0.0585286787, 0.178392204;0.136584646, 0.129931132, 0.0512646696, 0.163702039, 0.108511314, 0.344646551, 0.245600979, 0.16041814, 0.0848883737, 0.112614456, 0.0922234987, 0.0461571948, 0.151784758, 0.0951073091, 0.120365205, 0.0677242577, 0.123374068;0.192257131, 0.547971265, 0.11092078, 0.563653518, 0.359758275, 0.271041451, 0.340248488, 0.577296782, 0.253484441, 0.230627406, 0.363867015, 0.267919838, 0.131360329, 0.210316387, 0.34966602, 0.136176801, 0.332298816;0.078911419, 0.271034486, 0.0908066552, 0.36238093, 0.221341404, 0.148907458, 0.163239488, 0.383821025, 0.246070708, 0.248392982, 0.272585174, 0.136048433, 0.117694993, 0.080063355, 0.118970092, 0.049458245, 0.150824044;0.159486639, 0.521591738, 0.121474773, 0.238954598, 0.162244274, 0.114596154, 0.120145879, 0.231638958, 0.270566894, 0.113861892, 0.0687817493, 0.271885175, 0.0213114947, 0.130849323, 0.275571876, 0.132152763, 0.115564774;0.286270517, 0.183883433, 0.161545395, 0.241692096, 0.367774319, 0.377986017, 0.348988853, 0.27527908, 0.118694568, 0.16734548, 0.256671436, 0.175985983, 0.202056773, 0.246702232, 0.366702213, 0.290307298, 0.241485141;0.104299513, 0.0804295714, 0.446248673, 0.372733519, 0.22902664, 0.245261378, 0.273018917, 0.206554855, 0.193740198, 0.172602875, 0.459064877, 0.205751396, 0.27644072, 0.107530461, 0.119253864, 0.0935453239, 0.243378246;0.326994038, 0.429557192, 0.0952476706, 0.227931803, 0.37081775, 0.263768945, 0.389193306, 0.442535066, 0.40469858, 0.366999563, 0.349122557, 0.311313714, 0.167241044, 0.242616781, 0.509744944, 0.157234812, 0.254940575;0.369126046, 0.271973672, 0.131874154, 0.330467684, 0.375837675, 0.244812251, 0.523171334, 0.406773556, 0.150644959, 0.235986553, 0.215981457, 0.287831484, 0.201789086, 0.307548495, 0.427400299, 0.158213303, 0.578442641;0.327095616, 0.286294552, 0.202167347, 0.560676428, 0.380658493, 0.200222818, 0.335659145, 0.498850442, 0.293046772, 0.349746, 0.536990349, 0.22160371, 0.152940197, 0.409198826, 0.320584648, 0.14170061, 0.475832767];
data_class = 'rand';
kernel_arg0_width = 1.2;
feature_type = 'Real';
