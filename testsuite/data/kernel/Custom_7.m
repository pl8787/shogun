accuracy = 1e-06;
symdata = [1.71766245, 0.825980515, 1.33724218, 1.34512756, 1.12864059, 0.870924394, 1.04454755;0.825980515, 1.3130696, 0.657778897, 0.607665025, 0.509866026, 0.843347341, 0.343494147;1.33724218, 0.657778897, 1.31172797, 1.07183466, 1.61944208, 1.05511761, 1.35207892;1.34512756, 0.607665025, 1.07183466, 1.91967851, 1.03694891, 0.485134682, 1.14393134;1.12864059, 0.509866026, 1.61944208, 1.03694891, 0.131111087, 0.125205061, 1.00257689;0.870924394, 0.843347341, 1.05511761, 0.485134682, 0.125205061, 1.05963464, 0.734281784;1.04454755, 0.343494147, 1.35207892, 1.14393134, 1.00257689, 0.734281784, 0.118497055];
init_random = 42;
name = 'Custom';
dim_square = 7;
km_triangletriangle = [1.71766245, 0.825980544, 1.31306958, 1.33724213, 0.657778919, 1.311728, 1.34512758;0.825980544, 0.607665002, 1.07183468, 1.91967857, 1.12864053, 0.509865999, 1.61944211;1.31306958, 1.07183468, 1.03694892, 0.131111085, 0.870924413, 0.843347311, 1.05511761;1.33724213, 1.91967857, 0.131111085, 0.485134691, 0.125205055, 1.05963469, 1.04454756;0.657778919, 1.12864053, 0.870924413, 0.125205055, 0.343494147, 1.35207891, 1.14393139;1.311728, 0.509865999, 0.843347311, 1.05963469, 1.35207891, 1.00257695, 0.734281778;1.34512758, 1.61944211, 1.05511761, 1.04454756, 1.14393139, 0.734281778, 0.118497051];
data_class = 'rand';
feature_class = 'custom';
data_type = 'double';
km_fullfull = [0.858831227, 0.321910828, 0.499266535, 0.614877582, 0.750064015, 0.0955898389, 0.956922174;0.321910828, 0.656534791, 0.318371385, 0.458884329, 0.21749714, 0.407272995, 0.0535897911;0.499266535, 0.318371385, 0.655864, 0.917759538, 0.813819885, 0.230201855, 0.56863755;0.614877582, 0.458884329, 0.917759538, 0.959839284, 0.239808068, 0.18201609, 0.723758757;0.750064015, 0.21749714, 0.813819885, 0.239808068, 0.0655555427, 0.0385101736, 0.413279265;0.0955898389, 0.407272995, 0.230201855, 0.18201609, 0.0385101736, 0.529817343, 0.192955777;0.956922174, 0.0535897911, 0.56863755, 0.723758757, 0.413279265, 0.192955777, 0.0592485256];
km_fulltriangle = [1.71766245, 0.825980544, 1.33724213, 1.34512758, 1.12864053, 0.870924413, 1.04454756;0.825980544, 1.31306958, 0.657778919, 0.607665002, 0.509865999, 0.843347311, 0.343494147;1.33724213, 0.657778919, 1.311728, 1.07183468, 1.61944211, 1.05511761, 1.35207891;1.34512758, 0.607665002, 1.07183468, 1.91967857, 1.03694892, 0.485134691, 1.14393139;1.12864053, 0.509865999, 1.61944211, 1.03694892, 0.131111085, 0.125205055, 1.00257695;0.870924413, 0.843347311, 1.05511761, 0.485134691, 0.125205055, 1.05963469, 0.734281778;1.04454756, 0.343494147, 1.35207891, 1.14393139, 1.00257695, 0.734281778, 0.118497051];
data = [0.858831225, 0.321910837, 0.499266537, 0.614877557, 0.750063994, 0.0955898405, 0.956922167;0.504069678, 0.656534802, 0.318371382, 0.458884325, 0.217497146, 0.407272983, 0.0535897913;0.837975645, 0.339407515, 0.655863984, 0.917759551, 0.813819901, 0.230201856, 0.56863753;0.730250003, 0.1487807, 0.154075113, 0.959839257, 0.23980807, 0.182016085, 0.72375873;0.378576596, 0.29236888, 0.80562218, 0.797140843, 0.0655555434, 0.0385101737, 0.413279255;0.775334553, 0.436074358, 0.824915757, 0.303118597, 0.0866948871, 0.529817321, 0.19295578;0.0876253852, 0.289904356, 0.783441392, 0.420172615, 0.589297637, 0.541326004, 0.0592485275];
feature_type = '';
