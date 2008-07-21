accuracy = 1e-10;
init_random = 42;
name = 'PolyMatchWord';
data_type = 'ushort';
km_train = [1, 0, 0, 0, 0, 0.000751314801, 0, 0, 0, 0, 0;0, 1, 0.000751314801, 0, 0.000751314801, 0, 0, 0, 0, 0, 0.000751314801;0, 0.000751314801, 1, 0, 0, 0, 0, 0, 0, 0, 0;0, 0, 0, 1, 0, 0.000751314801, 0, 0, 0.0202854996, 0, 0.000751314801;0, 0.000751314801, 0, 0, 1, 0, 0.000751314801, 0, 0, 0.000751314801, 0.000751314801;0.000751314801, 0, 0, 0.000751314801, 0, 1, 0, 0, 0.000751314801, 0, 0.000751314801;0, 0, 0, 0, 0.000751314801, 0, 1, 0, 0.000751314801, 0, 0.000751314801;0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0;0, 0, 0, 0.0202854996, 0, 0.000751314801, 0.000751314801, 0, 1, 0, 0.000751314801;0, 0, 0, 0, 0.000751314801, 0, 0, 0, 0, 1, 0;0, 0.000751314801, 0, 0.000751314801, 0.000751314801, 0.000751314801, 0.000751314801, 0, 0.000751314801, 0, 1];
data_train = {'38', '12', '16', '7', '14', '37', '1', '9', '7', '29', '0';'25', '0', '14', '20', '9', '25', '9', '31', '38', '3', '39';'12', '35', '9', '18', '29', '26', '32', '25', '2', '28', '35';'36', '14', '41', '17', '1', '20', '29', '6', '8', '3', '30';'26', '31', '30', '21', '32', '29', '40', '12', '16', '36', '9';'18', '27', '24', '2', '30', '20', '9', '10', '29', '3', '39';'35', '41', '12', '29', '41', '29', '28', '26', '29', '14', '29';'0', '4', '4', '29', '18', '9', '6', '7', '22', '12', '6';'40', '39', '33', '15', '41', '24', '11', '23', '21', '25', '4';'31', '10', '26', '22', '27', '33', '21', '3', '22', '9', '27';'33', '21', '6', '24', '0', '23', '41', '13', '41', '0', '38'};
feature_class = 'simple';
data_test = {'19', '12', '34', '24', '3', '35', '31', '24', '13', '16', '14', '2', '14', '17', '9', '7', '31';'0', '18', '2', '41', '7', '8', '4', '15', '15', '33', '13', '35', '4', '24', '7', '25', '24';'18', '12', '32', '5', '11', '23', '19', '2', '16', '9', '10', '1', '36', '8', '26', '18', '39';'10', '28', '0', '21', '38', '35', '21', '16', '20', '20', '21', '17', '28', '32', '35', '38', '37';'40', '36', '25', '40', '16', '5', '6', '15', '34', '13', '31', '37', '1', '22', '0', '10', '28';'33', '24', '36', '5', '11', '38', '23', '16', '26', '4', '17', '25', '22', '6', '24', '21', '24';'3', '1', '13', '18', '38', '20', '6', '38', '25', '15', '20', '2', '3', '2', '11', '15', '19';'24', '3', '15', '21', '18', '28', '4', '24', '36', '12', '29', '22', '4', '13', '9', '25', '41';'17', '24', '39', '8', '14', '28', '21', '37', '1', '17', '25', '7', '29', '19', '17', '20', '19';'5', '9', '11', '9', '17', '7', '9', '6', '41', '12', '3', '17', '35', '25', '23', '30', '21';'3', '23', '25', '6', '40', '11', '24', '5', '27', '28', '18', '28', '5', '6', '31', '31', '2'};
kernel_arg0_degree = 3;
km_test = [0, 0.000751314801, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000751314801, 0;0.000751314801, 0.000751314801, 0.000751314801, 0, 0, 0, 0.000751314801, 0, 0, 0, 0.000751314801, 0, 0.000751314801, 0, 0, 0, 0;0, 0.000751314801, 0, 0.000751314801, 0, 0, 0.000751314801, 0, 0, 0.00601051841, 0, 0, 0.000751314801, 0.000751314801, 0.000751314801, 0, 0.000751314801;0.000751314801, 0, 0, 0, 0, 0, 0.000751314801, 0, 0, 0, 0.000751314801, 0.000751314801, 0, 0, 0, 0.00601051841, 0;0, 0, 0, 0, 0.000751314801, 0, 0, 0, 0, 0, 0.000751314801, 0, 0.000751314801, 0, 0, 0, 0;0, 0.00601051841, 0, 0, 0, 0, 0, 0, 0.000751314801, 0.000751314801, 0, 0, 0, 0, 0.00601051841, 0.000751314801, 0;0.000751314801, 0, 0.000751314801, 0.000751314801, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000751314801;0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.000751314801, 0, 0, 0, 0.000751314801, 0, 0;0, 0, 0, 0, 0.000751314801, 0, 0.000751314801, 0.000751314801, 0, 0, 0, 0.000751314801, 0, 0, 0, 0.000751314801, 0;0, 0.00601051841, 0, 0.000751314801, 0, 0, 0.000751314801, 0, 0, 0.000751314801, 0.000751314801, 0, 0, 0, 0, 0, 0;0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
kernel_arg1_inhomogene = 'False';
data_class = 'rand';
feature_type = 'Word';
