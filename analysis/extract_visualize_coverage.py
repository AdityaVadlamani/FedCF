import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# ============================================================================
# CONFIGURATION
# ============================================================================
# Choose visualization type: "label_only", "group_only", or "label_group"
option = "group_only"

csv_data = """
"""  # Set to empty string to read from file

# Option 1: Provide CSV data as a multiline string (paste rows with header)
# csv_data = """
# ,seed,c,k_reg,lambda,use_mle,sketch_method,base_coverage,base_efficiency,coverage,efficiency,violation,base_violation,class_cond_cov_y=0,base_class_cond_cov_y=0,class_cond_cov_y=1,base_class_cond_cov_y=1,class_cond_cov_y=2,base_class_cond_cov_y=2,class_cond_cov_y=4,base_class_cond_cov_y=4,class_cond_cov_y=4,base_class_cond_cov_y=4,class_cond_cov_y=5,base_class_cond_cov_y=5,class_cond_cov_y=6,base_class_cond_cov_y=6,class_cond_cov_y=7,base_class_cond_cov_y=7,class_cond_cov_y=8,base_class_cond_cov_y=8,pos_cov_y=1_g=0,base_pos_cov_y=1_g=0,pos_cov_y=1_g=1,base_pos_cov_y=1_g=1,pos_cov_y=1_g=2,base_pos_cov_y=1_g=2,pos_cov_y=1_g=3,base_pos_cov_y=1_g=3,pos_cov_y=1_g=4,base_pos_cov_y=1_g=4,pos_cov_y=1_g=5,base_pos_cov_y=1_g=5,violation_label=1,base_violation_label=1,pos_cov_y=2_g=0,base_pos_cov_y=2_g=0,pos_cov_y=2_g=1,base_pos_cov_y=2_g=1,pos_cov_y=2_g=2,base_pos_cov_y=2_g=2,pos_cov_y=2_g=3,base_pos_cov_y=2_g=3,pos_cov_y=2_g=4,base_pos_cov_y=2_g=4,pos_cov_y=2_g=5,base_pos_cov_y=2_g=5,violation_label=2,base_violation_label=2,pos_cov_y=3_g=0,base_pos_cov_y=3_g=0,pos_cov_y=3_g=1,base_pos_cov_y=3_g=1,pos_cov_y=3_g=2,base_pos_cov_y=3_g=2,pos_cov_y=3_g=3,base_pos_cov_y=3_g=3,pos_cov_y=3_g=4,base_pos_cov_y=3_g=4,pos_cov_y=3_g=5,base_pos_cov_y=3_g=5,violation_label=3,base_violation_label=3,pos_cov_y=4_g=0,base_pos_cov_y=4_g=0,pos_cov_y=4_g=1,base_pos_cov_y=4_g=1,pos_cov_y=4_g=2,base_pos_cov_y=4_g=2,pos_cov_y=4_g=3,base_pos_cov_y=4_g=3,pos_cov_y=4_g=4,base_pos_cov_y=4_g=4,pos_cov_y=4_g=5,base_pos_cov_y=4_g=5,violation_label=4,base_violation_label=4,pos_cov_y=5_g=0,base_pos_cov_y=5_g=0,pos_cov_y=5_g=1,base_pos_cov_y=5_g=1,pos_cov_y=5_g=2,base_pos_cov_y=5_g=2,pos_cov_y=5_g=3,base_pos_cov_y=5_g=3,pos_cov_y=5_g=4,base_pos_cov_y=5_g=4,pos_cov_y=5_g=5,base_pos_cov_y=5_g=5,violation_label=5,base_violation_label=5,pos_cov_y=6_g=0,base_pos_cov_y=6_g=0,pos_cov_y=6_g=1,base_pos_cov_y=6_g=1,pos_cov_y=6_g=2,base_pos_cov_y=6_g=2,pos_cov_y=6_g=3,base_pos_cov_y=6_g=3,pos_cov_y=6_g=4,base_pos_cov_y=6_g=4,pos_cov_y=6_g=5,base_pos_cov_y=6_g=5,violation_label=6,base_violation_label=6,pos_cov_y=7_g=0,base_pos_cov_y=7_g=0,pos_cov_y=7_g=1,base_pos_cov_y=7_g=1,pos_cov_y=7_g=2,base_pos_cov_y=7_g=2,pos_cov_y=7_g=3,base_pos_cov_y=7_g=3,pos_cov_y=7_g=4,base_pos_cov_y=7_g=4,pos_cov_y=7_g=5,base_pos_cov_y=7_g=5,violation_label=7,base_violation_label=7,pos_cov_y=8_g=0,base_pos_cov_y=8_g=0,pos_cov_y=8_g=1,base_pos_cov_y=8_g=1,pos_cov_y=8_g=2,base_pos_cov_y=8_g=2,pos_cov_y=8_g=3,base_pos_cov_y=8_g=3,pos_cov_y=8_g=4,base_pos_cov_y=8_g=4,pos_cov_y=8_g=5,base_pos_cov_y=8_g=5,violation_label=8,base_violation_label=8,qhat,lambda_opt_0,lambda_opt_1,lambda_opt_2,lambda_opt_3,lambda_opt_4,lambda_opt_5,lambda_opt_6,lambda_opt_7,lambda_opt_8,client_formulation_0,client_formulation_1,client_formulation_2,client_formulation_3,client_formulation_4,client_formulation_5,client_formulation_6,client_formulation_7,use_descent,use_group_info
# 0,42,0.1,3,0.1,False,tdigest,tensor(0.8954),tensor(3.5027),tensor(0.9268),tensor(5.2560),tensor(0.0892),tensor(0.1658),tensor(0.9966),tensor(0.9966),tensor(0.4134),tensor(0.4134),tensor(0.9956),tensor(0.8584),tensor(0.0345),tensor(0.0345),tensor(1.),tensor(0.8483),tensor(0.0952),tensor(0.0952),tensor(1.),tensor(0.0541),tensor(1.),tensor(0.9502),tensor(0.7059),tensor(0.7059),tensor(0.2171),tensor(0.2171),tensor(0.1744),tensor(0.1744),tensor(0.1959),tensor(0.1959),tensor(0.1755),tensor(0.1755),tensor(0.1279),tensor(0.1279),tensor(0.1456),tensor(0.1456),tensor(0.0892),tensor(0.0892),tensor(0.9878),tensor(0.7137),tensor(0.9875),tensor(0.6661),tensor(0.9940),tensor(0.6917),tensor(0.9914),tensor(0.7122),tensor(0.9817),tensor(0.7023),tensor(0.9873),tensor(0.7405),tensor(0.0122),tensor(0.0744),tensor(0.0109),tensor(0.0109),tensor(0.0116),tensor(0.0116),tensor(0.0157),tensor(0.0157),tensor(0.0072),tensor(0.0072),tensor(0.0052),tensor(0.0052),tensor(0.0190),tensor(0.0190),tensor(0.0138),tensor(0.0138),tensor(0.9376),tensor(0.6174),tensor(0.9502),tensor(0.6229),tensor(0.9528),tensor(0.6348),tensor(0.9770),tensor(0.7324),tensor(0.9739),tensor(0.7311),tensor(0.9873),tensor(0.6709),tensor(0.0498),tensor(0.1150),tensor(0.0041),tensor(0.0041),tensor(0.0141),tensor(0.0141),tensor(0.0073),tensor(0.0073),tensor(0.0101),tensor(0.0101),tensor(0.0131),tensor(0.0131),tensor(0.0063),tensor(0.0063),tensor(0.0100),tensor(0.0100),tensor(1.),tensor(0.0027),tensor(1.),tensor(0.0017),tensor(1.),tensor(0.0073),tensor(1.),tensor(0.0043),tensor(1.),tensor(0.0209),tensor(1.),tensor(0.0253),tensor(0.),tensor(0.0237),tensor(0.9959),tensor(0.8657),tensor(0.9983),tensor(0.8937),tensor(0.9976),tensor(0.8222),tensor(0.9942),tensor(0.7698),tensor(0.9974),tensor(0.7598),tensor(0.9937),tensor(0.7278),tensor(0.0047),tensor(0.1658),tensor(0.1547),tensor(0.1547),tensor(0.1337),tensor(0.1337),tensor(0.1306),tensor(0.1306),tensor(0.1165),tensor(0.1165),tensor(0.1358),tensor(0.1358),tensor(0.1709),tensor(0.1709),tensor(0.0543),tensor(0.0543),tensor(0.9650),0.9649927020072937,0.9649927020072937,1.2695510387420654,0.9649927020072937,1.297845721244812,0.9649927020072937,2.0,1.3767520189285278,0.9649927020072937,0,0,0,0,0,0,0,0,True,False
# 0,42,0.15,3,0.1,False,tdigest,tensor(0.8954),tensor(3.5050),tensor(0.9088),tensor(3.8629),tensor(0.0892),tensor(0.1595),tensor(0.9966),tensor(0.9966),tensor(0.4134),tensor(0.4134),tensor(0.8584),tensor(0.8584),tensor(0.0345),tensor(0.0345),tensor(1.),tensor(0.8483),tensor(0.0952),tensor(0.0952),tensor(0.0541),tensor(0.0541),tensor(0.9813),tensor(0.9502),tensor(0.7059),tensor(0.7059),tensor(0.2171),tensor(0.2171),tensor(0.1744),tensor(0.1744),tensor(0.1959),tensor(0.1959),tensor(0.1755),tensor(0.1755),tensor(0.1279),tensor(0.1279),tensor(0.1456),tensor(0.1456),tensor(0.0892),tensor(0.0892),tensor(0.7151),tensor(0.7151),tensor(0.6661),tensor(0.6661),tensor(0.6941),tensor(0.6941),tensor(0.7122),tensor(0.7122),tensor(0.7023),tensor(0.7023),tensor(0.7405),tensor(0.7405),tensor(0.0744),tensor(0.0744),tensor(0.0109),tensor(0.0109),tensor(0.0116),tensor(0.0116),tensor(0.0157),tensor(0.0157),tensor(0.0072),tensor(0.0072),tensor(0.0052),tensor(0.0052),tensor(0.0190),tensor(0.0190),tensor(0.0138),tensor(0.0138),tensor(0.9023),tensor(0.6174),tensor(0.9136),tensor(0.6229),tensor(0.9057),tensor(0.6348),tensor(0.9554),tensor(0.7324),tensor(0.9530),tensor(0.7311),tensor(0.9684),tensor(0.6709),tensor(0.0660),tensor(0.1150),tensor(0.0041),tensor(0.0041),tensor(0.0141),tensor(0.0141),tensor(0.0073),tensor(0.0073),tensor(0.0101),tensor(0.0101),tensor(0.0131),tensor(0.0131),tensor(0.0063),tensor(0.0063),tensor(0.0100),tensor(0.0100),tensor(0.0027),tensor(0.0027),tensor(0.0017),tensor(0.0017),tensor(0.0073),tensor(0.0073),tensor(0.0043),tensor(0.0043),tensor(0.0209),tensor(0.0209),tensor(0.0253),tensor(0.0253),tensor(0.0237),tensor(0.0237),tensor(0.9322),tensor(0.8657),tensor(0.9551),tensor(0.8937),tensor(0.9214),tensor(0.8222),tensor(0.8849),tensor(0.7698),tensor(0.9008),tensor(0.7598),tensor(0.8797),tensor(0.7342),tensor(0.0754),tensor(0.1595),tensor(0.1574),tensor(0.1574),tensor(0.1337),tensor(0.1337),tensor(0.1306),tensor(0.1306),tensor(0.1194),tensor(0.1194),tensor(0.1384),tensor(0.1384),tensor(0.1709),tensor(0.1709),tensor(0.0515),tensor(0.0515),tensor(0.9651),0.9650774598121643,0.9650774598121643,0.9650774598121643,0.9650774598121643,1.199760913848877,0.9650774598121643,0.9650774598121643,1.0479087829589844,0.9650774598121643,0,0,0,0,0,0,0,0,True,False
# 0,42,0.2,3,0.1,False,tdigest,tensor(0.8954),tensor(3.5050),tensor(0.8961),tensor(3.5307),tensor(0.1463),tensor(0.1595),tensor(0.9966),tensor(0.9966),tensor(0.4134),tensor(0.4134),tensor(0.8584),tensor(0.8584),tensor(0.0345),tensor(0.0345),tensor(0.8586),tensor(0.8483),tensor(0.0952),tensor(0.0952),tensor(0.0541),tensor(0.0541),tensor(0.9502),tensor(0.9502),tensor(0.7059),tensor(0.7059),tensor(0.2171),tensor(0.2171),tensor(0.1744),tensor(0.1744),tensor(0.1959),tensor(0.1959),tensor(0.1755),tensor(0.1755),tensor(0.1279),tensor(0.1279),tensor(0.1456),tensor(0.1456),tensor(0.0892),tensor(0.0892),tensor(0.7151),tensor(0.7151),tensor(0.6661),tensor(0.6661),tensor(0.6941),tensor(0.6941),tensor(0.7122),tensor(0.7122),tensor(0.7023),tensor(0.7023),tensor(0.7405),tensor(0.7405),tensor(0.0744),tensor(0.0744),tensor(0.0109),tensor(0.0109),tensor(0.0116),tensor(0.0116),tensor(0.0157),tensor(0.0157),tensor(0.0072),tensor(0.0072),tensor(0.0052),tensor(0.0052),tensor(0.0190),tensor(0.0190),tensor(0.0138),tensor(0.0138),tensor(0.6296),tensor(0.6174),tensor(0.6412),tensor(0.6229),tensor(0.6493),tensor(0.6348),tensor(0.7396),tensor(0.7324),tensor(0.7493),tensor(0.7311),tensor(0.6899),tensor(0.6709),tensor(0.1198),tensor(0.1150),tensor(0.0041),tensor(0.0041),tensor(0.0141),tensor(0.0141),tensor(0.0073),tensor(0.0073),tensor(0.0101),tensor(0.0101),tensor(0.0131),tensor(0.0131),tensor(0.0063),tensor(0.0063),tensor(0.0100),tensor(0.0100),tensor(0.0027),tensor(0.0027),tensor(0.0017),tensor(0.0017),tensor(0.0073),tensor(0.0073),tensor(0.0043),tensor(0.0043),tensor(0.0209),tensor(0.0209),tensor(0.0253),tensor(0.0253),tensor(0.0237),tensor(0.0237),tensor(0.8765),tensor(0.8657),tensor(0.8995),tensor(0.8937),tensor(0.8368),tensor(0.8222),tensor(0.7770),tensor(0.7698),tensor(0.7859),tensor(0.7598),tensor(0.7532),tensor(0.7342),tensor(0.1463),tensor(0.1595),tensor(0.1574),tensor(0.1574),tensor(0.1337),tensor(0.1337),tensor(0.1306),tensor(0.1306),tensor(0.1194),tensor(0.1194),tensor(0.1384),tensor(0.1384),tensor(0.1709),tensor(0.1709),tensor(0.0515),tensor(0.0515),tensor(0.9651),0.9651233553886414,0.9651233553886414,0.9651233553886414,0.9651233553886414,0.9682230949401855,0.9651233553886414,0.9651233553886414,0.9699499607086182,0.9651233553886414,0,0,0,0,0,0,0,0,True,False
# """

csv_data = """
,seed,c,k_reg,lambda,use_mle,sketch_method,base_coverage,base_efficiency,coverage,efficiency,violation,base_violation,class_cond_cov_y=0,base_class_cond_cov_y=0,class_cond_cov_y=1,base_class_cond_cov_y=1,class_cond_cov_y=2,base_class_cond_cov_y=2,class_cond_cov_y=4,base_class_cond_cov_y=4,class_cond_cov_y=4,base_class_cond_cov_y=4,class_cond_cov_y=5,base_class_cond_cov_y=5,class_cond_cov_y=6,base_class_cond_cov_y=6,class_cond_cov_y=7,base_class_cond_cov_y=7,class_cond_cov_y=8,base_class_cond_cov_y=8,group_cond_cov_g=0,base_group_cond_cov_g=0,group_cond_cov_g=1,base_group_cond_cov_g=1,group_cond_cov_g=2,base_group_cond_cov_g=2,group_cond_cov_g=3,base_group_cond_cov_g=3,group_cond_cov_g=4,base_group_cond_cov_g=4,group_cond_cov_g=5,base_group_cond_cov_g=5,pos_cov_y=1_g=0,base_pos_cov_y=1_g=0,pos_cov_y=1_g=1,base_pos_cov_y=1_g=1,pos_cov_y=1_g=2,base_pos_cov_y=1_g=2,pos_cov_y=1_g=3,base_pos_cov_y=1_g=3,pos_cov_y=1_g=4,base_pos_cov_y=1_g=4,pos_cov_y=1_g=5,base_pos_cov_y=1_g=5,violation_label=1,base_violation_label=1,pos_cov_y=2_g=0,base_pos_cov_y=2_g=0,pos_cov_y=2_g=1,base_pos_cov_y=2_g=1,pos_cov_y=2_g=2,base_pos_cov_y=2_g=2,pos_cov_y=2_g=3,base_pos_cov_y=2_g=3,pos_cov_y=2_g=4,base_pos_cov_y=2_g=4,pos_cov_y=2_g=5,base_pos_cov_y=2_g=5,violation_label=2,base_violation_label=2,pos_cov_y=3_g=0,base_pos_cov_y=3_g=0,pos_cov_y=3_g=1,base_pos_cov_y=3_g=1,pos_cov_y=3_g=2,base_pos_cov_y=3_g=2,pos_cov_y=3_g=3,base_pos_cov_y=3_g=3,pos_cov_y=3_g=4,base_pos_cov_y=3_g=4,pos_cov_y=3_g=5,base_pos_cov_y=3_g=5,violation_label=3,base_violation_label=3,pos_cov_y=4_g=0,base_pos_cov_y=4_g=0,pos_cov_y=4_g=1,base_pos_cov_y=4_g=1,pos_cov_y=4_g=2,base_pos_cov_y=4_g=2,pos_cov_y=4_g=3,base_pos_cov_y=4_g=3,pos_cov_y=4_g=4,base_pos_cov_y=4_g=4,pos_cov_y=4_g=5,base_pos_cov_y=4_g=5,violation_label=4,base_violation_label=4,pos_cov_y=5_g=0,base_pos_cov_y=5_g=0,pos_cov_y=5_g=1,base_pos_cov_y=5_g=1,pos_cov_y=5_g=2,base_pos_cov_y=5_g=2,pos_cov_y=5_g=3,base_pos_cov_y=5_g=3,pos_cov_y=5_g=4,base_pos_cov_y=5_g=4,pos_cov_y=5_g=5,base_pos_cov_y=5_g=5,violation_label=5,base_violation_label=5,pos_cov_y=6_g=0,base_pos_cov_y=6_g=0,pos_cov_y=6_g=1,base_pos_cov_y=6_g=1,pos_cov_y=6_g=2,base_pos_cov_y=6_g=2,pos_cov_y=6_g=3,base_pos_cov_y=6_g=3,pos_cov_y=6_g=4,base_pos_cov_y=6_g=4,pos_cov_y=6_g=5,base_pos_cov_y=6_g=5,violation_label=6,base_violation_label=6,pos_cov_y=7_g=0,base_pos_cov_y=7_g=0,pos_cov_y=7_g=1,base_pos_cov_y=7_g=1,pos_cov_y=7_g=2,base_pos_cov_y=7_g=2,pos_cov_y=7_g=3,base_pos_cov_y=7_g=3,pos_cov_y=7_g=4,base_pos_cov_y=7_g=4,pos_cov_y=7_g=5,base_pos_cov_y=7_g=5,violation_label=7,base_violation_label=7,pos_cov_y=8_g=0,base_pos_cov_y=8_g=0,pos_cov_y=8_g=1,base_pos_cov_y=8_g=1,pos_cov_y=8_g=2,base_pos_cov_y=8_g=2,pos_cov_y=8_g=3,base_pos_cov_y=8_g=3,pos_cov_y=8_g=4,base_pos_cov_y=8_g=4,pos_cov_y=8_g=5,base_pos_cov_y=8_g=5,violation_label=8,base_violation_label=8,qhat,lambda_opt_0,lambda_opt_1,lambda_opt_2,lambda_opt_3,lambda_opt_4,lambda_opt_5,lambda_opt_6,lambda_opt_7,lambda_opt_8,client_formulation_0,client_formulation_1,client_formulation_2,client_formulation_3,client_formulation_4,client_formulation_5,client_formulation_6,client_formulation_7,use_descent,use_group_info
0,42,0.2,3,0.1,False,tdigest,tensor(0.8954),tensor(3.5012),tensor(0.8961),tensor(3.5285),tensor(0.1463),tensor(0.1658),tensor(0.9966),tensor(0.9966),tensor(0.4134),tensor(0.4134),tensor(0.8584),tensor(0.8584),tensor(0.0345),tensor(0.0345),tensor(0.8586),tensor(0.8483),tensor(0.0952),tensor(0.0952),tensor(0.0541),tensor(0.0541),tensor(0.9502),tensor(0.9502),tensor(0.7059),tensor(0.7059),tensor(0.8901),tensor(0.8887),tensor(0.8779),tensor(0.8771),tensor(0.8972),tensor(0.8972),tensor(0.9252),tensor(0.9237),tensor(0.9034),tensor(0.9034),tensor(0.9114),tensor(0.9114),tensor(0.2171),tensor(0.2171),tensor(0.1736),tensor(0.1736),tensor(0.1959),tensor(0.1959),tensor(0.1755),tensor(0.1755),tensor(0.1279),tensor(0.1279),tensor(0.1456),tensor(0.1456),tensor(0.0892),tensor(0.0892),tensor(0.7137),tensor(0.7137),tensor(0.6661),tensor(0.6661),tensor(0.6917),tensor(0.6917),tensor(0.7122),tensor(0.7122),tensor(0.7023),tensor(0.7023),tensor(0.7405),tensor(0.7405),tensor(0.0744),tensor(0.0744),tensor(0.0109),tensor(0.0109),tensor(0.0116),tensor(0.0116),tensor(0.0157),tensor(0.0157),tensor(0.0072),tensor(0.0072),tensor(0.0052),tensor(0.0052),tensor(0.0190),tensor(0.0190),tensor(0.0138),tensor(0.0138),tensor(0.6296),tensor(0.6160),tensor(0.6412),tensor(0.6221),tensor(0.6493),tensor(0.6336),tensor(0.7396),tensor(0.7295),tensor(0.7493),tensor(0.7311),tensor(0.6899),tensor(0.6709),tensor(0.1198),tensor(0.1151),tensor(0.0041),tensor(0.0041),tensor(0.0141),tensor(0.0141),tensor(0.0073),tensor(0.0073),tensor(0.0101),tensor(0.0101),tensor(0.0131),tensor(0.0131),tensor(0.0063),tensor(0.0063),tensor(0.0100),tensor(0.0100),tensor(0.0027),tensor(0.0027),tensor(0.0017),tensor(0.0017),tensor(0.0073),tensor(0.0073),tensor(0.0043),tensor(0.0043),tensor(0.0209),tensor(0.0209),tensor(0.0253),tensor(0.0253),tensor(0.0237),tensor(0.0237),tensor(0.8765),tensor(0.8657),tensor(0.8995),tensor(0.8937),tensor(0.8368),tensor(0.8222),tensor(0.7770),tensor(0.7698),tensor(0.7859),tensor(0.7598),tensor(0.7532),tensor(0.7278),tensor(0.1463),tensor(0.1658),tensor(0.1547),tensor(0.1547),tensor(0.1337),tensor(0.1337),tensor(0.1306),tensor(0.1306),tensor(0.1165),tensor(0.1165),tensor(0.1358),tensor(0.1358),tensor(0.1709),tensor(0.1709),tensor(0.0543),tensor(0.0543),tensor(0.9649),0.9649151563644409,0.9649151563644409,0.9649151563644409,0.9649151563644409,0.9682230949401855,0.9649151563644409,0.9649151563644409,0.9699494242668152,0.9649151563644409,0,0,0,0,0,0,0,0,True,False
0,42,0.15,3,0.1,False,tdigest,tensor(0.8951),tensor(3.5010),tensor(0.9086),tensor(3.8601),tensor(0.0892),tensor(0.1658),tensor(0.9966),tensor(0.9966),tensor(0.4134),tensor(0.4134),tensor(0.8584),tensor(0.8584),tensor(0.0345),tensor(0.0345),tensor(1.),tensor(0.8448),tensor(0.0952),tensor(0.0952),tensor(0.0541),tensor(0.0541),tensor(0.9782),tensor(0.9502),tensor(0.7059),tensor(0.7059),tensor(0.8996),tensor(0.8887),tensor(0.8879),tensor(0.8762),tensor(0.9154),tensor(0.8972),tensor(0.9338),tensor(0.9237),tensor(0.9217),tensor(0.9034),tensor(0.9304),tensor(0.9114),tensor(0.2171),tensor(0.2171),tensor(0.1736),tensor(0.1736),tensor(0.1959),tensor(0.1959),tensor(0.1755),tensor(0.1755),tensor(0.1279),tensor(0.1279),tensor(0.1456),tensor(0.1456),tensor(0.0892),tensor(0.0892),tensor(0.7137),tensor(0.7137),tensor(0.6661),tensor(0.6661),tensor(0.6917),tensor(0.6917),tensor(0.7122),tensor(0.7122),tensor(0.7023),tensor(0.7023),tensor(0.7405),tensor(0.7405),tensor(0.0744),tensor(0.0744),tensor(0.0109),tensor(0.0109),tensor(0.0116),tensor(0.0116),tensor(0.0157),tensor(0.0157),tensor(0.0072),tensor(0.0072),tensor(0.0052),tensor(0.0052),tensor(0.0190),tensor(0.0190),tensor(0.0138),tensor(0.0138),tensor(0.9023),tensor(0.6160),tensor(0.9128),tensor(0.6213),tensor(0.9057),tensor(0.6336),tensor(0.9554),tensor(0.7295),tensor(0.9530),tensor(0.7311),tensor(0.9684),tensor(0.6709),tensor(0.0660),tensor(0.1151),tensor(0.0041),tensor(0.0041),tensor(0.0141),tensor(0.0141),tensor(0.0073),tensor(0.0073),tensor(0.0101),tensor(0.0101),tensor(0.0131),tensor(0.0131),tensor(0.0063),tensor(0.0063),tensor(0.0100),tensor(0.0100),tensor(0.0027),tensor(0.0027),tensor(0.0017),tensor(0.0017),tensor(0.0073),tensor(0.0073),tensor(0.0043),tensor(0.0043),tensor(0.0209),tensor(0.0209),tensor(0.0253),tensor(0.0253),tensor(0.0237),tensor(0.0237),tensor(0.9322),tensor(0.8657),tensor(0.9551),tensor(0.8937),tensor(0.9214),tensor(0.8222),tensor(0.8835),tensor(0.7698),tensor(0.9008),tensor(0.7598),tensor(0.8797),tensor(0.7278),tensor(0.0754),tensor(0.1658),tensor(0.1547),tensor(0.1547),tensor(0.1337),tensor(0.1337),tensor(0.1306),tensor(0.1306),tensor(0.1165),tensor(0.1165),tensor(0.1358),tensor(0.1358),tensor(0.1709),tensor(0.1709),tensor(0.0543),tensor(0.0543),tensor(0.9649),0.9648696184158325,0.9648696184158325,0.9648696184158325,0.9648696184158325,1.199745774269104,0.9648696184158325,0.9648696184158325,1.0478335618972778,0.9648696184158325,0,0,0,0,0,0,0,0,True,False
0,42,0.1,3,0.1,False,tdigest,tensor(0.8954),tensor(3.5052),tensor(0.9268),tensor(5.2575),tensor(0.0892),tensor(0.1595),tensor(0.9966),tensor(0.9966),tensor(0.4134),tensor(0.4134),tensor(0.9956),tensor(0.8584),tensor(0.0345),tensor(0.0345),tensor(1.),tensor(0.8483),tensor(0.0952),tensor(0.0952),tensor(1.),tensor(0.0541),tensor(1.),tensor(0.9502),tensor(0.7059),tensor(0.7059),tensor(0.9104),tensor(0.8887),tensor(0.9037),tensor(0.8771),tensor(0.9359),tensor(0.8972),tensor(0.9554),tensor(0.9237),tensor(0.9452),tensor(0.9034),tensor(0.9620),tensor(0.9114),tensor(0.2171),tensor(0.2171),tensor(0.1744),tensor(0.1744),tensor(0.1959),tensor(0.1959),tensor(0.1755),tensor(0.1755),tensor(0.1279),tensor(0.1279),tensor(0.1456),tensor(0.1456),tensor(0.0892),tensor(0.0892),tensor(0.9878),tensor(0.7151),tensor(0.9875),tensor(0.6661),tensor(0.9940),tensor(0.6941),tensor(0.9914),tensor(0.7122),tensor(0.9817),tensor(0.7023),tensor(0.9873),tensor(0.7405),tensor(0.0122),tensor(0.0744),tensor(0.0109),tensor(0.0109),tensor(0.0116),tensor(0.0116),tensor(0.0157),tensor(0.0157),tensor(0.0072),tensor(0.0072),tensor(0.0052),tensor(0.0052),tensor(0.0190),tensor(0.0190),tensor(0.0138),tensor(0.0138),tensor(0.9376),tensor(0.6174),tensor(0.9502),tensor(0.6229),tensor(0.9528),tensor(0.6348),tensor(0.9770),tensor(0.7324),tensor(0.9739),tensor(0.7311),tensor(0.9873),tensor(0.6709),tensor(0.0498),tensor(0.1150),tensor(0.0041),tensor(0.0041),tensor(0.0141),tensor(0.0141),tensor(0.0073),tensor(0.0073),tensor(0.0101),tensor(0.0101),tensor(0.0131),tensor(0.0131),tensor(0.0063),tensor(0.0063),tensor(0.0100),tensor(0.0100),tensor(1.),tensor(0.0027),tensor(1.),tensor(0.0017),tensor(1.),tensor(0.0073),tensor(1.),tensor(0.0043),tensor(1.),tensor(0.0209),tensor(1.),tensor(0.0253),tensor(0.),tensor(0.0237),tensor(0.9959),tensor(0.8657),tensor(0.9983),tensor(0.8937),tensor(0.9976),tensor(0.8222),tensor(0.9942),tensor(0.7698),tensor(0.9974),tensor(0.7598),tensor(0.9937),tensor(0.7342),tensor(0.0047),tensor(0.1595),tensor(0.1574),tensor(0.1574),tensor(0.1337),tensor(0.1337),tensor(0.1318),tensor(0.1318),tensor(0.1194),tensor(0.1194),tensor(0.1384),tensor(0.1384),tensor(0.1709),tensor(0.1709),tensor(0.0515),tensor(0.0515),tensor(0.9651),0.9651298522949219,0.9651298522949219,1.2687468528747559,0.9651298522949219,1.2978589534759521,0.9651298522949219,2.0,1.37711763381958,0.9651298522949219,0,0,0,0,0,0,0,0,True,False
"""
csv_data = csv_data.strip()  # Set to empty string to read from file path below

# Option 2: Read from CSV file
csv_file_path = "./fairness_trials/ACSIncome/continental_small/split_0.3_0.2/Demographic_Parity/raps_rand_no_mod.csv"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def parse_tensor_value(x):
    """Convert tensor string (e.g., 'tensor(0.5)') to float."""
    return float(str(x).replace("tensor(", "").replace(")", ""))


def parse_column_name(col):
    """Extract (y, g) from column name like 'pos_cov_y=1_g=0' or 'base_pos_cov_y=1_g=0'."""
    col = col.replace("base_", "")  # Strip "base_" prefix if present
    parts = col.replace("pos_cov_y=", "").replace("_g=", ",").split(",")
    return int(parts[0]), int(parts[1])


def visualize_label_only(df, c_value):
    """
    Visualize class-conditional coverage: (base_)class_cond_cov_y={y}
    Compares FCP (base) vs FedCF across different label classes.
    """
    base_cols = [col for col in df.columns if col.startswith("base_class_cond_cov_y=")]
    method_cols = [
        col
        for col in df.columns
        if col.startswith("class_cond_cov_y=") and not col.startswith("base_")
    ]

    if not base_cols or not method_cols:
        print(f"  Warning: No class-conditional coverage columns found for c={c_value}")
        return

    # Extract label values (handle both int and float)
    def extract_numeric(col, prefix):
        val = col.replace(prefix, "")
        try:
            return int(val)
        except ValueError:
            return int(float(val))

    base_labels = sorted(
        [extract_numeric(col, "base_class_cond_cov_y=") for col in base_cols]
    )
    method_labels = sorted(
        [extract_numeric(col, "class_cond_cov_y=") for col in method_cols]
    )
    labels = sorted(set(base_labels + method_labels))

    # Compute means
    base_values = [
        df[f"base_class_cond_cov_y={y}"].apply(parse_tensor_value).mean()
        for y in labels
    ]
    method_values = [
        df[f"class_cond_cov_y={y}"].apply(parse_tensor_value).mean() for y in labels
    ]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, base_values, width, label="FCP (Base)", alpha=0.8)
    ax.bar(x + width / 2, method_values, width, label="FedCF", alpha=0.8)

    ax.set_xlabel("Label Class", fontsize=12)
    ax.set_ylabel("Coverage", fontsize=12)
    ax.set_title(
        f"Class-Conditional Coverage (c={c_value})", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"coverage_label_only_c={c_value}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Print summary
    print(f"\n  Class-Conditional Coverage Summary (c={c_value}):")
    for y, base_val, method_val in zip(labels, base_values, method_values):
        print(
            f"    y={y}: FCP={base_val:.4f}, FedCF={method_val:.4f}, Δ={method_val - base_val:+.4f}"
        )


def visualize_group_only(df, c_value):
    """
    Visualize group-conditional coverage: (base_)group_cond_cov_g={g}
    Compares FCP (base) vs FedCF across different demographic groups.
    """
    base_cols = [col for col in df.columns if col.startswith("base_group_cond_cov_g=")]
    method_cols = [
        col
        for col in df.columns
        if col.startswith("group_cond_cov_g=") and not col.startswith("base_")
    ]

    if not base_cols or not method_cols:
        print(f"  Warning: No group-conditional coverage columns found for c={c_value}")
        return

    # Extract group values (handle both int and float)
    def extract_numeric(col, prefix):
        val = col.replace(prefix, "")
        try:
            return int(val)
        except ValueError:
            return int(float(val))

    base_groups = sorted(
        [extract_numeric(col, "base_group_cond_cov_g=") for col in base_cols]
    )
    method_groups = sorted(
        [extract_numeric(col, "group_cond_cov_g=") for col in method_cols]
    )
    groups = sorted(set(base_groups + method_groups))

    # Compute means
    base_values = [
        df[f"base_group_cond_cov_g={g}"].apply(parse_tensor_value).mean()
        for g in groups
    ]
    method_values = [
        df[f"group_cond_cov_g={g}"].apply(parse_tensor_value).mean() for g in groups
    ]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(groups))
    width = 0.35

    ax.bar(x - width / 2, base_values, width, label="FCP (Base)", alpha=0.8)
    ax.bar(x + width / 2, method_values, width, label="FedCF", alpha=0.8)

    ax.set_xlabel("Demographic Group", fontsize=12)
    ax.set_ylabel("Coverage", fontsize=12)
    ax.set_title(
        f"Group-Conditional Coverage (c={c_value})", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"coverage_group_only_c={c_value}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Print summary
    print(f"\n  Group-Conditional Coverage Summary (c={c_value}):")
    for g, base_val, method_val in zip(groups, base_values, method_values):
        print(
            f"    g={g}: FCP={base_val:.4f}, FedCF={method_val:.4f}, Δ={method_val - base_val:+.4f}"
        )


def visualize_label_group(df, c_value):
    """
    Visualize joint label-group coverage: (base_)pos_cov_y={y}_g={g}
    Compares FCP (base) vs FedCF as 2D heatmaps (y classes vs g groups).
    """
    base_cols = [col for col in df.columns if col.startswith("base_pos_cov_y=")]
    method_cols = [
        col
        for col in df.columns
        if col.startswith("pos_cov_y=") and not col.startswith("base_")
    ]

    if not base_cols or not method_cols:
        print(f"  Warning: No joint coverage columns found for c={c_value}")
        return

    # Parse column names to extract (y, g) pairs
    base_pairs = {parse_column_name(col): col for col in base_cols}
    method_pairs = {parse_column_name(col): col for col in method_cols}
    all_pairs = sorted(set(base_pairs.keys()) | set(method_pairs.keys()))

    y_values = sorted(set(y for y, g in all_pairs))
    g_values = sorted(set(g for y, g in all_pairs))

    # Create matrices
    base_matrix = np.zeros((len(y_values), len(g_values)))
    method_matrix = np.zeros((len(y_values), len(g_values)))

    for i, y in enumerate(y_values):
        for j, g in enumerate(g_values):
            if (y, g) in base_pairs:
                base_matrix[i, j] = (
                    df[base_pairs[(y, g)]].apply(parse_tensor_value).mean()
                )
            if (y, g) in method_pairs:
                method_matrix[i, j] = (
                    df[method_pairs[(y, g)]].apply(parse_tensor_value).mean()
                )

    # Create side-by-side heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Base heatmap
    sns.heatmap(
        base_matrix,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        ax=axes[0],
        xticklabels=g_values,
        yticklabels=y_values,
        cbar_kws={"label": "Coverage"},
    )
    axes[0].set_title(
        f"FCP (Base) Joint Coverage (c={c_value})", fontsize=12, fontweight="bold"
    )
    axes[0].set_xlabel("Group (g)", fontsize=11)
    axes[0].set_ylabel("Label (y)", fontsize=11)

    # Method heatmap
    sns.heatmap(
        method_matrix,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        ax=axes[1],
        xticklabels=g_values,
        yticklabels=y_values,
        cbar_kws={"label": "Coverage"},
    )
    axes[1].set_title(
        f"FedCF Joint Coverage (c={c_value})", fontsize=12, fontweight="bold"
    )
    axes[1].set_xlabel("Group (g)", fontsize=11)
    axes[1].set_ylabel("Label (y)", fontsize=11)

    plt.tight_layout()
    plt.savefig(f"coverage_label_group_c={c_value}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Print summary statistics
    print(f"\n  Joint Coverage Summary (c={c_value}):")
    print(
        f"    Base (FCP)  - Mean: {base_matrix.mean():.4f}, Std: {base_matrix.std():.4f}"
    )
    print(
        f"    Method (FedCF) - Mean: {method_matrix.mean():.4f}, Std: {method_matrix.std():.4f}"
    )
    print(f"    Difference - Mean: {(method_matrix - base_matrix).mean():+.4f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Load data, extract unique c values, and generate visualizations per c value."""
    global csv_data

    # Load data
    if csv_data:
        df = pd.read_csv(StringIO(csv_data))
    else:
        df = pd.read_csv(csv_file_path)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Available c values: {sorted(df['c'].unique())}\n")

    # Process each unique c value
    c_values = sorted(df["c"].unique())
    for c_val in c_values:
        df_c = df[df["c"] == c_val]
        print(f"Processing c={c_val} ({len(df_c)} rows)")

        if option == "label_only":
            visualize_label_only(df_c, c_val)
        elif option == "group_only":
            visualize_group_only(df_c, c_val)
        elif option == "label_group":
            visualize_label_group(df_c, c_val)
        else:
            print(f"  Unknown option: {option}")

        print(f"  ✓ Saved: coverage_{option}_c={c_val}.png")

    print(f"\n✓ All visualizations completed!")


if __name__ == "__main__":
    main()
