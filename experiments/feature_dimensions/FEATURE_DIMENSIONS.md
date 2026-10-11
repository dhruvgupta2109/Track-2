# Verified feature dimensions

Source of truth: saved FINAL_SUBMISSION_Track2_C2/model.pkl and notebook definition cells 3-5 (1-based). The train-only candidate has identical dimensions and feature names.

Actual instrumented LR.fit calls from reproducing training factories:
- model: X=[20000, 73], C=0.1, maximum coefficient difference vs saved C2=5.9e-14.
- poly_model: X=[20000, 147], C=0.003, maximum coefficient difference vs saved C2=4.99e-15.
- galr_model: X=[20000, 70], C=0.05, maximum coefficient difference vs saved C2=1.41e-11.

Train columns: id, transaction_amount, transaction_hour, merchant_category, country, transaction_channel, transactions_last_24h, spend_last_24h, account_age, new_device, transactions_last_1h, fraud
Test columns: id, transaction_amount, transaction_hour, merchant_category, country, transaction_channel, transactions_last_24h, spend_last_24h, account_age, new_device, transactions_last_1h

No model uses id or fraud as predictors. Numeric imputers add no missing indicators; one-hot encoders have drop=None and handle_unknown="ignore". ColumnTransformer remainder="drop". Each model has one separate intercept.

## model

Pipeline DataFrame: [12000, 25]; estimator matrix: [12000, 73]; coefficient shape: [1, 73].

### Input DataFrame columns

1. `merchant_category`
2. `country`
3. `transaction_channel`
4. `hc`
5. `log_amt`
6. `log_spend`
7. `log_age`
8. `new_device`
9. `transactions_last_24h`
10. `amt_rel_cat`
11. `avg_spend`
12. `amt_rel_avg`
13. `young`
14. `t24_hi`
15. `burst`
16. `night`
17. `foreign`
18. `t1_ge4_nb`
19. `nd_x_logage`
20. `nd_x_night`
21. `age_lt120`
22. `amt_rel_transaction_channel`
23. `amt_rel_country`
24. `amt_pct_cat`
25. `d_ratio`

### Full ordered estimator feature list

1. `c__merchant_category_NA`
2. `c__merchant_category_cash_transfer`
3. `c__merchant_category_electronics`
4. `c__merchant_category_entertainment`
5. `c__merchant_category_food_delivery`
6. `c__merchant_category_gaming`
7. `c__merchant_category_grocery`
8. `c__merchant_category_luxury`
9. `c__merchant_category_retail`
10. `c__merchant_category_transport`
11. `c__merchant_category_travel`
12. `c__merchant_category_utilities`
13. `c__country_AU`
14. `c__country_GB`
15. `c__country_ID`
16. `c__country_JP`
17. `c__country_MY`
18. `c__country_NA`
19. `c__country_PH`
20. `c__country_SG`
21. `c__country_TH`
22. `c__country_US`
23. `c__country_VN`
24. `c__transaction_channel_NA`
25. `c__transaction_channel_bank_transfer`
26. `c__transaction_channel_card_present`
27. `c__transaction_channel_ecommerce`
28. `c__transaction_channel_mobile_app`
29. `c__hc_0`
30. `c__hc_1`
31. `c__hc_10`
32. `c__hc_11`
33. `c__hc_12`
34. `c__hc_13`
35. `c__hc_14`
36. `c__hc_15`
37. `c__hc_16`
38. `c__hc_17`
39. `c__hc_18`
40. `c__hc_19`
41. `c__hc_2`
42. `c__hc_20`
43. `c__hc_21`
44. `c__hc_22`
45. `c__hc_23`
46. `c__hc_3`
47. `c__hc_4`
48. `c__hc_5`
49. `c__hc_6`
50. `c__hc_7`
51. `c__hc_8`
52. `c__hc_9`
53. `n__log_amt`
54. `n__log_spend`
55. `n__log_age`
56. `n__new_device`
57. `n__transactions_last_24h`
58. `n__amt_rel_cat`
59. `n__avg_spend`
60. `n__amt_rel_avg`
61. `n__young`
62. `n__t24_hi`
63. `n__burst`
64. `n__night`
65. `n__foreign`
66. `n__t1_ge4_nb`
67. `n__nd_x_logage`
68. `n__nd_x_night`
69. `n__age_lt120`
70. `n__amt_rel_transaction_channel`
71. `n__amt_rel_country`
72. `n__amt_pct_cat`
73. `n__d_ratio`

### Transform branches

`c`: merchant_category, country, transaction_channel, hc
- SimpleImputer: [12000, 4] -> [12000, 4]
- OneHotEncoder: [12000, 4] -> [12000, 52]
  Category counts: [12, 11, 5, 24]; learned values: [['NA', 'cash_transfer', 'electronics', 'entertainment', 'food_delivery', 'gaming', 'grocery', 'luxury', 'retail', 'transport', 'travel', 'utilities'], ['AU', 'GB', 'ID', 'JP', 'MY', 'NA', 'PH', 'SG', 'TH', 'US', 'VN'], ['NA', 'bank_transfer', 'card_present', 'ecommerce', 'mobile_app'], ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '3', '4', '5', '6', '7', '8', '9']]
`n`: log_amt, log_spend, log_age, new_device, transactions_last_24h, amt_rel_cat, avg_spend, amt_rel_avg, young, t24_hi, burst, night, foreign, t1_ge4_nb, nd_x_logage, nd_x_night, age_lt120, amt_rel_transaction_channel, amt_rel_country, amt_pct_cat, d_ratio
- SimpleImputer: [12000, 21] -> [12000, 21]
- StandardScaler: [12000, 21] -> [12000, 21]

## poly_model

Pipeline DataFrame: [12000, 17]; estimator matrix: [12000, 147]; coefficient shape: [1, 147].

### Input DataFrame columns

1. `log_amt`
2. `log_spend`
3. `log_age`
4. `transactions_last_24h`
5. `transactions_last_1h`
6. `new_device`
7. `amt_vs_cat`
8. `foreign`
9. `young`
10. `burst`
11. `night`
12. `merchant_category`
13. `merchant_category_risk`
14. `country`
15. `country_risk`
16. `transaction_channel`
17. `transaction_channel_risk`

### Full ordered estimator feature list

1. `num__log_amt`
2. `num__log_spend`
3. `num__log_age`
4. `num__transactions_last_24h`
5. `num__transactions_last_1h`
6. `num__new_device`
7. `num__amt_vs_cat`
8. `num__foreign`
9. `num__young`
10. `num__burst`
11. `num__night`
12. `num__merchant_category_risk`
13. `num__country_risk`
14. `num__transaction_channel_risk`
15. `num__log_amt^2`
16. `num__log_amt log_spend`
17. `num__log_amt log_age`
18. `num__log_amt transactions_last_24h`
19. `num__log_amt transactions_last_1h`
20. `num__log_amt new_device`
21. `num__log_amt amt_vs_cat`
22. `num__log_amt foreign`
23. `num__log_amt young`
24. `num__log_amt burst`
25. `num__log_amt night`
26. `num__log_amt merchant_category_risk`
27. `num__log_amt country_risk`
28. `num__log_amt transaction_channel_risk`
29. `num__log_spend^2`
30. `num__log_spend log_age`
31. `num__log_spend transactions_last_24h`
32. `num__log_spend transactions_last_1h`
33. `num__log_spend new_device`
34. `num__log_spend amt_vs_cat`
35. `num__log_spend foreign`
36. `num__log_spend young`
37. `num__log_spend burst`
38. `num__log_spend night`
39. `num__log_spend merchant_category_risk`
40. `num__log_spend country_risk`
41. `num__log_spend transaction_channel_risk`
42. `num__log_age^2`
43. `num__log_age transactions_last_24h`
44. `num__log_age transactions_last_1h`
45. `num__log_age new_device`
46. `num__log_age amt_vs_cat`
47. `num__log_age foreign`
48. `num__log_age young`
49. `num__log_age burst`
50. `num__log_age night`
51. `num__log_age merchant_category_risk`
52. `num__log_age country_risk`
53. `num__log_age transaction_channel_risk`
54. `num__transactions_last_24h^2`
55. `num__transactions_last_24h transactions_last_1h`
56. `num__transactions_last_24h new_device`
57. `num__transactions_last_24h amt_vs_cat`
58. `num__transactions_last_24h foreign`
59. `num__transactions_last_24h young`
60. `num__transactions_last_24h burst`
61. `num__transactions_last_24h night`
62. `num__transactions_last_24h merchant_category_risk`
63. `num__transactions_last_24h country_risk`
64. `num__transactions_last_24h transaction_channel_risk`
65. `num__transactions_last_1h^2`
66. `num__transactions_last_1h new_device`
67. `num__transactions_last_1h amt_vs_cat`
68. `num__transactions_last_1h foreign`
69. `num__transactions_last_1h young`
70. `num__transactions_last_1h burst`
71. `num__transactions_last_1h night`
72. `num__transactions_last_1h merchant_category_risk`
73. `num__transactions_last_1h country_risk`
74. `num__transactions_last_1h transaction_channel_risk`
75. `num__new_device^2`
76. `num__new_device amt_vs_cat`
77. `num__new_device foreign`
78. `num__new_device young`
79. `num__new_device burst`
80. `num__new_device night`
81. `num__new_device merchant_category_risk`
82. `num__new_device country_risk`
83. `num__new_device transaction_channel_risk`
84. `num__amt_vs_cat^2`
85. `num__amt_vs_cat foreign`
86. `num__amt_vs_cat young`
87. `num__amt_vs_cat burst`
88. `num__amt_vs_cat night`
89. `num__amt_vs_cat merchant_category_risk`
90. `num__amt_vs_cat country_risk`
91. `num__amt_vs_cat transaction_channel_risk`
92. `num__foreign^2`
93. `num__foreign young`
94. `num__foreign burst`
95. `num__foreign night`
96. `num__foreign merchant_category_risk`
97. `num__foreign country_risk`
98. `num__foreign transaction_channel_risk`
99. `num__young^2`
100. `num__young burst`
101. `num__young night`
102. `num__young merchant_category_risk`
103. `num__young country_risk`
104. `num__young transaction_channel_risk`
105. `num__burst^2`
106. `num__burst night`
107. `num__burst merchant_category_risk`
108. `num__burst country_risk`
109. `num__burst transaction_channel_risk`
110. `num__night^2`
111. `num__night merchant_category_risk`
112. `num__night country_risk`
113. `num__night transaction_channel_risk`
114. `num__merchant_category_risk^2`
115. `num__merchant_category_risk country_risk`
116. `num__merchant_category_risk transaction_channel_risk`
117. `num__country_risk^2`
118. `num__country_risk transaction_channel_risk`
119. `num__transaction_channel_risk^2`
120. `cat__merchant_category_NA`
121. `cat__merchant_category_cash_transfer`
122. `cat__merchant_category_electronics`
123. `cat__merchant_category_entertainment`
124. `cat__merchant_category_food_delivery`
125. `cat__merchant_category_gaming`
126. `cat__merchant_category_grocery`
127. `cat__merchant_category_luxury`
128. `cat__merchant_category_retail`
129. `cat__merchant_category_transport`
130. `cat__merchant_category_travel`
131. `cat__merchant_category_utilities`
132. `cat__country_AU`
133. `cat__country_GB`
134. `cat__country_ID`
135. `cat__country_JP`
136. `cat__country_MY`
137. `cat__country_NA`
138. `cat__country_PH`
139. `cat__country_SG`
140. `cat__country_TH`
141. `cat__country_US`
142. `cat__country_VN`
143. `cat__transaction_channel_NA`
144. `cat__transaction_channel_bank_transfer`
145. `cat__transaction_channel_card_present`
146. `cat__transaction_channel_ecommerce`
147. `cat__transaction_channel_mobile_app`

### Transform branches

`num`: log_amt, log_spend, log_age, transactions_last_24h, transactions_last_1h, new_device, amt_vs_cat, foreign, young, burst, night, merchant_category_risk, country_risk, transaction_channel_risk
- SimpleImputer: [12000, 14] -> [12000, 14]
- StandardScaler: [12000, 14] -> [12000, 14]
- PolynomialFeatures: [12000, 14] -> [12000, 119]
  Degree=2, interaction_only=False, include_bias=False; 14 linear + 14 squares + 91 cross products.
- StandardScaler: [12000, 119] -> [12000, 119]
`cat`: merchant_category, country, transaction_channel
- SimpleImputer: [12000, 3] -> [12000, 3]
- OneHotEncoder: [12000, 3] -> [12000, 28]
  Category counts: [12, 11, 5]; learned values: [['NA', 'cash_transfer', 'electronics', 'entertainment', 'food_delivery', 'gaming', 'grocery', 'luxury', 'retail', 'transport', 'travel', 'utilities'], ['AU', 'GB', 'ID', 'JP', 'MY', 'NA', 'PH', 'SG', 'TH', 'US', 'VN'], ['NA', 'bank_transfer', 'card_present', 'ecommerce', 'mobile_app']]

## galr_model

Pipeline DataFrame: [12000, 22]; estimator matrix: [12000, 70]; coefficient shape: [1, 70].

### Input DataFrame columns

1. `amt_gt100`
2. `amt_gt200`
3. `amt_gt300`
4. `amt_gt500`
5. `amt_gt750`
6. `amt_gt1000`
7. `amt_gt1500`
8. `amt_gt2500`
9. `night`
10. `age_lt120`
11. `age_lt270`
12. `age_lt365`
13. `t1_pos`
14. `t24_clip`
15. `t24_excess`
16. `new_device`
17. `nd_x_night`
18. `nd_x_ecom`
19. `merchant_category`
20. `country`
21. `transaction_channel`
22. `hour_c`

### Full ordered estimator feature list

1. `c__merchant_category_NA`
2. `c__merchant_category_cash_transfer`
3. `c__merchant_category_electronics`
4. `c__merchant_category_entertainment`
5. `c__merchant_category_food_delivery`
6. `c__merchant_category_gaming`
7. `c__merchant_category_grocery`
8. `c__merchant_category_luxury`
9. `c__merchant_category_retail`
10. `c__merchant_category_transport`
11. `c__merchant_category_travel`
12. `c__merchant_category_utilities`
13. `c__country_AU`
14. `c__country_GB`
15. `c__country_ID`
16. `c__country_JP`
17. `c__country_MY`
18. `c__country_NA`
19. `c__country_PH`
20. `c__country_SG`
21. `c__country_TH`
22. `c__country_US`
23. `c__country_VN`
24. `c__transaction_channel_NA`
25. `c__transaction_channel_bank_transfer`
26. `c__transaction_channel_card_present`
27. `c__transaction_channel_ecommerce`
28. `c__transaction_channel_mobile_app`
29. `c__hour_c_0`
30. `c__hour_c_1`
31. `c__hour_c_10`
32. `c__hour_c_11`
33. `c__hour_c_12`
34. `c__hour_c_13`
35. `c__hour_c_14`
36. `c__hour_c_15`
37. `c__hour_c_16`
38. `c__hour_c_17`
39. `c__hour_c_18`
40. `c__hour_c_19`
41. `c__hour_c_2`
42. `c__hour_c_20`
43. `c__hour_c_21`
44. `c__hour_c_22`
45. `c__hour_c_23`
46. `c__hour_c_3`
47. `c__hour_c_4`
48. `c__hour_c_5`
49. `c__hour_c_6`
50. `c__hour_c_7`
51. `c__hour_c_8`
52. `c__hour_c_9`
53. `n__amt_gt100`
54. `n__amt_gt200`
55. `n__amt_gt300`
56. `n__amt_gt500`
57. `n__amt_gt750`
58. `n__amt_gt1000`
59. `n__amt_gt1500`
60. `n__amt_gt2500`
61. `n__night`
62. `n__age_lt120`
63. `n__age_lt270`
64. `n__age_lt365`
65. `n__t1_pos`
66. `n__t24_clip`
67. `n__t24_excess`
68. `n__new_device`
69. `n__nd_x_night`
70. `n__nd_x_ecom`

### Transform branches

`c`: merchant_category, country, transaction_channel, hour_c
- SimpleImputer: [12000, 4] -> [12000, 4]
- OneHotEncoder: [12000, 4] -> [12000, 52]
  Category counts: [12, 11, 5, 24]; learned values: [['NA', 'cash_transfer', 'electronics', 'entertainment', 'food_delivery', 'gaming', 'grocery', 'luxury', 'retail', 'transport', 'travel', 'utilities'], ['AU', 'GB', 'ID', 'JP', 'MY', 'NA', 'PH', 'SG', 'TH', 'US', 'VN'], ['NA', 'bank_transfer', 'card_present', 'ecommerce', 'mobile_app'], ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '3', '4', '5', '6', '7', '8', '9']]
`n`: amt_gt100, amt_gt200, amt_gt300, amt_gt500, amt_gt750, amt_gt1000, amt_gt1500, amt_gt2500, night, age_lt120, age_lt270, age_lt365, t1_pos, t24_clip, t24_excess, new_device, nd_x_night, nd_x_ecom
- SimpleImputer: [12000, 18] -> [12000, 18]
- StandardScaler: [12000, 18] -> [12000, 18]

## Model 2 decomposition

11 core + 3 smoothed category log-odds = 14 numeric inputs. Standardize BEFORE expansion. Degree-two expansion is 14 linear + 14 squares + C(14,2)=91 cross-products = 119 numeric terms. Standardize again, then concatenate 12 merchant + 11 country + 5 channel one-hot columns = 28. Total 147 coefficients plus a separate intercept.

By source: 11 core linear + 3 risk linear + 11 core squares + 3 risk squares + C(11,2)=55 core/core products + 11*3=33 core/risk products + C(3,2)=3 risk/risk products + 28 categorical one-hot columns = 147.

Risk mapping: p_g=(fraud_count_g+20*training_prior)/(group_count_g+20); feature=log(p_g/(1-p_g)). Missing categories use NA; unseen categories fall back to prior log-odds. These risk features go through polynomial expansion alongside the core.

## Documentation discrepancy

The submitted technical_report.pdf gives no dimensionalities or 147 claim. Its pairwise-interaction description and notebook introduction omit squares and products involving risk-map features. The polynomial expansion includes 14 inputs, not only the 11 core features. Model 2 has no hour one-hot encoding.

No saved models or submission notebooks were changed; fresh reproductions were fitted only in memory. Reproduce with `.venv_clean/Scripts/python.exe src/inspect_feature_dimensions.py`. The JSON contains all polynomial powers and branch shapes.
