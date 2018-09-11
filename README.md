# Zinga_Project
Business Problem Description
Beta is an online e-commerce company. The company is interested to know in an early stage, after their customer convert to a paid customer, whether they could become a VIP consumer of their website or not within a month ( 30 days). The have a dataset where is observed and aggregated during their first 7 days since the first date they made their first purchase. The dataset and its features are explained as below. Once they have the classifier, they could target those VIP customers with personalized treatment.

The Task
Use this dataset to build a binary classifier to use the first 7 days of data since a customer convert, whether they will become a VIP customer for the business within 30 days since their first conversion. The definition VIP by day 30 conversion is defined as a customer spend equal or more than $500 by day 30.

Dataset Description
1. IsVIP_500 : target variable, class label, 1 means is a VIP by day 30, 0 means not.
2. payment_7_day : total payment made by day 7 of conversion
3. dau_days: distinct days of customer login to the website.
4. days_between_install_first_pay: number of days since the user registered on the website
5. total_txns_7_day: total transactions the customer made on the website in the first 7 days.
6. total_page_views: number of product items the customer viewed on the website in the first 7 days.
7. total_product_liked: number of product items they have marked like during their views in the first 7 days
8. product_like_rate: the products liked divided by viewed products
9. total_free_coupon_got: number of free coupons the customer got during the first 7 days after conversion.
10. total_bonus_xp_points: total bonus points customer got during the first 7 days, where they could use it as cash with certain redeem rate.

Please refer to the python file for results. Task is implemented as below.
1. Performed statistical analysis for each feature.
2. Visualization of each feature and the target variable (class distribution).
3. Figured out if there any missing data in the dataset. Figured out the methods to deal with missing data.
4. Figured out if there is any outlier in the data set and the methods to deal with outliers.
5. Built diffrent classifiers and evaluated the results? Used diffrent metrics as Accuracy, Recall, Precision for evaluation.
6. Used diffrent classifiers and did parameters tuning to improve the accuracy.
7. As the data is highly imbalanced, implemented oversampling and undersampling methods to improve the final model.
