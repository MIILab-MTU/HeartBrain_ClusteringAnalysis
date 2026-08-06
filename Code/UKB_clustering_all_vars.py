# Import libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st
import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.impute import KNNImputer
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency
from sklearn.metrics import (silhouette_score,davies_bouldin_score,calinski_harabasz_score,adjusted_mutual_info_score,)
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
from scipy.stats import tukey_hsd
from math import sqrt
import sys
from contextlib import redirect_stdout
from sklearn.preprocessing import StandardScaler
import os

#define working directory
wd = ""
#set working directory for the outputs of this analysis
os.chdir(wd + "Parents_AD")


# Read in data set
path_to_file = wd + "clustering_data_parents_ad.csv"
df = pd.read_csv(path_to_file)
#print(df.head(3))

# Creates a binary label based on any form of CVD diagnosis
new_label = []
for value in df["CVD"]:
    if value == False :
        new_label.append("_NOCVD")
    else:
        new_label.append("_CVD")
df["New_label"] = new_label
#print(df.head)

#creates composite IDs from AD diagnosis and eid
df[['eid']] = df[['eid']].astype(str)

df["New_ID"] = df[["eid", "New_label"]].apply(lambda x: "".join(x), axis=1)
df = df.set_index("New_ID", inplace=False)
#print(df.head)


# Deleting the worst predictors
# These are responses, identifiers, and uninformative predictors
# I don't think I need this step since I will be bringing in only the predictors that I want to include
del df["eid"]
del df["New_label"]
del df["CVD"]
del df["Ethnicity_Asian"]
del df["Ethnicity_Black"]
del df["Ethnicity_Chinese"]
del df["Ethnicity_Mixed"]
del df["Ethnicity_Other"]
print(df.head)

# Remove observations greater than 5 MAD from the median for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    median = df[col].median()
    mad = np.median(np.abs(df[col] - median))
    if mad == 0:  # avoid division by zero
        continue
    z_mad = np.abs(df[col] - median) / mad
    df = df[z_mad <= 5]  # keep only rows within 5 MAD
    
# Scale data
temp_scaler = StandardScaler()
df_scaled = pd.DataFrame(temp_scaler.fit_transform(df), columns=df.columns, index=df.index)
'''
#Determine number of clusters to use
#Performs clustering for each number of clusters from 3 to 10
for n_clusters in range(3, 11):
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="complete")
    clustering_model.fit(df_scaled)
    labels = clustering_model.labels_

    #the cluster evaluation metrics are calculated for each solution
    silhouette_avg = silhouette_score(df_scaled, labels)
    print("For ",n_clusters," clusters, the average silhouette score is : ",silhouette_avg,)
    davies_bouldin_avg = davies_bouldin_score(df_scaled, labels)
    print("For ",n_clusters," clusters, the average davies bouldin score is : ",davies_bouldin_avg,)
    calinski_harabasz_avg = calinski_harabasz_score(df_scaled, labels)
    print("For ",n_clusters," clusters, the average calinski harbabasz score is : ",calinski_harabasz_avg,)
    
    #this generates crosstabs between cluster labels and response status
    #prints response rates within each cluster
    df2 = pd.DataFrame({"Labels": labels, "CVD/No_CVD": df.index.tolist()})
    ct = pd.crosstab(df2["Labels"], df2["CVD/No_CVD"])
    arranged_labels = df2.sort_values("Labels")
    print("For ", n_clusters, "clusters, the response is as follows.")
    for x in range(0, n_clusters):
        labx = arranged_labels[(arranged_labels["Labels"] == x)]
        search1 = "_NOCVD"
        search2 = "_CVD"
        NOCVD_df = labx["CVD/No_CVD"].str.endswith(search1, na=False)
        CVD_df = labx["CVD/No_CVD"].str.endswith(search2, na=False)
        print("For cluster ", x + 1, "the response is as follows.")
        del labx["Labels"]
        print("No CVD ", labx[NOCVD_df].shape[0])
        print("CVD ", labx[CVD_df].shape[0])
        print("The CVD rate is",labx[CVD_df].shape[0]/ (labx[CVD_df].shape[0] + labx[NOCVD_df].shape[0]),)
'''


# Decide between 5 and 6 clusters
clustering_model = AgglomerativeClustering(n_clusters=4, metric="euclidean", linkage="complete")
clustering_model.fit(df_scaled)
labels4 = clustering_model.labels_

'''
clustering_model = AgglomerativeClustering(n_clusters=5, metric="euclidean", linkage="complete")
clustering_model.fit(df_scaled)
labels5 = clustering_model.labels_
print(adjusted_mutual_info_score(labels4, labels5))
'''

# Original data dendrogram
mergings = linkage(df_scaled, method="complete")
plt.figure(figsize=(20, 8))
dendrogram(mergings, truncate_mode="lastp", p=100, leaf_rotation=90, leaf_font_size=10, color_threshold=mergings[-6, 2])
plt.axhline(y=26, c="blue", lw=1, linestyle="dashed")
plt.title("Original Data Dendrogram")
plt.savefig(f"dendrogram.png", dpi=300)
plt.close()

# Shuffled data dendrogram
df_shuf = df_scaled.sample(frac=1)
df_shuf.index.tolist()
mergings1 = linkage(df_shuf, method="complete")
plt.figure(figsize=(20, 8))
dendrogram(mergings1, truncate_mode="lastp", p=100, leaf_rotation=90, leaf_font_size=10, color_threshold=mergings[-6, 2])
plt.axhline(y=26, c="red", lw=1, linestyle="dashed")
plt.title("Shuffled Data Dendrogram")
plt.show()

# Create flat clusters by cutting 
#labels = fcluster(mergings, 1200, criterion="distance")
labels = clustering_model.labels_ + 1

# map cluster labels back to response labels, assign to individual cluster dataframes
df2 = pd.DataFrame({"Labels": labels, "CVD/No_CVD": df.index.tolist()})
arranged_labels = df2.sort_values("Labels")
ct = pd.crosstab(df2["Labels"], df2["CVD/No_CVD"])
# Create cluster 1
lab1 = arranged_labels[(arranged_labels["Labels"] == 1)]
search1 = "_NOCVD"
search2 = "_CVD"
NOCVD_df1 = lab1["CVD/No_CVD"].str.endswith(search1, na=False)
CVD_df1 = lab1["CVD/No_CVD"].str.endswith(search2, na=False)
print(lab1[NOCVD_df1].count())
print(lab1[CVD_df1].count())
lab1 = lab1.set_index("CVD/No_CVD", inplace=False)
clust1 = lab1.index.tolist()
clust_1 = df[df.index.isin(clust1)]
clust_1["Cluster"] = 1
# Create cluster 2
lab2 = arranged_labels[(arranged_labels["Labels"] == 2)]
NOCVD_df2 = lab2["CVD/No_CVD"].str.endswith(search1, na=False)
CVD_df2 = lab2["CVD/No_CVD"].str.endswith(search2, na=False)
print(lab2[NOCVD_df2].count())
print(lab2[CVD_df2].count())
lab2 = lab2.set_index("CVD/No_CVD", inplace=False)
clust2 = lab2.index.tolist()
clust_2 = df[df.index.isin(clust2)]
clust_2["Cluster"] = 2
# Create cluster 3
lab3 = arranged_labels[(arranged_labels["Labels"] == 3)]
NOCVD_df3 = lab3["CVD/No_CVD"].str.endswith(search1, na=False)
CVD_df3 = lab3["CVD/No_CVD"].str.endswith(search2, na=False)
print(lab3[NOCVD_df3].count())
print(lab3[CVD_df3].count())
lab3 = lab3.set_index("CVD/No_CVD", inplace=False)
clust3 = lab3.index.tolist()
clust_3 = df[df.index.isin(clust3)]
clust_3["Cluster"] = 3
# Create cluster 4
lab4 = arranged_labels[(arranged_labels["Labels"] == 4)]
NOCVD_df4 = lab4["CVD/No_CVD"].str.endswith(search1, na=False)
CVD_df4 = lab4["CVD/No_CVD"].str.endswith(search2, na=False)
print(lab4[NOCVD_df4].count())
print(lab4[CVD_df4].count())
lab4 = lab4.set_index("CVD/No_CVD", inplace=False)
clust4 = lab4.index.tolist()
clust_4 = df[df.index.isin(clust4)]
clust_4["Cluster"] = 4
'''
# Create cluster 5
lab5 = arranged_labels[(arranged_labels["Labels"] == 5)]
NOCVD_df5 = lab5["CVD/No_CVD"].str.endswith(search1, na=False)
CVD_df5 = lab5["CVD/No_CVD"].str.endswith(search2, na=False)
print(lab5[NOCVD_df5].count())
print(lab5[CVD_df5].count())
lab5 = lab5.set_index("CVD/No_CVD", inplace=False)
clust5 = lab5.index.tolist()
clust_5 = df[df.index.isin(clust5)]
clust_5["Cluster"] = 5
# Create cluster 6
lab6 = arranged_labels[(arranged_labels["Labels"] == 6)]
NOCVD_df6 = lab6["CVD/No_CVD"].str.endswith(search1, na=False)
CVD_df6 = lab6["CVD/No_CVD"].str.endswith(search2, na=False)
print(lab6[NOCVD_df6].count())
print(lab6[CVD_df6].count())
lab6 = lab6.set_index("CVD/No_CVD", inplace=False)
clust6 = lab6.index.tolist()
clust_6 = df[df.index.isin(clust6)]
clust_6["Cluster"] = 6
'''
# Combine all clusters into one dataframe
df3 = pd.concat([clust_1, clust_2, clust_3, clust_4])
# Save the combined dataframe with cluster labels to CSV
output_path = "cluster_results_allVars.csv"
df3.to_csv(output_path, index=True)
print(f"Clustered data with group labels saved to {output_path}")

# Check for significance of categorical variables
# Perfrom chi-squared test
with open("clustering_analysis_output.txt", "w") as f:
    with redirect_stdout(f):
        for column in df3[["Hypertension","Hypotension","Angina.Pectoris","Acute.Myocardial.Infarction","Pulmonary.Embolism","Atrial.Fibriliation","Heart.Failure","Blockage","Chronic.Rheumatic.Heart.Disease","Chronic.Ischemic.Heart.Disease","Cerebral.Infarction","Alzheimers.Disease","Sex_Male","Smoking_Current", "Smoking_Previous", "Depression.Bipolar_Yes", "Diabetes", "statins","diuretics","beta_blockers","angiotensin_antagonists","calcium_channel_blockers","anticoagulants", "Father.Alzheimers", "Mother.Alzheimers"]]:
            columnSeriesObj = df3[column]
            column_crosstab = pd.crosstab(columnSeriesObj, df3["Cluster"], margins=False)
            print(column_crosstab.to_string())
            # Crosstab percentages (prevalence of '1' in each cluster)
            column_crosstab_pct = pd.crosstab(df3[column], df3["Cluster"], normalize='columns') * 100
            print(f"Prevalence (%) for each level of {column}:\n", column_crosstab_pct.to_string())
            stat, p, dof, expected = chi2_contingency(column_crosstab)
            alpha = 0.05
            print("significance=%.3f,p=%.3f" % (alpha, p))
            if p <= alpha:
                print("Variables are associated: Reject H0")
            else:
                print("Variables are not associated: Do not reject H0")

        # Check for signficance of all continous variables
        # Run ANOVA testing
        # same variable name considerations as before
        for column in df3[["heartPC1","heartPC2","heartPC3","brainPC1","brainPC2","brainPC3","brainPC4","DBP","SBP","LDL","HDL","Age","BMI","cognitivePC1","Physical.Activity", "Alcohol", "Income","Education"]]:
            columnSeriesObj2 = df3[column]
            print("ANOVA for :", column)
            mod = ols("columnSeriesObj2~Cluster", data=df3).fit()
            anova_table = sm.stats.anova_lm(mod, typ=2)
            print(anova_table)

        #Perform Tukey's HSD for continous variables 
        # See which clusters significantly differ after ANOVA
        for column in df3[["heartPC1","heartPC2","heartPC3","brainPC1","brainPC2","brainPC3","brainPC4","DBP","SBP","LDL","HDL","Age","BMI","cognitivePC1","Physical.Activity", "Alcohol", "Income","Education"]]:
            print("Tukey's HSD for "+ column)
            tukey = tukey_hsd(clust_1[column],clust_2[column],clust_3[column],clust_4[column])
            print(tukey)
 

        # Analyze the data for all relevant continous variables
        # Calculate mean, SD, and 95% CI for each cluster and overall
        #Calculate Means for each cluster
        print("Cluster 1 means")
        print(clust_1.mean())
        print("Cluster 2 means")
        print(clust_2.mean())
        print("Cluster 3 means")
        print(clust_3.mean())
        print("Cluster 4 means")
        print(clust_4.mean())
        #print("Cluster 5 means")
        #print(clust_5.mean())
        #print("Cluster 6 means")
        #print(clust_6.mean())
        print("Overall means")
        print(df3.mean())
        #Calculate standard deviations for each cluster
        print("Cluster 1 sds")
        print(clust_1.std())
        print("Cluster 2 sds")
        print(clust_2.std())
        print("Cluster 3 sds")
        print(clust_3.std())
        print("Cluster 4 sds")
        print(clust_4.std())
        #print("Cluster 5 sds")
        #print(clust_5.std())
        #print("Cluster 6 sds")
        #print(clust_6.std())
        print("Overall standard deviations")
        print(df3.std())
        #Calculate 95 percent confidence intervals for each cluster
        print("Overall CI's")
        for column in df3[["heartPC1","heartPC2","heartPC3","brainPC1","brainPC2","brainPC3","brainPC4","DBP","SBP","LDL","HDL","Age","BMI","cognitivePC1","Physical.Activity", "Alcohol", "Income","Education"]]:
            print("CI for " + column)
            print(st.norm.interval(confidence=.95, loc=np.mean(df3[column]), scale= st.sem(df3[column])))
        print("Cluster 1 CI's")
        for column in clust_1[["heartPC1","heartPC2","heartPC3","brainPC1","brainPC2","brainPC3","brainPC4","DBP","SBP","LDL","HDL","Age","BMI","cognitivePC1","Physical.Activity", "Alcohol", "Income","Education"]]:
            print("CI for " + column)
            print(st.norm.interval(confidence=.95, loc=np.mean(clust_1[column]), scale= st.sem(clust_1[column])))
        print("Cluster 2 CI's")
        for column in clust_2[["heartPC1","heartPC2","heartPC3","brainPC1","brainPC2","brainPC3","brainPC4","DBP","SBP","LDL","HDL","Age","BMI","cognitivePC1","Physical.Activity", "Alcohol", "Income","Education"]]:
            print("CI for " + column)
            print(st.norm.interval(confidence=.95, loc=np.mean(clust_2[column]), scale= st.sem(clust_2[column])))
        print("Cluster 3 CI's")
        for column in clust_3[["heartPC1","heartPC2","heartPC3","brainPC1","brainPC2","brainPC3","brainPC4","DBP","SBP","LDL","HDL","Age","BMI","cognitivePC1","Physical.Activity", "Alcohol", "Income","Education"]]:
            print("CI for " + column)
            print(st.norm.interval(confidence=.95, loc=np.mean(clust_3[column]), scale= st.sem(clust_3[column])))
        print("Cluster 4 CI's")
        for column in clust_4[["heartPC1","heartPC2","heartPC3","brainPC1","brainPC2","brainPC3","brainPC4","DBP","SBP","LDL","HDL","Age","BMI","cognitivePC1","Physical.Activity", "Alcohol", "Income","Education"]]:
            print("CI for " + column)
            print(st.norm.interval(confidence=.95, loc=np.mean(clust_4[column]), scale= st.sem(clust_4[column])))
        '''
        print("Cluster 5 CI's")
        for column in clust_5[["heartPC1","heartPC2","heartPC3","brainPC1","brainPC2","brainPC3","brainPC4","DBP","SBP","LDL","HDL","Age","BMI","cognitivePC1","Physical.Activity", "Alcohol", "Income","Education"]]:
            print("CI for " + column)
            print(st.norm.interval(confidence=.95, loc=np.mean(clust_5[column]), scale= st.sem(clust_5[column])))
        
        print("Cluster 6 CI's")
        for column in clust_6[["heartPC1","heartPC2","heartPC3","brainPC1","brainPC2","brainPC3","brainPC4","DBP","SBP","LDL","HDL","Age","BMI","cognitivePC1","Physical.Activity", "Alcohol", "Income","Education"]]:
            print("CI for " + column)
            print(st.norm.interval(confidence=.95, loc=np.mean(clust_6[column]), scale= st.sem(clust_6[column])))
        #'''


# Generate boxplots of some relevant variables
# visualize distribution of variables across clusters
# I will have to edit this after running the code to find out which of my variables will be of interest for plotting
'''
df3.boxplot("heartPC1", by="Cluster")
df3.boxplot("SPECT_pre_ESE", by="Cluster")
df3.boxplot("SPECT_pre_SRscore", by="Cluster")
df3.boxplot("SPECT_pre_StrokeVolume", by="Cluster")
df3.boxplot("SPECT_pre_gMyoMass", by="Cluster")
df3.boxplot("SPECT_pre_50scar", by="Cluster")
df3.boxplot("SPECT_pre_Diastolic_PBW", by="Cluster")
df3.boxplot("SPECT_pre_Diastolic_PSD", by="Cluster")
df3.boxplot("SPECT_pre_PBW", by="Cluster")
df3.boxplot("SPECT_pre_PSD", by="Cluster")
df3.boxplot("SPECT_pre_PSD_noScar", by="Cluster")
df3.boxplot("SPECT_pre_PBW_noScar", by="Cluster")
'''

#Plot confidence intervals for numeric variables of interest
#draw points and error bars
#I'm not sure why this is necessary if we also have boxplots, seems like they would do pretty much the same thing
def plt_CI(x, data):
    mean=data.mean()
    std= data.std()
    confidence_interval= 1.96*std/sqrt(len(data))
    left= x-.10
    top= mean + confidence_interval
    right= x+ .10
    bottom= mean - confidence_interval
    plt.plot([x,x], [top,bottom],color='blue')
    plt.plot([left,right],[top,top],color='blue')
    plt.plot([left,right],[bottom,bottom],color='blue')
    plt.plot(x, mean, 'o', color='red')
    return mean, confidence_interval
for column in df3[["heartPC1","heartPC2","heartPC3","brainPC1","brainPC2","brainPC3","brainPC4","DBP","SBP","LDL","HDL","Age","BMI","cognitivePC1","Physical.Activity", "Alcohol", "Income","Education"]]:
    plt.figure(column)
    plt.xticks([1,2,3,4],['Cluster1','Cluster 2','Cluster 3','Cluster 4'])
    plt.title('Confidence Intervals for '+ column)
    plt_CI(1,clust_1[column])
    plt_CI(2,clust_2[column])
    plt_CI(3,clust_3[column])
    plt_CI(4,clust_4[column])
    #plt_CI(5,clust_5[column])
    #plt_CI(6,clust_6[column])
    plt.axhline(df3[column].mean(),color='red',linestyle='--')
    plt.savefig(f"CI_plot_{column}.png", dpi=300)
    plt.close() 