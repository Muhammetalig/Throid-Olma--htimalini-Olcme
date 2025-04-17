from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
differentiated_thyroid_cancer_recurrence = fetch_ucirepo(id=915) 
  
# data (as pandas dataframes) 
X = differentiated_thyroid_cancer_recurrence.data.features 
y = differentiated_thyroid_cancer_recurrence.data.targets 
  
# metadata 
print(differentiated_thyroid_cancer_recurrence.metadata) 
  
# variable information 
print(differentiated_thyroid_cancer_recurrence.variables) 
