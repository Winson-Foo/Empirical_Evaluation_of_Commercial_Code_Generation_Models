
#%%
'''
follow the guide of webApi by SonarCloud
https://docs.sonarcloud.io/advanced-setup/web-api/
'''
import requests
# get the auth key
with open ("key.txt", "r") as myfile:
    auth_token = myfile.read().splitlines()[0]

# adding header and parameter
hed = {'Authorization': 'Bearer ' + auth_token}
param = {'componentKeys' : 'Winson-Foo_Empirical_Evaluation_of_Commercial_Code_Generation_Models', 'resolved':'false','branch':'test'}

main_url = 'https://sonarcloud.io/api/'
url = main_url + "issues/search"
response = requests.get(url, headers=hed, params=param)
issues = response.json()["issues"]

# br is before refactoring, ar is after refactoring
total_issues_br = 0
type_br = {"BUG":0 , "CODE_SMELL":0 , "VULNERABILITY":0}
severity_br = {"INFO":0 , "MINOR":0 , "MAJOR":0 , "CRITICAL":0 , "BLOCKER":0}

total_issues_ar_1 = 0
type_ar_1 = {"BUG":0 , "CODE_SMELL":0 , "VULNERABILITY":0}
severity_ar_1 = {"INFO":0 , "MINOR":0 , "MAJOR":0 , "CRITICAL":0 , "BLOCKER":0}

total_issues_ar_2 = 0
type_ar_2 = {"BUG":0 , "CODE_SMELL":0 , "VULNERABILITY":0}
severity_ar_2 = {"INFO":0 , "MINOR":0 , "MAJOR":0 , "CRITICAL":0 , "BLOCKER":0}

total_issues_ar_3 = 0
type_ar_3 = {"BUG":0 , "CODE_SMELL":0 , "VULNERABILITY":0}
severity_ar_3 = {"INFO":0 , "MINOR":0 , "MAJOR":0 , "CRITICAL":0 , "BLOCKER":0}

# count the severity and type of issue for the current analysis
for i in range(len(issues)):
    if ("Before_Refactor" in issues[i]["component"]):
        total_issues_br = total_issues_br + 1
        type_br[issues[i]['type']] = type_br[issues[i]['type']] + 1
        severity_br[issues[i]['severity']] = severity_br[issues[i]['severity']] + 1
    # After_Refactor
    elif ("After_Refactor_1" in issues[i]["component"]):
        total_issues_ar_1 = total_issues_ar_1 + 1
        type_ar_1[issues[i]['type']] = type_ar_1[issues[i]['type']] + 1
        severity_ar_1[issues[i]['severity']] = severity_ar_1[issues[i]['severity']] + 1
    elif ("After_Refactor_2" in issues[i]["component"]):
            total_issues_ar_2 = total_issues_ar_2 + 1
            type_ar_2[issues[i]['type']] = type_ar_2[issues[i]['type']] + 1
            severity_ar_2[issues[i]['severity']] = severity_ar_2[issues[i]['severity']] + 1
    elif ("After_Refactor_3" in issues[i]["component"]):
        total_issues_ar_3 = total_issues_ar_1 + 3
        type_ar_3[issues[i]['type']] = type_ar_3[issues[i]['type']] + 1
        severity_ar_3[issues[i]['severity']] = severity_ar_3[issues[i]['severity']] + 1

# display the results
print("Before Refactoring")
print("Total issues before refactoring : " + str(total_issues_br))
print(type_br)
print(severity_br)

print("------")
print("After Refactoring 1")
print("Total issues after refactoring : " + str(total_issues_ar_1))
print(type_ar_1)
print(severity_ar_1)

print("------")
print("After Refactoring 2")
print("Total issues after refactoring : " + str(total_issues_ar_2))
print(type_ar_2)
print(severity_ar_2)

print("------")
print("After Refactoring 3")
print("Total issues after refactoring : " + str(total_issues_ar_3))
print(type_ar_3)
print(severity_ar_3)
# %%
