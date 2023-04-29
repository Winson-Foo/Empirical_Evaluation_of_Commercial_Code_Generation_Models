
#%%
'''
follow the guide of webApi by SonarCloud
https://docs.sonarcloud.io/advanced-setup/web-api/
'''
import requests
# get the auth key
with open ("key.txt", "r") as myfile:
    auth_token = myfile.read().splitlines()[0]

FILE_NAME = ["Before_Refactor","After_Refactor_1","After_Refactor_2","After_Refactor_3"]

# adding header and parameter
hed = {'Authorization': 'Bearer ' + auth_token}

main_url = 'https://sonarcloud.io/api/'
url = main_url + "issues/search"
issues_matrix = []

# appending all matrix
for i in range(len(FILE_NAME)):
    issues_matrix.append([0,{"BUG":0 , "CODE_SMELL":0 , "VULNERABILITY":0},{"INFO":0 , "MINOR":0 , "MAJOR":0 , "CRITICAL":0 , "BLOCKER":0}])

# going through the each folder
for i in range(len(FILE_NAME)):
    param = {'componentKeys' : 'Winson-Foo_Empirical_Evaluation_of_Commercial_Code_Generation_Models:' + FILE_NAME[i], 'resolved':'false','branch':'test','ps':'500'}
    response = requests.get(url, headers=hed, params=param)
    all_issues = response.json()['issues']
    for k in range(len(all_issues)):
        # number of issues
        issues_matrix[i][0] = issues_matrix[i][0] + 1
        # type of errer
        issues_matrix[i][1][all_issues[k]['type']] = issues_matrix[i][1][all_issues[k]['type']] + 1
        # severity of the error
        issues_matrix[i][2][all_issues[k]['severity']] = issues_matrix[i][2][all_issues[k]['severity']] + 1

# displaying all the value
for i in range(len(FILE_NAME)):
    print(FILE_NAME[i])
    print("Total number of issues : " + str(issues_matrix[i][0]))
    print("Type of issues : " + str(issues_matrix[i][1]))
    print("Severity of issues : " + str(issues_matrix[i][2]))
    print("-----")