import openai
import tiktoken
import os,glob
import shutil
import time

def openai_api(code_snippet, txt_copy, iteration, file_name):
    try:
      openai.api_key = ""
      response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
          {"role": "user", "content": "How can I improve the maintainability of this codebase? Provide me the refactored code. \n" + code_snippet}
        ]
      )
      txt_copy.write(str(response.choices[0].message.content + "\n"))
      txt_copy.write("Original ChatGPT response: \n")
      txt_copy.write(str(response))
      txt_copy.write("\n")
      print(response.choices[0].message.content + "\n")
      print("Original ChatGPT response: \n")
      print(response)
      print("\n")
      #Write into python files based on iterations
      python_file_name = str(file_name)
      new_file = open(python_file_name, "a")
      new_file.write(str(response.choices[0].message.content))
      new_file.close()
      # Move into folders to be evaluated by SonarCube
      detination_file = "C:\\Users\\tan weijin\\Desktop\\FYP\\After_Refactor_" + str(iteration) + "\\" + python_file_name
      source_file = "C:\\Users\\tan weijin\\Desktop\\FYP\\" + python_file_name
      shutil.move(source_file, detination_file)
    except:
      time.sleep(20)
      openai_api(code_snippet, txt_copy, iteration, file_name)



def tokeniser(file, model_name = "gpt-3.5-turbo"):
    """
    Function to calculate the number of tokens for each file (Model name can be changed)
    """
    enc = tiktoken.encoding_for_model(model_name)
    token_count = len(enc.encode(file))
    return token_count

def file_filter():
  # Process of Filtering data, renaming and moving the selected data into Selected_files (Check if its python file and less than 2k tokens)
  for root, dirs, files in os.walk(r"C:\Users\tan weijin\Desktop\FYP\Dataset"):
      prefix = ""
      if not files:
          prefix += os.path.basename(root)
          continue
      for f in files:
          # Check if its python file
          if f.endswith(".py"):
              # Check if meet token requirement
              with open(os.path.join(root, f), 'r') as filename:
                try:
                  if prefix == "":
                     file_prefix = os.path.basename(root)
                  else:
                     file_prefix = os.path.join(root, "{}_{}".format(prefix, os.path.basename(root)))
                  prefix += os.path.basename(root)
                  text = filename.read()
                  if tokeniser(text) <= 2000:
                    # Close the file
                    filename.close()
                    # Rename the file format:(foldername_filename)
                    os.rename(os.path.join(root, f), os.path.join(root, "{}_{}".format(file_prefix, f)))
                    # Move into Selected_files folder
                    shutil.move(os.path.join(root, "{}_{}".format(file_prefix, f)), r"C:\Users\tan weijin\Desktop\FYP\Before_Refactor")
                    prefix = ""
                except:
                   prefix = ""
                   filename.close()


def code_evaluation():
  # Process to run the selected files through chatgpt
  folder_path = r'C:\Users\tan weijin\Desktop\FYP\Before_Refactor'
  refactored_files = []
  new_file = open("evaluation.txt", "a")
  # Get every python file in the folder
  for filename in glob.glob(os.path.join(folder_path, '*.py')):
    # Read the file
    with open(filename, 'r') as f:
      text = f.read()
      # Append to know which file has been passed through
      refactored_files.append(os.path.split(filename)[1])
      new_file.write(str(os.path.split(filename)[1]) + "\n")
      # How many iterations
      for i in range(3):
        new_file.write(str("Response: " + str(i + 1) + "\n"))
        print("Response: " + str(i + 1) + "\n")
        openai_api(text, new_file, i+1, os.path.split(filename)[1])
    f.close()
  print(refactored_files)
  new_file.close()

#file_filter()
code_evaluation()