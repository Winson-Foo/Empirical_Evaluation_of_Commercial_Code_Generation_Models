import openai
import tiktoken
import os,glob
import shutil
import time

def openai_api(code_snippet, new_file):
    try:
        openai.api_key = ("API Key")
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "How can I improve the maintainability of this codebase? Provide me the refactored code. \n" + code_snippet}
        ]
        )
        new_file.write(str(response.choices[0].message.content + "\n"))
        # new_file.write("Original ChatGPT response: \n")
        # new_file.write(str(response))
        new_file.write("\n")
        print(response.choices[0].message.content + "\n")
        # print("Original ChatGPT response: \n")
        # print(response)
        print("\n")
    except:
       time.sleep(20)
       openai_api(code_snippet, new_file)


def tokeniser(file, model_name = "gpt-3.5-turbo"):
    """
    Function to calculate the number of tokens for each file (Model name can be changed)
    """
    enc = tiktoken.encoding_for_model(model_name)
    token_count = len(enc.encode(file))
    return token_count

def file_filer():
  # Process of Filtering data, renaming and moving the selected data into Selected_files (Check if its python file and less than 2k tokens)
  for root, dirs, files in os.walk(r"C:\Users\Sylv3r\Documents\FIT4701\FYP\Dataset"):
      if not files:
          continue
      prefix = os.path.basename(root)
      for f in files:
          # Check if its python file
          if f.endswith(".py"):
              # Check if meet token requirement
              with open(os.path.join(root, f), 'r') as filename:
                text = filename.read()
                if tokeniser(text) <= 2000:
                  # Close the file
                  filename.close()
                  # Move into Selected_files folder
                  shutil.copy(os.path.join(root, f), r"C:\Users\Sylv3r\Documents\FIT4701\FYP\Selected_files")
                  # Rename the file format:(foldername_filename)
                  new_root = r"C:\Users\Sylv3r\Documents\FIT4701\FYP\Selected_files"
                  os.rename(os.path.join(new_root, f), os.path.join(new_root, "{}_{}".format(prefix, f)))

def code_evaluation():
  # Process to run the selected files through chatgpt
  folder_path = r'C:\Users\Sylv3r\Documents\FIT4701\FYP\Selected_files'
  refactored_files = []
  new_file = open("evaluation_4.txt", "a")
  # Get every python file in the folder
  for filename in glob.glob(os.path.join(folder_path, '*.py')):
    # Read the file
    with open(filename, 'r') as f:
      text = f.read()
      # Append to know which file has been passed through
      refactored_files.append(os.path.split(filename)[1])
      new_file.write(str(os.path.split(filename)[1]) + "\n")
      # How many iterations
      for i in range(1):
        new_file.write(str("Response: " + str(i + 1) + "\n"))
        print("Response: " + str(i + 1) + "\n")
        openai_api(text, new_file)
    f.close()
  print(refactored_files)
  new_file.close()


# file_filer()
code_evaluation()