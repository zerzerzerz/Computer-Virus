import os

commit_string = "init"
not_add = ['results', 'data', 'weights']
for item in os.listdir():
    if item in not_add:
        # print(item)
        continue
    else:
        os.system(f"git add {item}")
os.system(f'git commit -m "{commit_string}"')
os.system("git push origin main")