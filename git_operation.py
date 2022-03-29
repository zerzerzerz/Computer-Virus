import os

commit_string = "选择data的前多少个维度参与训练"
not_add = ['results', 'data', 'weights']
for item in os.listdir():
    if item in not_add:
        # print(item)
        continue
    else:
        os.system(f"git add {item}")
os.system(f'git commit -m "{commit_string}"')
os.system("git push origin main")