import os

label_list = ['확진자수','완치자수','사망여부','집단감염','백신관련','방역지침','경제지원','마스크','국제기구','병원관련']

os.system('echo cross_entropy >> multi_log.log')
for lab in label_list:
    os.system('python run_one_task_multi.py %s >> multi_log.log' % lab)

os.system('echo focal_loss >> multi_log.log')
for lab in label_list:
    os.system('python run_one_task_multi_focal.py %s >> multi_log.log' % lab)
