import subprocess
import chardet
import sys

sys.path.append('C:\\Users\\DELL')
sys.path.append('C:\\Users\\DELL\\Desktop\\')


class AES(object):
    def __init__(self, maxlen, src_filename, dst_filename):
        self.maxlen = maxlen
        self.src_filename = src_filename
        self.dst_filename = dst_filename

    def decrypt(self):
        command = "java -jar AiCmnTika.jar"
        arg0 = self.maxlen
        arg1 = self.src_filename
        arg2 = self.dst_filename
        cmd = [command, str(arg0), arg1, arg2]
        new_cmd = " ".join(cmd)
        subprocess.Popen(new_cmd)
        return


if __name__ == '__main__':
    maxlen = 1000000
    src_filename = 'C:\\Users\\DELL\\Desktop\\2018.07.16周工作进度.docx'
    dst_filename = 'test.txt'
    AES = AES(maxlen, src_filename, dst_filename)
    AES.decrypt()

subprocess.call('java -jar AiCmnTika.jar chinese_stopwords test.txt', shell=True)
# sys.path.append('C:\\Users\\DELL')
#
# command = "java -jar C:\\Users\\DELLAiCmnTika.jar"
# arg0 = maxlen
# arg1 = src_filename
# arg2 = dst_filename
# cmd = [command, str(arg0), arg1, arg2]
# new_cmd = " ".join(cmd)
# stdout, stderr = subprocess.Popen(new_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
# encoding = chardet.detect(stdout)["encoding"]
# result = stdout.decode(encoding)
