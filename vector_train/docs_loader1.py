# !/usr/bin/python3
# encoding:utf-8

import pymysql

mysql_host = "172.18.90.24"
mysql_port = 3306
mysql_user = "root"
mysql_password = "zhangrui"
mysql_database = "webfiles"


class DocObj(object):
    def __init__(self):
        self.id = None
        self.title = None
        self.body = None
        self.keywords = None


class DocsLoader(object):
    def __init__(self):
        self.mysql_conn = pymysql.connect(host=mysql_host, port=mysql_port, user=mysql_user, password=mysql_password,
                                          database=mysql_database, charset='utf8')

    def load_docs(self):
        cursorObj = self.mysql_conn.cursor()
        page_size = 100
        cursor = 0
        i = 0
        with open('corpus_gongwen_20000.txt', 'w', encoding='utf-8') as f:
            while cursor >= 0:
                sql = "select id,title,body,keywords from recom_doc limit %s,%s" % (cursor, page_size)
                cursorObj.execute(sql)
                results = cursorObj.fetchall()
                for result in results:
                    docObj = DocObj()
                    docObj.id = result[0]
                    docObj.title = result[1]
                    docObj.body = result[2].decode('utf8')
                    docObj.keywords = result[3]
                    self.on_load_doc(docObj)
                    f.write('###')
                    f.write(docObj.body)
                    f.write('\n')
                    print(i)
                    i += 1
                size = cursorObj.rowcount
                cursor = -1 if (size < page_size) else (cursor + size)
        self.on_load_finish()

    def on_load_doc(self, docObj):
        # print("ID==%s\ntitle==%s\nbody==%s\nkeywords==%s" % (docObj.id,docObj.title,docObj.body,docObj.keywords))
        # print(docObj.id)
        pass

    def on_load_finish(self):
        print("#on_load_finish")
        pass


if __name__ == "__main__":
    DocsLoader().load_docs()
