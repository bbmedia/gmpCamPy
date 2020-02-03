import pymysql.cursors
import pymysql


class DataSaverSQL:

    def __init__(self, host, user, password, db):
        self.host = host
        self.user = user
        self.pw = password
        self.db_name = db

    def connect(self):
        self.connection = pymysql.connect(host=self.host,
                             user=self.user,
                             password=self.pw,
                             db=self.db_name,
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

    def count_up(self, num, is_demo):

        ok = False

        try:
            with self.connection.cursor() as cursor:
                # Create a new record
                sql = "INSERT INTO cam_tbar_counter (tb_counted, tb_is_demo) VALUES (%s, %s);"
                cursor.execute(sql, (int(num), is_demo))

            # connection is not autocommit by default. So you must commit to save
            # your changes.
            self.connection.commit()
            ok = True
        except pymysql.err.OperationalError:
            print("error executing sql query!")

        return ok

    def is_connected(self):
        return self.connection.open()
