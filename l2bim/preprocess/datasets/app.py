from flask import Flask, request, jsonify
import mysql.connector
import psycopg2
from datetime import datetime

app = Flask(__name__)

# ====== 配置区：根据你真实情况改一下 ======
# 中心库 MySQL（放 sync_conflict 表）
CENTER_MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "Root1234!",
    "database": "sync_center"
}

# 业务库 A：MySQL，biz_db_a
MYSQL_BIZ_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "Root1234!",
    "database": "biz_db_a"
}

# 业务库 B：PostgreSQL，biz_db_pg
PG_BIZ_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "user": "pguser",
    "password": "pg123456",
    "dbname": "biz_db_pg"
}

# 简单登录账号
LOGIN_USER = "admin"
LOGIN_PASS = "123456"
# ===========================================


def get_center_conn():
    return mysql.connector.connect(**CENTER_MYSQL_CONFIG)


def get_mysql_biz_conn():
    return mysql.connector.connect(**MYSQL_BIZ_CONFIG)


def get_pg_biz_conn():
    return psycopg2.connect(**PG_BIZ_CONFIG)


# -------- 1. 登录接口 --------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = data.get("username")
    password = data.get("password")

    if username == LOGIN_USER and password == LOGIN_PASS:
        # Demo 版直接返回一个假 token
        return jsonify({"token": "demo-token"}), 200
    else:
        return jsonify({"error": "用户名或密码错误"}), 401


# -------- 2. 冲突列表接口 --------
@app.route("/conflicts", methods=["GET"])
def list_conflicts():
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("pageSize", 10))
    offset = (page - 1) * page_size

    conn = get_center_conn()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT COUNT(*) FROM sync_conflict")
    total = cursor.fetchone()["COUNT(*)"]

    cursor.execute(
        """
        SELECT id, table_name, pk_value, conflict_type,
               status, created_at, resolved_at
        FROM sync_conflict
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
        """,
        (page_size, offset),
    )
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return jsonify({
        "total": total,
        "page": page,
        "pageSize": page_size,
        "items": rows
    })


# -------- 3. 冲突详情接口 --------
@app.route("/conflicts/<int:conflict_id>", methods=["GET"])
def get_conflict(conflict_id):
    conn = get_center_conn()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM sync_conflict WHERE id = %s",
        (conflict_id,)
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        return jsonify({"error": "not found"}), 404

    return jsonify(row)


# -------- 4. 冲突处理接口 --------
@app.route("/conflicts/<int:conflict_id>/resolve", methods=["POST"])
def resolve_conflict(conflict_id):
    data = request.get_json() or {}
    take = data.get("take")  # 'db_a' or 'db_b'

    if take not in ("db_a", "db_b"):
        return jsonify({"error": "take 必须是 'db_a' 或 'db_b'"}), 400

    # 1. 取出冲突记录
    center_conn = get_center_conn()
    center_cursor = center_conn.cursor(dictionary=True)
    center_cursor.execute(
        "SELECT * FROM sync_conflict WHERE id = %s",
        (conflict_id,)
    )
    conflict = center_cursor.fetchone()

    if not conflict:
        center_cursor.close()
        center_conn.close()
        return jsonify({"error": "not found"}), 404

    table_name = conflict["table_name"]
    pk_value = conflict["pk_value"]

    # Demo 版：只处理 product 表，并且 db_a = MySQL，db_b = PG
    if table_name != "product":
        center_cursor.close()
        center_conn.close()
        return jsonify({"error": "目前只支持 product 表的演示"}), 400

    if take == "db_a":
        # 以 MySQL 为准，更新 PG
        mysql_conn = get_mysql_biz_conn()
        mysql_cursor = mysql_conn.cursor(dictionary=True)
        mysql_cursor.execute(
            "SELECT * FROM product WHERE id = %s",
            (pk_value,)
        )
        row = mysql_cursor.fetchone()
        mysql_cursor.close()
        mysql_conn.close()

        if not row:
            center_cursor.close()
            center_conn.close()
            return jsonify({"error": "MySQL 中找不到这条记录"}), 400

        # upsert 到 PG
        pg_conn = get_pg_biz_conn()
        pg_cur = pg_conn.cursor()
        pg_cur.execute("""
            INSERT INTO product (id, sku, name, category, price, stock, status, created_at, updated_at, version)
            VALUES (%(id)s, %(sku)s, %(name)s, %(category)s, %(price)s, %(stock)s, %(status)s, %(created_at)s, %(updated_at)s, %(version)s)
            ON CONFLICT (id) DO UPDATE SET
                sku = EXCLUDED.sku,
                name = EXCLUDED.name,
                category = EXCLUDED.category,
                price = EXCLUDED.price,
                stock = EXCLUDED.stock,
                status = EXCLUDED.status,
                created_at = EXCLUDED.created_at,
                updated_at = EXCLUDED.updated_at,
                version = EXCLUDED.version;
        """, row)
        pg_conn.commit()
        pg_cur.close()
        pg_conn.close()

    else:  # take == 'db_b'
        # 以 PG 为准，更新 MySQL
        pg_conn = get_pg_biz_conn()
        pg_cur = pg_conn.cursor()
        pg_cur.execute("SELECT * FROM product WHERE id = %s", (pk_value,))
        cols = [desc[0] for desc in pg_cur.description]
        row_vals = pg_cur.fetchone()
        pg_conn.commit()
        pg_cur.close()
        pg_conn.close()

        if not row_vals:
            center_cursor.close()
            center_conn.close()
            return jsonify({"error": "PostgreSQL 中找不到这条记录"}), 400

        row = dict(zip(cols, row_vals))

        mysql_conn = get_mysql_biz_conn()
        mysql_cursor = mysql_conn.cursor()
        mysql_cursor.execute("""
            INSERT INTO product (id, sku, name, category, price, stock, status, created_at, updated_at, version)
            VALUES (%(id)s, %(sku)s, %(name)s, %(category)s, %(price)s, %(stock)s, %(status)s, %(created_at)s, %(updated_at)s, %(version)s)
            ON DUPLICATE KEY UPDATE
                sku = VALUES(sku),
                name = VALUES(name),
                category = VALUES(category),
                price = VALUES(price),
                stock = VALUES(stock),
                status = VALUES(status),
                created_at = VALUES(created_at),
                updated_at = VALUES(updated_at),
                version = VALUES(version);
        """, row)
        mysql_conn.commit()
        mysql_cursor.close()
        mysql_conn.close()

    # 3. 标记冲突已解决
    center_cursor.execute(
        "UPDATE sync_conflict SET status = 1, resolved_at = %s WHERE id = %s",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), conflict_id)
    )
    center_conn.commit()
    center_cursor.close()
    center_conn.close()

    return jsonify({"message": "已按 %s 解决冲突" % take})


if __name__ == "__main__":
    # 开发模式运行
    app.run(host="0.0.0.0", port=5000, debug=True)
