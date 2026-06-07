import os
import time
import traceback
import uuid
from datetime import datetime, timedelta

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
from zhipuai import ZhipuAI

import rag_core
from db import init_db, get_db

load_dotenv()

app = Flask(__name__)
CORS(app)

# ====== 配置 ======
SECRET_KEY = os.environ.get("JWT_SECRET", "CHANGE_THIS_TO_A_RANDOM_SECRET")
UPLOAD_DIR = "uploads"
TOKEN_EXPIRE_HOURS = 24
ZHIPUAI_API_KEY = os.environ.get("ZHIPUAI_API_KEY")

os.makedirs(UPLOAD_DIR, exist_ok=True)
client = ZhipuAI(api_key=ZHIPUAI_API_KEY)

# ====== 初始化数据库 ======
init_db()

# ====== 工具函数 ======
def make_token(username):
    payload = {
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth.replace("Bearer ", "")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["username"]
    except Exception:
        return None

def auth_required():
    username = verify_token()
    if not username:
        return None, jsonify({"msg": "未登录或 token 无效"}), 401
    return username, None, None

# ====== Auth ======
@app.post("/register")
def register():
    data = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"msg": "用户名或密码不能为空"}), 400

    password_hash = generate_password_hash(password)

    with get_db() as db:
        try:
            db.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, password_hash, int(time.time()))
            )
        except Exception:
            return jsonify({"msg": "用户已存在"}), 400

    return jsonify({"msg": "注册成功"})

@app.post("/login")
def login():
    data = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    with get_db() as db:
        row = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if not row or not check_password_hash(row["password_hash"], password):
            return jsonify({"msg": "用户名或密码错误"}), 400

    token = make_token(username)
    return jsonify({"token": token, "username": username})

# ====== Documents ======
@app.get("/documents")
def list_documents():
    username, err, code = auth_required()
    if err:
        return err, code

    # 分页参数
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", 10))
    offset = (page - 1) * page_size

    with get_db() as db:
        total = db.execute("SELECT COUNT(*) as cnt FROM documents").fetchone()["cnt"]
        rows = db.execute(
            "SELECT * FROM documents ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (page_size, offset)
        ).fetchall()

    docs = [dict(r) for r in rows]
    return jsonify({
        "documents": docs,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total
        }
    })

@app.post("/upload")
def upload():
    username, err, code = auth_required()
    if err:
        return err, code

    if 'file' not in request.files:
        return jsonify({"status": "error", "msg": "没有找到文件"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "msg": "文件名为空"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.pdf']:
        return jsonify({"status": "error", "msg": "只支持 PDF 文件"}), 400

    doc_id = str(uuid.uuid4())
    filename = f"{doc_id}{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)

    try:
        abs_path = os.path.abspath(file_path)
        rag_core.process_pdf(abs_path)

        with get_db() as db:
            db.execute(
                "INSERT INTO documents (id, name, path, created_at) VALUES (?, ?, ?, ?)",
                (doc_id, file.filename, abs_path, int(time.time()))
            )

        return jsonify({"status": "success", "msg": "知识库构建完成"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "msg": f"解析失败: {str(e)}"}), 500

@app.delete("/document/<doc_id>")
def delete_document(doc_id):
    username, err, code = auth_required()
    if err:
        return err, code

    with get_db() as db:
        row = db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            return jsonify({"msg": "文档不存在"}), 404

        try:
            if os.path.exists(row["path"]):
                os.remove(row["path"])
        except Exception:
            pass

        db.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    return jsonify({"msg": "删除成功"})

# ====== Sessions ======
@app.get("/sessions")
def sessions():
    username, err, code = auth_required()
    if err:
        return err, code

    with get_db() as db:
        rows = db.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()

    return jsonify({"sessions": [dict(r) for r in rows]})

@app.post("/session")
def session():
    username, err, code = auth_required()
    if err:
        return err, code

    session_id = str(uuid.uuid4())
    with get_db() as db:
        db.execute(
            "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
            (session_id, int(time.time()))
        )

    return jsonify({"session": {"id": session_id, "created_at": int(time.time())}})

@app.delete("/session/<session_id>")
def delete_session(session_id):
    username, err, code = auth_required()
    if err:
        return err, code

    with get_db() as db:
        db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        db.execute("DELETE FROM messages WHERE session_id=?", (session_id,))

    return jsonify({"msg": "删除成功"})

@app.get("/session/<session_id>/history")
def history(session_id):
    username, err, code = auth_required()
    if err:
        return err, code

    with get_db() as db:
        rows = db.execute(
            "SELECT role, content, time FROM messages WHERE session_id=? ORDER BY time ASC",
            (session_id,)
        ).fetchall()

    messages = []
    for r in rows:
        messages.append({
            "type": r["role"],
            "text": r["content"],
            "time": datetime.fromtimestamp(r["time"]).strftime("%H:%M:%S")
        })

    return jsonify({"messages": messages})

# ====== Chat ======
@app.post("/chat")
def chat():
    username, err, code = auth_required()
    if err:
        return err, code

    data = request.json or {}
    session_id = data.get("session_id")
    user_msg = data.get("message", "")

    if not session_id:
        return jsonify({"msg": "缺少 session_id"}), 400
    if not user_msg:
        return jsonify({"status": "error", "msg": "提问不能为空"}), 400

    try:
        print(f"收到用户提问: {user_msg}")

        # ===== RAG 检索 =====
        context, sources = rag_core.retrieve_with_sources(user_msg, k=5)
        print("context:",context)
        if context:
            system_prompt = f"""你是一个幽默、专业的AI助手。
以下是知识库检索到的资料，请优先参考：
{context}
"""
        else:
            system_prompt = "你是一个幽默、专业的AI助手。"

        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ]
        )

        ai_reply = response.choices[0].message.content
        print(ai_reply)

        with get_db() as db:
            db.execute(
                "INSERT INTO messages (session_id, role, content, time) VALUES (?, ?, ?, ?)",
                (session_id, "user", user_msg, int(time.time()))
            )
            db.execute(
                "INSERT INTO messages (session_id, role, content, time) VALUES (?, ?, ?, ?)",
                (session_id, "assistant", ai_reply, int(time.time()))
            )

        return jsonify({
            "status": "success",
            "reply": ai_reply,
            "sources": sources
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "msg": f"AI罢工了: {str(e)}"
        }), 500


# ====== 运行 ======
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)