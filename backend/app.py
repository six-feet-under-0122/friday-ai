
from flask import Flask, request, jsonify
from flask_cors import CORS
import main
import os
import uuid
# 假设你的核心代码写在 rag_core.py 中，里面有两个函数：
# 1. build_knowledge_base(pdf_path) -> 处理PDF存入Chroma
# 2. ask_question(query) -> 从Chroma检索并调用大模型返回结果
# 下面这两行是你实际需要引入的（这里暂时注释掉，供你参考）
# from rag_core import build_knowledge_base, ask_question

app = Flask(__name__)
CORS(app)  # 允许跨域请求，Vue前端才能连上

# 设置上传文件保存的目录
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"status": "error", "msg": "没有找到文件"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "msg": "文件名为空"}), 400
    ext = os.path.splitext(file.filename)[1]
    if file and ext  in ['.pdf', '.PDF']:
        print(file.filename)
        filename = uuid.uuid4().hex + ext
        file_path = os.path.join("uploads", filename)
        file.save(file_path)
        print("文件是否存在:", os.path.exists(file_path))
        try:
            # 4. === 把文件路径丢给你的Python处理脚本 ===
            # build_knowledge_base(file_path)
            file_path = os.path.abspath(file_path)#c存成绝对路径
            main.process_pdf(file_path)
            print(f"成功保存并处理文件: {file_path}")
            return jsonify({"status": "success", "msg": "知识库构建完成"})
        except Exception as e:
            print("file_path:", file_path)
            return jsonify({"status": "error", "msg": f"解析失败: {str(e)}"}), 500

    return jsonify({"status": "error", "msg": "只支持 PDF 文件"}), 400



if __name__ == '__main__':
    # 启动服务，运行在 5000 端口
    app.run(host='0.0.0.0', port=5000, debug=True)