<script setup>
import { ref } from 'vue';
import { ElMessage } from 'element-plus'
import axios from 'axios';

const messages = ref([
  { type: 'assistant', text: '我是助手assistant,哈哈哈西内！！！' },
  { type: 'user', text: '我是用户user,哈哈哈西内！！！' },
]);

const handleExceed = () => {
  ElMessage.warning(`只能上传一个文件哦 OAO`);
};
const input = ref('');
const sendMessage = async () => {
  // 1. 如果输入为空，直接返回不处理
  if (!input.value.trim()) return;

  const userText = input.value; // 把用户输入存到局部变量

  // 2. 把用户的提问 push 到消息列表
  messages.value.push({
    type: 'user',
    text: userText
  });

 
  input.value = '';

  messages.value.push({
    type: 'assistant',
    text: 'AI 正在思考中...'
  });

  try {
   
    const res = await axios.post('http://localhost:5000/chat', {
      message: userText
    });

    if (res.data.status === 'success') {
      messages.value[messages.value.length - 1].text = res.data.reply;
    } else {
      messages.value[messages.value.length - 1].text = '哎呀，出错啦：' + res.data.msg;
    }
  } catch (error) {
    
    console.error(error);
    messages.value[messages.value.length - 1].text = '网络错误或服务器未启动，请检查后端状态！';
  }
};
</script>

<template>
  <div class="container">
    <h2>demo</h2>
    <div class="upload">
      <el-upload
        class="upload-demo"
        action="http://localhost:5000/upload"
        drag
        :limit="1"
        :on-exceed="handleExceed"
        accept=".pdf"
      >
      <el-button size="large" type="primary">请上传一个PDF哦 OVO</el-button>
      </el-upload>
    </div>
      <div class="content">
        <div class="chat">
          <div v-for="(message, index) in messages" :key="index" :class="['message', message.type]">
            <div class="bubble">
              {{ message.text }}
            </div>
          </div>
        </div>
        <div class="input_box">
          <input v-model="input" @keyup.enter="sendMessage" />
          <button @click="sendMessage">发送</button>
        </div>
      </div>
    </div>

  
</template>

<style scoped>
.container {
  max-width: 900px;
  margin: auto;
  padding: 20px;
}

.title {
  text-align: center;
  margin-bottom: 20px;
}

.main {
  display: flex;
  gap: 20px;
}


.upload {
  width: 100%;
}


.content {
  flex: 1;
  display: flex;
  flex-direction: column;
  height: 500px;
  border: 1px solid #ddd;
  border-radius: 8px;
}


.chat {
  flex: 1;
  overflow-y: auto;
  padding: 15px;
  background: #fafafa;
}


.message {
  display: flex;
  margin-bottom: 10px;
}


.message.assistant {
  justify-content: flex-start;
}


.message.user {
  justify-content: flex-end;
}


.bubble {
  max-width: 60%;
  padding: 10px 14px;
  border-radius: 10px;
  background: #e5e5ea;
}


.message.user .bubble {
  background: #409eff;
  color: white;
}


.input_box {
  display: flex;
  border-top: 1px solid #ddd;
}

.input_box input {
  flex: 1;
  padding: 10px;
  border: none;
  outline: none;
}

.input_box button {
  padding: 0 20px;
  border: none;
  background: #409eff;
  color: white;
}
</style>
