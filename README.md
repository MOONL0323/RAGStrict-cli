# RAGStrict

本地文档知识库,支持语义搜索和AI对话

## 这是什么

一个命令行工具,让你的文档变成可搜索的知识库:
- 上传文档到本地数据库
- 智能语义搜索(不是关键词匹配)
- AI对话功能(基于文档内容回答问题)
- 支持MCP协议,可被Claude Desktop调用

## 安装

```bash
git clone https://github.com/MOONL0323/RAGStrict.git
cd RAGStrict
pip install -e .
```

安装后配置PATH:
```powershell
# Windows
.\setup_path.ps1
```

验证: `rags version`

## 快速使用

```bash
rags init                    # 初始化
rags add document.txt        # 添加文档
rags search "你的问题"       # 搜索
rags list                    # 查看文档
```

## 配置说明

### 基础配置

文件位置: `.ragstrict/.env`

```bash
# 本地模型
EMBEDDING_LOCAL_MODEL=sentence-transformers/all-MiniLM-L6-v2

# 分块大小
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# 离线模式(不下载模型)
OFFLINE_MODE=false
```

### API配置(可选)

需要使用chat对话功能时配置。

方式1 - 命令行(推荐):
```bash
rags config set enable_api true
rags config set llm_api_url https://your-api.com/v1/chat/completions
rags config set llm_api_key your-key
```

方式2 - 手动创建 `.ragstrict/.env.api`:
```bash
ENABLE_API=true
LLM_API_URL=https://your-api.com/v1/chat/completions
LLM_API_KEY=your-key
```

## 命令列表

```bash
rags init              # 初始化
rags add <文件>        # 添加文档
rags list              # 列出文档
rags search "查询"     # 搜索
rags show <ID>         # 查看详情
rags delete <ID>       # 删除
rags stats             # 统计
rags chat "问题"       # 对话(需API)
rags config show       # 查看配置
rags mcp               # 启动MCP服务器
rags help              # 帮助
```

## MCP服务器

启动MCP服务器后,可在Claude Desktop中调用:

```bash
rags mcp
```

Claude Desktop配置(`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "ragstrict": {
      "command": "rags",
      "args": ["mcp"]
    }
  }
}
```

## 常见问题

**Q: 找不到rags命令?**
A: 执行 `.\setup_path.ps1` 配置PATH

**Q: 模型下载慢?**
A: 设置镜像 `export HF_ENDPOINT=https://hf-mirror.com`

**Q: chat命令不工作?**
A: 需要先配置API,运行 `rags config show` 查看

## 开发贡献

```bash
# 克隆项目
git clone https://github.com/MOONL0323/RAGStrict.git
cd RAGStrict

# 安装依赖
pip install -e .

# 运行测试
pytest tests/
```

提交规范:
- `feat:` 新功能
- `fix:` 修复
- `docs:` 文档

## 许可证

MIT License

## 链接

- 项目: https://github.com/MOONL0323/RAGStrict
- 问题: https://github.com/MOONL0323/RAGStrict/issues
